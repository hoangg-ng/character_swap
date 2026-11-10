#!/usr/bin/env python3
"""Grade the swapped character so she sits in the scene's light (post-process, mask-only).

Inside the (feathered) mask of the swapped character, per frame in LAB:
  * luminance mean/std pulled toward the ORIGINAL actress's luminance in the same frame
    (same light falling on the same spot), blended by --lum
  * chroma desaturated by --sat and shifted toward the scene light colour, measured from bright
    (white) fabric inside the source actress's mask (her white jacket = neutral reflector)
  * slight softening (--soften) to match the scene's focus
Outside the mask the frame is untouched.

Usage:
  python color_match.py --video OUT.mp4 --out-masks DIR --src-frames DIR --src-masks DIR --dst graded.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def seq(d: Path, gray=False):
    files = sorted(d.glob("*.png"))
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR) for f in files]


def read_video(p: Path):
    cap, fr = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="swapped result to grade")
    ap.add_argument("--out-masks", type=Path, required=True, help="masks of the swapped character")
    ap.add_argument("--src-frames", type=Path, required=True)
    ap.add_argument("--src-masks", type=Path, required=True, help="masks of the original actress")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--lum", type=float, default=0.0,
                    help="0..1 match luminance to the original actress (only if the outfits have similar "
                         "brightness; a white jacket vs black top makes this wash the character out)")
    ap.add_argument("--contrast", type=float, default=0.9, help="contrast scale around the mean (<1 = softer)")
    ap.add_argument("--black-lift", type=float, default=4.0, help="raise the darkest values (L units, 0..255)")
    ap.add_argument("--sat", type=float, default=0.85, help="chroma scale (<1 desaturates)")
    ap.add_argument("--tint", type=float, default=0.35, help="0..1 shift toward the scene light colour")
    ap.add_argument("--soften", type=float, default=0.8, help="gaussian sigma px blended in (0 = off)")
    ap.add_argument("--audio-from", type=Path, default=None)
    args = ap.parse_args()

    frames = read_video(args.video)
    om = seq(args.out_masks, gray=True)
    sf = seq(args.src_frames)
    sm = seq(args.src_masks, gray=True)
    n = min(len(frames), len(om), len(sf), len(sm))
    H, W = frames[0].shape[:2]
    rs = lambda im, interp=cv2.INTER_LINEAR: cv2.resize(im, (W, H), interpolation=interp)

    # scene light colour: bright pixels inside the source actress mask (white jacket), whole clip
    ab = []
    for i in range(n):
        lab = cv2.cvtColor(rs(sf[i]), cv2.COLOR_BGR2LAB).astype(np.float32)
        m = (rs(sm[i], cv2.INTER_NEAREST) > 127) & (lab[..., 0] > np.percentile(lab[..., 0], 85))
        if m.sum() > 200:
            ab.append(lab[..., 1:][m].mean(0))
    light = np.mean(ab, 0) if ab else np.array([128.0, 128.0])
    print(f"scene light a/b offset from neutral: {light[0] - 128:+.1f}, {light[1] - 128:+.1f}")

    with tempfile.TemporaryDirectory() as td:
        for i in range(len(frames)):
            f = frames[i]
            if i >= n:
                cv2.imwrite(f"{td}/{i:05d}.png", f)
                continue
            mo = rs(om[i], cv2.INTER_NEAREST) > 127
            ms = rs(sm[i], cv2.INTER_NEAREST) > 127
            soft = cv2.GaussianBlur(mo.astype(np.float32), (0, 0), 3)[..., None]
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
            g = lab.copy()
            if mo.sum() > 200 and ms.sum() > 200:
                ls = cv2.cvtColor(rs(sf[i]), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)[ms]
                lo = lab[..., 0][mo]
                L = (lab[..., 0] - lo.mean()) * (ls.std() / max(lo.std(), 1e-3)) + ls.mean()
                g[..., 0] = lab[..., 0] * (1 - args.lum) + L * args.lum
            if mo.sum() > 200:
                mean = g[..., 0][mo].mean()
                g[..., 0] = (g[..., 0] - mean) * args.contrast + mean
                g[..., 0] = args.black_lift + g[..., 0] * (1 - args.black_lift / 255)
            g[..., 1:] = 128 + (lab[..., 1:] - 128) * args.sat + args.tint * (light - 128)
            out = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
            if args.soften > 0:
                out = cv2.GaussianBlur(out, (0, 0), args.soften)
            res = f.astype(np.float32) * (1 - soft) + out * soft
            cv2.imwrite(f"{td}/{i:05d}.png", np.clip(res, 0, 255).astype(np.uint8))
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{args.fps}",
               "-i", f"{td}/%05d.png"]
        if args.audio_from:
            cmd += ["-i", str(args.audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(args.dst)]
        subprocess.run(cmd, check=True)
    print(f"graded -> {args.dst}")


if __name__ == "__main__":
    main()
