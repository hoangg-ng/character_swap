#!/usr/bin/env python3
"""Merge the generated character into the ORIGINAL footage by her real silhouette.

H3 regenerates a whole masked region (rounded to 32 px cells), so the background inside the mask
differs from the source and the mask outline shows as a box/seam. Here:
  alpha  = MediaPipe person matte (hair + face + body skin + clothes) of the GENERATED frame,
           limited to the target's (dilated) person mask, so other people are never included
  bg     = source, except where the original actress was (her mask) and the new character does
           not cover it -> generated background there (no leftover actress hair)
  out    = bg * (1 - alpha) + generated * alpha
  crown  = near-black, colourless pixels inside alpha (below the letterbox) -> character hair colour
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_composite import Segmenter  # noqa: E402
from make_guide import load_seq  # noqa: E402


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
    ap.add_argument("--gen", type=Path, required=True, help="generated shot (raw, before any bg restore)")
    ap.add_argument("--src-frames", type=Path, required=True)
    ap.add_argument("--person", type=Path, required=True, help="target person masks (occluder removed)")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--limit-px", type=int, default=24, help="matte may extend this far beyond the person mask")
    ap.add_argument("--actress-px", type=int, default=8, help="dilation of the original actress area")
    ap.add_argument("--feather", type=float, default=1.5)
    args = ap.parse_args()

    gen = read_video(args.gen)
    H, W = gen[0].shape[:2]
    src = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in load_seq(args.src_frames)]
    per = [cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127 for m in load_seq(args.person, gray=True)]
    n = min(len(gen), len(src), len(per))
    row_luma = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean(1) for f in src[:n]], 0)
    letterbox = row_luma < 12
    seg = Segmenter()
    k_lim = np.ones((2 * args.limit_px + 1,) * 2, np.uint8)
    k_act = np.ones((2 * args.actress_px + 1,) * 2, np.uint8)
    with tempfile.TemporaryDirectory() as td:
        for i in range(len(gen)):
            g = gen[i]
            if i >= n or not per[i].any():
                out = src[i] if i < n else g
                cv2.imwrite(f"{td}/{i:05d}.png", out)
                continue
            import mediapipe as mp
            res = seg.seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=np.ascontiguousarray(g[..., ::-1])))
            conf = np.stack([res.confidence_masks[c].numpy_view() for c in (1, 2, 3, 4)]).max(0)
            conf = cv2.resize(conf.astype(np.float32), (W, H))
            limit = cv2.dilate(per[i].astype(np.uint8), k_lim).astype(np.float32)
            alpha = np.clip((conf - 0.25) / 0.5, 0, 1) * cv2.GaussianBlur(limit, (0, 0), 6)
            alpha[letterbox] = 0
            alpha = cv2.GaussianBlur(alpha, (0, 0), args.feather)[..., None]

            # background: source, but generated where the original actress was (no leftover actress)
            act = cv2.dilate(per[i].astype(np.uint8), k_act).astype(np.float32)
            act = cv2.GaussianBlur(act, (0, 0), 3)[..., None]
            bg = src[i].astype(np.float32) * (1 - act) + g.astype(np.float32) * act

            # crown: near-black, colourless pixels inside the character -> her hair colour
            lab = cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)
            chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
            hair = (res.confidence_masks[1].numpy_view() > 0.5)
            hair = cv2.resize(hair.astype(np.uint8), (W, H)) > 0
            red = hair & (chroma > 24) & (alpha[..., 0] > 0.5)
            gg = g.astype(np.float32)
            if red.sum() > 500:
                r_ab, r_L = lab[..., 1:][red].mean(0), lab[..., 0][red].mean()
                top = np.zeros((H, W), np.float32)
                top[: int(H * 0.45)] = 1
                dark = (alpha[..., 0] > 0.5) & (chroma < 12) & (lab[..., 0] < r_L + 10) & ~letterbox[:, None]
                w = cv2.GaussianBlur((dark * top).astype(np.float32), (0, 0), 3)[..., None]
                new = lab.copy()
                new[..., 1:] = r_ab
                new[..., 0] = np.maximum(lab[..., 0] * 0.6 + r_L * 0.4, r_L * 0.6)
                gg = cv2.cvtColor(np.clip(lab * (1 - w) + new * w, 0, 255).astype(np.uint8),
                                  cv2.COLOR_LAB2BGR).astype(np.float32)
            out = bg * (1 - alpha) + gg * alpha
            out[letterbox] = src[i][letterbox]          # letterbox bars always from the source
            cv2.imwrite(f"{td}/{i:05d}.png", np.clip(out, 0, 255).astype(np.uint8))
        seg.seg.close()
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{args.fps}",
                        "-i", f"{td}/%05d.png", "-i", str(args.gen), "-map", "0:v", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy", str(args.dst)],
                       check=True)
    print(f"silhouette merge -> {args.dst}")


if __name__ == "__main__":
    main()
