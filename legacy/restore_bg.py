#!/usr/bin/env python3
"""Pixel-exact background restore after generation.

H3 regenerates the edit mask rounded up to 32 px DiT cells and unioned over 4-frame latent spans,
so the background around the person is regenerated too (blocky halo, slight drift). This puts the
SOURCE back everywhere outside the person's own SAM mask (+ a small margin, feathered).

Usage: python restore_bg.py --video gen.mp4 --src-frames DIR --tight DIR --dst out.mp4 [--margin 10]
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
from make_guide import load_seq  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--src-frames", type=Path, required=True)
    ap.add_argument("--tight", type=Path, required=True, help="person masks (occluder already subtracted)")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--margin", type=int, default=10, help="px kept around the person (hair edges, motion)")
    ap.add_argument("--feather", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    cap, gen = cv2.VideoCapture(str(args.video)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        gen.append(f)
    H, W = gen[0].shape[:2]
    src = load_seq(args.src_frames)
    tight = load_seq(args.tight, gray=True)
    n = min(len(gen), len(src), len(tight))
    k = np.ones((2 * args.margin + 1, 2 * args.margin + 1), np.uint8)
    row_luma = np.mean([cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2GRAY).mean(1) for f in src[:n]], 0)
    letterbox = row_luma < 12
    with tempfile.TemporaryDirectory() as td:
        for i in range(len(gen)):
            g = gen[i].astype(np.float32)
            if i < n:
                s = cv2.resize(src[i], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
                m = cv2.resize(tight[i], (W, H), interpolation=cv2.INTER_NEAREST) > 127
                m = cv2.dilate(m.astype(np.uint8), k) if args.margin > 0 else m.astype(np.uint8)
                a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), args.feather)[..., None]
                g = s * (1 - a) + g * a
                g[letterbox] = s[letterbox]                 # letterbox bars always from the source
            cv2.imwrite(f"{td}/{i:05d}.png", np.clip(g, 0, 255).astype(np.uint8))
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{args.fps}",
               "-i", f"{td}/%05d.png", "-i", str(args.video), "-map", "0:v", "-map", "1:a?",
               "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy", str(args.dst)]
        subprocess.run(cmd, check=True)
    print(f"background restored -> {args.dst}")


if __name__ == "__main__":
    main()
