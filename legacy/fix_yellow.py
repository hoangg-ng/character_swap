#!/usr/bin/env python3
"""Neutralise excess yellow/orange on the character's skin (glow from makeup/relighting).

Per frame (LAB): skin reference = median a/b of the frame's non-yellow skin-like pixels; pixels where
yellow clearly dominates (b > max(--b-min, a * --ratio)) get their a/b pulled to the skin reference
and their glow (lightness) reduced a little, with a soft weight. Red hair (a >= b) is untouched.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--b-min", type=float, default=22.0, help="b* offset above which yellow is considered")
    ap.add_argument("--ratio", type=float, default=1.05, help="yellow must exceed red (a*) by this factor")
    ap.add_argument("--dim", type=float, default=6.0, help="lightness reduction at full weight")
    ap.add_argument("--orange-margin", type=float, default=10.0, help="chroma above skin before correcting")
    ap.add_argument("--orange-L", type=float, default=115.0, help="only bright pixels (excludes red hair)")
    args = ap.parse_args()

    cap, frames = cv2.VideoCapture(str(args.video)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
            L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
            skin = (L > 150) & (a > 4) & (a < 30) & (b > 0) & (b < 22) & (b <= a * 1.6)
            if skin.sum() > 500:
                ref_a, ref_b = float(np.median(a[skin])), float(np.median(b[skin]))
            else:
                ref_a, ref_b = 12.0, 12.0
            # (1) yellow clearly dominating red
            excess_y = b - np.maximum(args.b_min, a * args.ratio)
            # (2) orange glow: much more saturated than her skin, orange-to-yellow hue, and bright
            #     (the darker red hair/bangs are excluded by lightness and hue)
            skin_c = float(np.hypot(ref_a, ref_b))
            chroma = np.hypot(a, b)
            excess_o = (chroma - (skin_c + args.orange_margin)) * (b > 0.55 * a) * (L > args.orange_L)
            w = np.clip(np.maximum(excess_y / 16.0, excess_o / 22.0), 0, 1) * (L > 50)
            w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 7)
            out = lab.copy()
            out[..., 1] = lab[..., 1] * (1 - w) + (ref_a + 128) * w
            out[..., 2] = lab[..., 2] * (1 - w) + (ref_b + 128) * w
            out[..., 0] = L - args.dim * w
            res = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            cv2.imwrite(f"{td}/{i:05d}.png", res)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{args.fps}",
                        "-i", f"{td}/%05d.png", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                        str(args.dst)], check=True)
    print(f"yellow neutralised -> {args.dst}")


if __name__ == "__main__":
    main()
