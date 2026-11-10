#!/usr/bin/env python3
"""Post-fix: recolour leftover dark (black, low-chroma) hair to the red hair of the same frame.

In each frame: MediaPipe hair mask -> split into 'red' hair (chroma high) and 'dark' hair
(low chroma); dark hair pixels get the red hair's mean a/b and a lifted lightness, blended by how
dark/neutral they are. Only above the eyes by default (the crown band at the frame edge).
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--top-frac", type=float, default=0.45, help="only fix the top part of the frame")
    ap.add_argument("--chroma", type=float, default=12.0, help="chroma below this counts as 'dark' hair")
    args = ap.parse_args()

    cap, frames = cv2.VideoCapture(str(args.video)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    seg = Segmenter()
    H, W = frames[0].shape[:2]
    top = np.zeros((H, W), np.float32)
    top[: int(H * args.top_frac)] = 1
    top = cv2.GaussianBlur(top, (0, 0), 20)
    # letterbox bars: rows that are near-black across the whole clip are never touched
    row_luma = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean(1) for f in frames], 0)
    lb = row_luma < 12
    letterbox_free = np.ones((H, W), bool)
    letterbox_free[lb] = False
    letterbox_free = cv2.erode(letterbox_free.astype(np.uint8), np.ones((7, 1), np.uint8)) > 0
    fixed = 0
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            out = f
            hp = seg.hair_prob(f)
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
            chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
            hair = hp > 0.5
            red = hair & (chroma > 2 * args.chroma)
            if red.sum() > 500:
                r_ab = lab[..., 1:][red].mean(0)
                r_L = lab[..., 0][red].mean()
                # also catch dark pixels just above the detected hair (crown fades out of the hair mask)
                hair_zone = cv2.dilate(hair.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
                dark = hair_zone & (chroma < args.chroma) & (lab[..., 0] < r_L + 10) & (lab[..., 0] > 18)
                dark &= letterbox_free
                w = np.clip((args.chroma - chroma) / args.chroma, 0, 1) * dark * top
                w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 3)[..., None]
                if w.max() > 0.05:
                    new = lab.copy()
                    new[..., 1:] = r_ab
                    new[..., 0] = lab[..., 0] * 0.6 + r_L * 0.4
                    lab2 = lab * (1 - w) + new * w
                    out = cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                    fixed += 1
            cv2.imwrite(f"{td}/{i:05d}.png", out)
        seg.seg.close()
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{args.fps}",
                        "-i", f"{td}/%05d.png", "-i", str(args.video), "-map", "0:v", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy", str(args.dst)],
                       check=True)
    print(f"fixed {fixed}/{len(frames)} frames -> {args.dst}")


if __name__ == "__main__":
    main()
