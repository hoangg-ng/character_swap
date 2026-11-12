#!/usr/bin/env python3
"""Outfit reference from the video, for a character sheet that wears the video's clothes (KEEP_OUTFIT=1).

Picks the frame where the person is shown largest and most complete (YOLO11-pose: shoulders and hips visible,
facing the camera, knees visible if possible), cuts her out with the YOLO11-seg silhouette onto white and
removes the head (cut between nose and shoulders), because a visible face in the outfit image makes
Qwen-Image draw a second person.

  python utils/make_outfit_ref.py --video cases/x/input/video.mp4 --out cases/x/input/outfit_ref_nohead.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import tone_match as tm  # noqa: E402  (YOLO11-seg)

KP_NOSE, KP_LSH, KP_RSH, KP_LHIP, KP_RHIP, KP_LKNEE, KP_RKNEE = 0, 5, 6, 11, 12, 13, 14


def best_frame(video: Path, every: int = 3):
    from ultralytics import YOLO
    pose = YOLO(str(ROOT / "ComfyUI" / "models" / "ultralytics" / "person" / "yolo11m-pose.pt"))
    cap = cv2.VideoCapture(str(video)); best, i = None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            r = pose.predict(f, conf=0.4, verbose=False, device="cpu")[0]
            if r.keypoints is not None and len(r.boxes):
                for b, kp in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()):
                    c = kp[:, 2]
                    if min(c[KP_LSH], c[KP_RSH], c[KP_LHIP], c[KP_RHIP]) < 0.5 or c[KP_NOSE] < 0.5:
                        continue                       # need shoulders, hips and a frontal face
                    h = (b[3] - b[1]) / f.shape[0]
                    if b[1] < 2 or b[3] > f.shape[0] - 2:
                        h *= 0.8                       # touching the frame edge: probably cut off
                    score = h * (1.3 if min(c[KP_LKNEE], c[KP_RKNEE]) > 0.5 else 1.0)
                    if best is None or score > best[0]:
                        best = (score, i, f.copy(), b, kp)
        i += 1
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    best = best_frame(a.video)
    if best is None:
        sys.exit("no frame with a complete, frontal person found")
    _, idx, f, box, kp = best
    mask, _ = tm.person_mask(f)
    if mask is None:
        sys.exit("no person silhouette in the chosen frame")
    sh_y = (kp[KP_LSH, 1] + kp[KP_RSH, 1]) / 2
    cut = int(kp[KP_NOSE, 1] + 0.6 * (sh_y - kp[KP_NOSE, 1]))      # below the chin, above the shoulders
    img = np.full_like(f, 255)
    img[mask] = f[mask]
    img[:cut] = 255
    ys, xs = np.where(mask[cut:]); ys = ys + cut
    pad = 20
    y0, y1 = max(0, cut - pad), min(f.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(f.shape[1], xs.max() + pad)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), img[y0:y1, x0:x1])
    print(f"outfit reference from frame {idx} -> {a.out}")


if __name__ == "__main__":
    main()
