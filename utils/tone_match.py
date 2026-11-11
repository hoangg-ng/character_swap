#!/usr/bin/env python3
"""Match the generated person's tone to the original person, frame by frame.

BFS sometimes lights the new character brighter / more saturated than the guide (e.g. a dark, low-key shot where
the original is in shadow). For each frame: YOLO11 finds the person in the source frame and in the generated
frame; inside the generated person's box the LAB channels are shifted/scaled to the source person's statistics
(Reinhard transfer), blended by --strength through a feathered box mask. Only the person region changes.

  python utils/tone_match.py --src shot_in.mp4 --gen save_*.mp4 --out toned.mp4 [--strength 0.8]
Used by assemble_bfs.py --tone-person.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # project root
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(ROOT / "ComfyUI" / "models" / "ultralytics" / "person" / "yolo11m.pt"))
    return _MODEL


def person_box(frame: np.ndarray, conf: float = 0.3):
    r = model().predict(frame, classes=[0], conf=conf, verbose=False, device="cpu")[0]
    boxes = r.boxes.xyxy.tolist()
    if not boxes:
        return None
    return [int(v) for v in max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]


def tone_frame(src: np.ndarray, gen: np.ndarray, strength: float, feather: float = 0.08) -> np.ndarray:
    """src and gen have the same size. Returns gen with its person toned like the source person."""
    bs, bg = person_box(src), person_box(gen)
    if bs is None or bg is None:
        return gen
    ls = cv2.cvtColor(src[bs[1]:bs[3], bs[0]:bs[2]], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    lg = lab[bg[1]:bg[3], bg[0]:bg[2]].reshape(-1, 3)
    ms, ss, mg, sg = ls.mean(0), ls.std(0) + 1e-3, lg.mean(0), lg.std(0) + 1e-3
    toned = (lab - mg) / sg * ss + ms
    h, w = gen.shape[:2]
    m = np.zeros((h, w), np.float32)
    m[bg[1]:bg[3], bg[0]:bg[2]] = 1.0
    k = max(3, int(feather * max(bg[2] - bg[0], bg[3] - bg[1])) | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)[..., None] * strength
    out = lab * (1 - m) + toned * m
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strength", type=float, default=0.8)
    a = ap.parse_args()
    cs, cg = cv2.VideoCapture(str(a.src)), cv2.VideoCapture(str(a.gen))
    fps = cg.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cg.get(3)), int(cg.get(4))
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok1, s = cs.read()
        ok2, g = cg.read()
        if not (ok1 and ok2):
            break
        vw.write(tone_frame(cv2.resize(s, (w, h)), g, a.strength))
    vw.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
