#!/usr/bin/env python3
"""Match the generated person's tone to the original person, frame by frame.

BFS sometimes lights the new character brighter / more saturated than the guide (e.g. a dark, low-key shot where
the original is in shadow). For each frame: YOLO11-seg segments the person in the source frame and in the generated
frame; inside the generated person's silhouette the LAB channels are shifted/scaled to the source person's
statistics (Reinhard transfer), blended by --strength through the feathered silhouette mask. Only the person
changes - no rectangle around her.

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
        _MODEL = YOLO(str(ROOT / "ComfyUI" / "models" / "ultralytics" / "person" / "yolo11m-seg.pt"))
    return _MODEL


def person_mask(frame: np.ndarray, conf: float = 0.3):
    """(mask, box) of the largest person: mask is a bool HxW silhouette (YOLO11-seg), box is [x0, y0, x1, y1]."""
    r = model().predict(frame, classes=[0], conf=conf, verbose=False, device="cpu", retina_masks=True)[0]
    if r.masks is None or not len(r.boxes):
        return None, None
    boxes = r.boxes.xyxy.tolist()
    k = max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    m = r.masks.data[k].cpu().numpy() > 0.5
    if m.shape != frame.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m, [int(v) for v in boxes[k]]


def person_box(frame: np.ndarray, conf: float = 0.3):
    return person_mask(frame, conf)[1]


def tone_frame(src: np.ndarray, gen: np.ndarray, strength: float, feather: float = 0.012) -> np.ndarray:
    """src and gen have the same size. Returns gen with its person toned like the source person."""
    ms_, _ = person_mask(src)
    mg_, bg = person_mask(gen)
    if ms_ is None or mg_ is None or ms_.sum() < 50 or mg_.sum() < 50:
        return gen
    ls = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)[ms_].astype(np.float32)
    lab = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    lg = lab[mg_]
    ms, ss, mg, sg = ls.mean(0), ls.std(0) + 1e-3, lg.mean(0), lg.std(0) + 1e-3
    toned = (lab - mg) / sg * ss + ms
    h, w = gen.shape[:2]
    r = max(1, int(feather * max(h, w)))                    # small grow + feather around her silhouette
    m = cv2.dilate(mg_.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), r)[..., None] * strength
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
