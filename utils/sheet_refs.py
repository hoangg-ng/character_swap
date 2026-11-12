#!/usr/bin/env python3
"""Assign the panels of a character sheet to the reference roles of run_swap.sh.

The grid sheet's layout is chosen by the model (3x2, 1x4, ...), so the roles are found by content:
a panel whose face fills >= 25% of its height (or is >= 1.6x the median panel's) is a face close-up (-> face references), the others are
body views; the most complete frontal body view (frontal face, smallest face) is the main reference.
Prints shell lines CHAR_REF, CHAR_REF_BACK (a body panel without a face), CHAR_REF_EXTRA, CHAR_REF_FACE.

  python utils/sheet_refs.py --sheet cases/x/input/sheet_outfit [--face-photo cases/x/input/face.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--face-photo", type=Path, default=None)
    a = ap.parse_args()
    from utils.face_refine import _face_app
    app = _face_app()
    panels = sorted(a.sheet.glob("view_panel*.png"), key=lambda p: int(p.stem.replace("view_panel", "")))
    if not panels:
        sys.exit(f"no view_panel*.png in {a.sheet}")
    sizes, frontal = [], {}
    for p in panels:
        im = cv2.imread(str(p)); fs = app.get(im)
        f = max(fs, key=lambda f: f.bbox[3] - f.bbox[1]) if fs else None
        sizes.append(((f.bbox[3] - f.bbox[1]) / im.shape[0] if f is not None else 0.0, p))
        if f is not None:      # frontal: nose near the middle of the eyes
            le, re_, nose = f.kps[0], f.kps[1], f.kps[2]
            frontal[p] = abs(nose[0] - (le[0] + re_[0]) / 2) / max(abs(re_[0] - le[0]), 1) < 0.2
    med = sorted(fh for fh, _ in sizes)[len(sizes) // 2]
    body, face = [], []
    for fh, p in sizes:      # face close-up: face >= 25% of the panel, or clearly larger than in the other panels
        (face if fh >= 0.25 or (med > 0 and fh >= 1.6 * med) else body).append((fh, p))
    if not body:                                   # only close-ups: use them all as body views too
        body = face
    # main reference = the most complete frontal body view: frontal face, smallest face (= most of the body shown);
    # a waist-up panel would otherwise win and pull every shot toward a half-body framing
    front = [t for t in body if frontal.get(t[1]) and t[0] > 0]
    main_ref = (min(front, key=lambda t: t[0]) if front else max(body, key=lambda t: t[0]))[1]
    extra = [p for _, p in body if p != main_ref]
    faces = [p for _, p in face]
    if (a.sheet / "view_face.png").exists():
        faces.append(a.sheet / "view_face.png")
    if a.face_photo:
        faces.append(a.face_photo)
    back = [p for fh, p in body if fh == 0 and p != main_ref]       # no face found: the back view
    print(f"CHAR_REF={main_ref}")
    print(f"CHAR_REF_BACK={back[0] if back else ''}")
    print('CHAR_REF_EXTRA="' + " ".join(map(str, extra)) + '"')
    print('CHAR_REF_FACE="' + " ".join(map(str, faces)) + '"')


if __name__ == "__main__":
    main()
