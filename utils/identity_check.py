#!/usr/bin/env python3
"""Per-shot identity check of a generation job (AUTO_RETRY).

For every generated shot, each generated frame is compared with the guide frame at the same index
(<job>/shotNN_in.mp4, which shows the ORIGINAL person): ArcFace similarity of the generated face to the new
character (--face-ref) and to the original person's face in that guide frame. A shot fails when its face looks
more like the original than like the character (the model kept the original face, typical for short close-ups).

  --report            print the failed shot numbers, comma-separated (empty if none)
  --keep-best 3,11    for each listed shot, score every attempt (save_*.mp4) and make the best one the newest,
                      so assembly uses it

  python utils/identity_check.py --job cases/x/output/gen_seed42 --face-ref cases/x/input/face.png --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _frames(p: Path, every: int = 2) -> list:
    cap = cv2.VideoCapture(str(p)); out = []; k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f if k % every == 0 else None); k += 1
    return out


def score(gen: Path, guide: Path, ref: np.ndarray, lead: int) -> tuple[float, float, int]:
    """(mean similarity to the character, mean similarity to the original person, frames with a face)."""
    from utils.face_refine import _face_app
    app = _face_app()
    g, s = _frames(gen), _frames(guide)
    sc, so = [], []
    for k in range(lead, min(len(g), len(s))):
        if g[k] is None or s[k] is None:
            continue
        fg = app.get(g[k])
        fo = app.get(s[k])
        if not fg or not fo:
            continue
        f = max(fg, key=lambda x: x.bbox[2] - x.bbox[0])           # the main generated face
        o = max(fo, key=lambda x: x.bbox[2] - x.bbox[0])           # the original person's face in the guide
        sc.append(float(np.dot(f.normed_embedding, ref))); so.append(float(np.dot(f.normed_embedding, o.normed_embedding)))
    if not sc:
        return 0.0, 0.0, 0
    return float(np.mean(sc)), float(np.mean(so)), len(sc)


def failed(sc: float, so: float, n: int, min_frames: int = 3) -> bool:
    return n >= min_frames and (so > sc + 0.05 or (sc < 0.15 and so > 0.30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--face-ref", type=Path, required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--keep-best", default="")
    a = ap.parse_args()
    from utils.face_refine import _face_app
    fs = _face_app().get(cv2.imread(str(a.face_ref)))
    if not fs:
        sys.exit(f"no face in {a.face_ref}")
    ref = max(fs, key=lambda f: f.det_score).normed_embedding
    tc = json.loads((a.job / "target_check.json").read_text()) if (a.job / "target_check.json").exists() else {}
    if a.report:
        bad = []
        for k, v in sorted(tc.items(), key=lambda t: int(t[0])):
            d = a.job / f"shot{int(k):02d}"
            gens = sorted(d.glob("save_*.mp4"), key=lambda p: p.stat().st_mtime) if v.get("generate") else []
            if not gens or not (a.job / f"shot{int(k):02d}_in.mp4").exists():
                continue
            lead = int((d / "lead.txt").read_text()) if (d / "lead.txt").exists() else 0
            sc, so, n = score(gens[-1], a.job / f"shot{int(k):02d}_in.mp4", ref, lead)
            bad_ = failed(sc, so, n)
            print(f"  shot {k}: character {sc:.2f} / original {so:.2f} ({n} frames)" + ("  <- FAILED" if bad_ else ""),
                  file=sys.stderr)
            if bad_:
                bad.append(k)
        print(",".join(bad))
    for k in [x for x in a.keep_best.split(",") if x]:
        d = a.job / f"shot{int(k):02d}"
        lead = int((d / "lead.txt").read_text()) if (d / "lead.txt").exists() else 0
        res = []
        for gp in sorted(d.glob("save_*.mp4")):
            sc, so, n = score(gp, a.job / f"shot{int(k):02d}_in.mp4", ref, lead)
            res.append((sc - so, gp)); print(f"  shot {k}: {gp.name} character {sc:.2f} / original {so:.2f}", file=sys.stderr)
        if res:
            best = max(res)[1]
            os.utime(best)                                         # newest = the one assembly uses
            print(f"  shot {k}: keeping {best.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
