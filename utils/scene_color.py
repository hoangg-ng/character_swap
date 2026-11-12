#!/usr/bin/env python3
"""Reference colour from the scene only (REF_COLOR_SCENE=1).

The default reference recolouring (ComfyUI ColorTransfer, reinhard_lab, "uniform" stats) takes the colour
statistics from the whole frames of the shot, so the clothes of the person being replaced count as lighting:
a red dress turns the new character's skin red. Here the same Reinhard LAB transfer is computed in Python,
but the target statistics come only from pixels outside every person (YOLO11-seg silhouettes, dilated).

  scene_stats(video)                    -> (mean, std) of LAB over the scene without people
  recolor(image, stats, strength, out)  -> the image moved to those statistics, like ColorTransfer reinhard_lab

CLI (preview):  python utils/scene_color.py --video shot.mp4 --image ref.png --strength 1.0 --out ref_scene.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import tone_match as tm  # noqa: E402  (shares the YOLO11-seg model)


def _lab(bgr: np.ndarray) -> np.ndarray:
    """CIELAB as float (L 0-100), the same space as kornia's rgb_to_lab used by ColorTransfer."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def people_mask(frame: np.ndarray, conf: float = 0.3, grow: float = 0.02) -> np.ndarray:
    """Union of all person silhouettes in the frame, dilated by `grow` x the frame height."""
    r = tm.model().predict(frame, classes=[0], conf=conf, verbose=False, device="cpu", retina_masks=True)[0]
    m = np.zeros(frame.shape[:2], bool)
    if r.masks is not None:
        for d in r.masks.data.cpu().numpy():
            d = d > 0.5
            if d.shape != m.shape:
                d = cv2.resize(d.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            m |= d
    k = max(3, int(frame.shape[0] * grow) | 1)
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def scene_stats(video: Path, max_frames: int = 24, min_scene: float = 0.15) -> tuple[np.ndarray, np.ndarray, float]:
    """LAB mean/std pooled over up to `max_frames` evenly spaced frames, people excluded. A frame where the scene
    is under `min_scene` of the picture (a close-up filled by the person) falls back to the whole frame.
    Returns (mean[3], std[3], fraction of pixels that were scene)."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = set(np.linspace(0, n - 1, min(max_frames, n)).astype(int).tolist())
    px, kept, total = [], 0, 0
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        if i not in idx:
            continue
        sc = 384 / max(f.shape[:2])
        f = cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        scene = ~people_mask(f)
        if scene.mean() < min_scene:
            scene[:] = True
        lab = _lab(f)
        px.append(lab[scene]); kept += int(scene.sum()); total += scene.size
    allpx = np.concatenate(px)
    return allpx.mean(0), np.maximum(allpx.std(0), 1e-3), kept / max(total, 1)


def face_pixels(bgr: np.ndarray) -> np.ndarray | None:
    """LAB pixels of the central face region (cheeks, nose, forehead) of every face found; None if no face."""
    from utils.face_refine import _face_app
    px = []
    for f in _face_app().get(bgr):
        x0, y0, x1, y1 = f.bbox.astype(int); w, h = x1 - x0, y1 - y0
        c = bgr[max(0, y0 + int(h * .25)):y0 + int(h * .75), max(0, x0 + int(w * .2)):x0 + int(w * .8)]
        if c.size:
            px.append(_lab(c).reshape(-1, 3))
    return np.concatenate(px) if px else None


def skin_stats(video: Path, max_frames: int = 24):
    """LAB mean/std of the face skin of the people in the shot (the light falling on a face). None if no face."""
    cap = cv2.VideoCapture(str(video)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = set(np.linspace(0, n - 1, min(max_frames, n)).astype(int).tolist()); px = []
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            p = face_pixels(f)
            if p is not None:
                px.append(p)
    if not px:
        return None
    a = np.concatenate(px)
    return a.mean(0), np.maximum(a.std(0), 1e-3), 1.0


def recolor(image: Path, stats: tuple, strength: float, out: Path, scale_clip: tuple = (0.8, 1.25),
            ab_factor: float = 1.0, src_stats: tuple | None = None) -> Path:
    """Reinhard LAB transfer of the whole image to `stats` (source stats = the whole image, as ColorTransfer's
    uniform mode with one image), blended in LAB by `strength`. The per-channel std ratio is clamped to
    `scale_clip` so a low-contrast scene does not flatten the reference."""
    bgr = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    alpha = bgr[:, :, 3:] if bgr.ndim == 3 and bgr.shape[2] == 4 else None
    bgr = bgr[:, :, :3]
    lab = _lab(bgr)
    flat = lab.reshape(-1, 3)
    if src_stats is not None:                      # e.g. the character's face skin (skin -> skin transfer)
        s_mean, s_std = src_stats
    else:
        s_mean, s_std = flat.mean(0), np.maximum(flat.std(0), 1e-3)
    r_mean, r_std = stats[0], stats[1]
    # a soft background (bokeh) has a small std: unclamped, the reference would be flattened to grey
    scale = np.clip(r_std / s_std, *scale_clip)
    corrected = (lab - s_mean) * scale + r_mean
    k = np.array([strength, strength * ab_factor, strength * ab_factor], np.float32)   # L, a, b
    res = lab + k * (corrected - lab)
    rgb = cv2.cvtColor(res.astype(np.float32), cv2.COLOR_LAB2BGR)
    img = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    if alpha is not None:
        img = np.concatenate([img, alpha], 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    st = scene_stats(a.video)
    print(f"scene LAB mean {np.round(st[0], 1)} std {np.round(st[1], 1)} (scene = {st[2]:.0%} of sampled pixels)")
    recolor(a.image, st, a.strength, a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
