#!/usr/bin/env python3
"""Re-light generated frames with the light of the source frames (RELIGHT).

The model draws the new character with its own, usually brighter and flatter light. Here the broad light of each
source frame is transferred onto the generated frame:
  light(x)  = blurred lightness (LAB L) of the SCENE (people masked out, their area filled from the surroundings),
              sigma = `scale` x frame size: only where it is bright or dark (backlight, a stage spot), not the details
              and not the person's own skin or clothes (a fair-skinned original must not brighten the character)
  gain      = light(source) / light(generated), clamped to [lo, hi]
  result L  = generated L * gain (blended by `strength`); where the light is reduced the colour (a, b) is
              reduced by the same factor (dim light desaturates), otherwise it stays the generated one.
A backlit silhouette therefore stays a dark silhouette with the NEW character's outline, and fabric or a face that
the model lit too brightly in a dark shot gets the source's light pattern.

  python utils/relight.py --source src.png --generated gen.png --out out.png   (test on one frame)
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def light_map(lab_l: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(lab_l, (0, 0), sigma) + 4.0          # +4: no division blow-up in black areas


def people(frame: np.ndarray) -> np.ndarray:
    """Union of person silhouettes (YOLO11-seg on a 480 px copy, dilated)."""
    from utils.scene_color import people_mask
    sc = 480 / max(frame.shape[:2])
    small = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else frame
    m = people_mask(small).astype(np.uint8)
    return cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def scene_light(lab_l: np.ndarray, ppl: np.ndarray, sigma: float) -> np.ndarray:
    """Blurred lightness of the SCENE only: people masked out and their area filled from the surroundings, so a
    person's own skin and clothes (light skin, a white dress) are not taken for light."""
    w = (~ppl).astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    near = cv2.GaussianBlur(lab_l * w, (0, 0), sigma) / np.maximum(den, 1e-3)
    wide = cv2.GaussianBlur(lab_l * w, (0, 0), sigma * 4) / np.maximum(cv2.GaussianBlur(w, (0, 0), sigma * 4), 1e-3)
    return np.where(den > 0.05, near, wide) + 4.0               # large holes (close-ups): wider surroundings


def relight_frame(src: np.ndarray, gen: np.ndarray, strength: float = 1.0, scale: float = 0.03,
                  lo: float = 0.3, hi: float = 1.5, scene_only: bool = True) -> np.ndarray:
    """src and gen are BGR frames of the same size. Returns gen with the source's broad light."""
    if strength <= 0:
        return gen
    sl = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    lab = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    sigma = max(2.0, scale * max(gen.shape[:2]))
    if scene_only:
        gain = np.clip(scene_light(sl, people(src), sigma) / scene_light(lab[..., 0], people(gen), sigma), lo, hi)
    else:
        gain = np.clip(light_map(sl, sigma) / light_map(lab[..., 0], sigma), lo, hi)
    gain = 1.0 + strength * (gain - 1.0)
    lab[..., 0] = np.clip(lab[..., 0] * gain, 0, 255)
    dim = np.minimum(gain, 1.0)[..., None]                        # in dimmer light colour fades too: without this
    lab[..., 1:] = 128 + (lab[..., 1:] - 128) * dim               # a darkened red kimono reads as glowing red
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.03)
    a = ap.parse_args()
    s, g = cv2.imread(a.source), cv2.imread(a.generated)
    s = cv2.resize(s, (g.shape[1], g.shape[0]))
    cv2.imwrite(a.out, relight_frame(s, g, a.strength, a.scale))


if __name__ == "__main__":
    main()
