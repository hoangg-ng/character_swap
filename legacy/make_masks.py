#!/usr/bin/env python3
"""Occlusion-aware masks for the swap (replaces the workflow's internal masks).

Inputs (stage-A outputs, prepared 24 fps frames):
  --frames   prepared frames
  --raw      SAM target masks before occluder subtraction (H3RelayRecoverMasks output)
  --occ      SAM occluder masks (e.g. "man"), optional
Builds, per frame:
  face   : face hull from a sensitive per-frame MediaPipe detection (catches partly hidden faces)
  near   : pixels much nearer than the target (Depth Anything) -> blurry foreground SAM misses
  occ    : (SAM occluder U near) ; core = occ eroded by --occ-erode (only the opaque part is kept
           from the source; the semi-transparent defocus edge is regenerated)
  person : (raw U face) - core           -> composite / guide / background restore
  edit   : dilate(person, --expand) - core -> H3 edit mask
Writes PNG dirs person/, edit/, occ/ and edit.mp4 (lossless) for ComfyUI.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_guide import FACE_MODEL, depth_maps, load_seq  # noqa: E402


def face_hulls(frames: np.ndarray, raw: np.ndarray, conf: float) -> list:
    """Per-frame IMAGE-mode detection (no tracking state) with low thresholds; keep the face that
    overlaps the target's SAM mask."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
                                        running_mode=vision.RunningMode.IMAGE, num_faces=4,
                                        min_face_detection_confidence=conf, min_face_presence_confidence=conf)
    H, W = frames.shape[1:3]
    out = []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for f, m in zip(frames, raw):
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f[..., ::-1])))
            best, score = None, 0.0
            for face in res.face_landmarks:
                pts = np.array([[p.x * W, p.y * H] for p in face], np.float32)
                xi = np.clip(pts[:, 0].astype(int), 0, W - 1)
                yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
                s = (m[yi, xi] > 127).mean()
                if s > score:
                    best, score = pts, s
            out.append(best if score >= 0.3 else None)
    return out


def write_video(masks: np.ndarray, path: Path, fps: float) -> None:
    with tempfile.TemporaryDirectory() as td:
        for i, m in enumerate(masks):
            cv2.imwrite(f"{td}/{i:05d}.png", cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{fps}",
                        "-i", f"{td}/%05d.png", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
                        str(path)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--occ", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expand", type=int, default=12)
    ap.add_argument("--occ-erode", type=int, default=14, help="px; keep only the opaque core of the occluder")
    ap.add_argument("--min-width", type=int, default=15,
                    help="drop person-mask parts thinner than 2x this (stale-mask rings around a full occluder)")
    ap.add_argument("--near", type=float, default=0.15,
                    help="depth margin: pixel is 'in front' if its relative depth exceeds the target's by this "
                         "fraction of the range (0 = off)")
    ap.add_argument("--face-conf", type=float, default=0.2)
    ap.add_argument("--extra", type=Path, default=None,
                    help="extra region to include in the person mask (e.g. pasted reference hair)")
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    frames = load_seq(args.frames)
    raw = load_seq(args.raw, gray=True)[:len(frames)]
    n, H, W = frames.shape[:3]
    occ = (load_seq(args.occ, gray=True)[:n] > 127) if args.occ else np.zeros((n, H, W), bool)
    rawm = raw > 127
    if args.extra:
        ex = load_seq(args.extra, gray=True)[:n]
        rawm = rawm | np.stack([cv2.resize(e, (W, H), interpolation=cv2.INTER_NEAREST) > 127 for e in ex])

    faces = face_hulls(frames, raw, args.face_conf)
    facem = np.zeros((n, H, W), np.uint8)
    for i, p in enumerate(faces):
        if p is not None:
            cv2.fillConvexPoly(facem[i], cv2.convexHull(p[:468].astype(np.int32)), 1)
    print(f"{n} frames; face found {sum(p is not None for p in faces)}/{n}")

    near = np.zeros((n, H, W), bool)
    if args.near > 0:
        d = depth_maps(frames, "cuda" if torch.cuda.is_available() else "cpu")
        # target depth: median inside the face (else the SAM mask) on frames without occluders
        ref = [np.median(d[i][facem[i] > 0]) for i in range(n) if facem[i].any() and not occ[i].any()]
        if not ref:
            ref = [np.median(d[i][rawm[i]]) for i in range(n) if rawm[i].any()]
        t_ref = float(np.median(ref))
        span = float(np.percentile(d, 99.5) - np.percentile(d, 0.5))
        near = d > t_ref + args.near * span
        near = np.stack([cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
                         for m in near])
        print(f"depth: target {t_ref:.2f}, threshold {t_ref + args.near * span:.2f}; "
              f"near-occluder frames {int(near.reshape(n, -1).any(1).sum())}")

    # never let the edit region reach into letterbox bars (the model would paint the head into them)
    row_luma = frames.mean(axis=(0, 2, 3))
    content = np.ones((H, W), bool)
    content[row_luma < 12] = False
    k_er = np.ones((2 * args.occ_erode + 1,) * 2, np.uint8)
    k_ex = np.ones((2 * args.expand + 1,) * 2, np.uint8)
    person, edit, occ_core = [], [], []
    for i in range(n):
        o = (occ[i] | near[i]).astype(np.uint8)
        core = cv2.erode(o, k_er) if args.occ_erode > 0 else o
        p = ((rawm[i] | (facem[i] > 0)) & ~(core > 0)).astype(np.uint8)
        if args.min_width > 0 and p.any():
            k_open = np.ones((2 * args.min_width + 1,) * 2, np.uint8)
            p = cv2.morphologyEx(p, cv2.MORPH_OPEN, k_open)
        e = (cv2.dilate(p, k_ex) & ~core.astype(bool)).astype(np.uint8) if p.any() else p
        p = p * content; e = e * content
        person.append(p * 255); edit.append(e * 255); occ_core.append(core * 255)
    for name, arr in (("person", person), ("edit", edit), ("occ", occ_core)):
        d_ = args.out / name
        d_.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(arr):
            cv2.imwrite(str(d_ / f"{i:05d}.png"), m)
    write_video(np.stack(edit), args.out / "edit.mp4", args.fps)
    print(f"masks -> {args.out}  (person/, edit/, occ/, edit.mp4)")


if __name__ == "__main__":
    main()
