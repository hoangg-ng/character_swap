#!/usr/bin/env python3
"""Build the depth + face-mesh guide video used as <Video 1> in --guide depth-mesh mode.

Inside the (expanded) edit mask the original person is replaced by
  * a Depth Anything V2 depth map (body shape / silhouette / motion), normalised over the
    whole clip so it does not flicker, and
  * a MediaPipe FaceLandmarker mesh (head pose, gaze, eyes / brows / lips = expression)
    of the face that lies inside the tight person mask.
Everything outside the mask stays the original frame.

Usage: python make_guide.py --frames DIR --tight DIR --expanded DIR --out guide.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

MODELS_BASE = Path("/common/users/hn315/checkpoints/char_swap_models")
ROOT = Path(__file__).resolve().parent
DEPTH_MODEL = MODELS_BASE / "depth/Depth-Anything-V2-Large-hf"
FACE_MODEL = MODELS_BASE / "mediapipe/face_landmarker.task"


def load_seq(d: Path, gray: bool = False) -> np.ndarray:
    files = sorted(d.glob("*.png"))
    if not files:
        raise SystemExit(f"no PNGs in {d}")
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    return np.stack([cv2.imread(str(f), flag) for f in files])  # BGR / gray uint8


@torch.no_grad()
def depth_maps(frames_bgr: np.ndarray, device: str, batch: int = 8) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
    h, w = frames_bgr.shape[1:3]
    out = []
    for i in range(0, len(frames_bgr), batch):
        rgb = [np.ascontiguousarray(f[..., ::-1]) for f in frames_bgr[i:i + batch]]
        inp = proc(images=rgb, return_tensors="pt").to(device)
        pred = model(**inp).predicted_depth                      # relative inverse depth (near = large)
        pred = torch.nn.functional.interpolate(pred[:, None], size=(h, w), mode="bicubic",
                                               align_corners=False)[:, 0]
        out.append(pred.float().cpu().numpy())
    return np.concatenate(out)


def face_meshes(frames_bgr: np.ndarray, tight: np.ndarray, fps: float):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_faces=4,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, min_tracking_confidence=0.3)
    h, w = frames_bgr.shape[1:3]
    result = []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, f in enumerate(frames_bgr):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f[..., ::-1]))
            res = lm.detect_for_video(img, int(i * 1000 / fps))
            best, best_score = None, 0.0
            for face in res.face_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in face])
                xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
                yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
                score = (tight[i][yi, xi] > 127).mean()        # fraction of landmarks on the target person
                if score > best_score:
                    best, best_score = pts, score
            result.append(best if best_score >= 0.5 else None)
    return result


def draw_mesh(img: np.ndarray, pts: np.ndarray, face_flat: bool = False) -> None:
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    face_w = np.ptp(pts[:, 0])
    t = max(1, int(round(face_w / 220)))
    if face_flat:
        # hide the source face's 3D relief (identity) - keep only expression lines on a flat face
        cv2.fillConvexPoly(img, cv2.convexHull(pts.astype(np.int32)), (110, 110, 110), cv2.LINE_AA)
    groups = [] if face_flat else [(C.FACE_LANDMARKS_TESSELATION, (150, 150, 150), max(1, t - 1))]
    groups += [  # (connections, BGR colour, thickness)
        (C.FACE_LANDMARKS_FACE_OVAL, (255, 255, 255), t),
        (C.FACE_LANDMARKS_LEFT_EYEBROW, (0, 255, 0), t + 1),
        (C.FACE_LANDMARKS_RIGHT_EYEBROW, (0, 255, 0), t + 1),
        (C.FACE_LANDMARKS_LEFT_EYE, (255, 128, 0), t + 1),
        (C.FACE_LANDMARKS_RIGHT_EYE, (255, 128, 0), t + 1),
        (C.FACE_LANDMARKS_LEFT_IRIS, (0, 255, 255), t + 1),
        (C.FACE_LANDMARKS_RIGHT_IRIS, (0, 255, 255), t + 1),
        (C.FACE_LANDMARKS_NOSE, (255, 255, 255), t),
        (C.FACE_LANDMARKS_LIPS, (0, 0, 255), t + 1),
    ]
    p = pts.astype(int)
    for conns, col, th in groups:
        for c in conns:
            a, b = (c.start, c.end) if hasattr(c, "start") else c
            if a < len(p) and b < len(p):
                cv2.line(img, tuple(p[a]), tuple(p[b]), col, th, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--tight", type=Path, required=True, help="tight person masks (PNG)")
    ap.add_argument("--expanded", type=Path, required=True, help="expanded edit masks (PNG)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--face-flat", action="store_true",
                    help="flat-fill the face, draw only expression contours (keeps identity out of the guide)")
    ap.add_argument("--full-frame", action="store_true",
                    help="ControlNet mode: whole frame becomes depth (+ mesh), not just the mask region")
    args = ap.parse_args()

    frames = load_seq(args.frames)
    tight = load_seq(args.tight, gray=True)
    exp = load_seq(args.expanded, gray=True)
    n = min(len(frames), len(tight), len(exp))
    frames, tight, exp = frames[:n], tight[:n], exp[:n]
    region = exp > 127
    if args.full_frame:
        region = np.ones_like(region)
    # keep letterbox bars untouched (black rows/cols over the whole clip)
    luma = frames.mean(axis=(0, 3))
    region &= (luma > 12)[None]
    print(f"{n} frames {frames.shape[2]}x{frames.shape[1]}, mask coverage "
          f"{region.mean() * 100:.1f}% (max {region.reshape(n, -1).mean(1).max() * 100:.1f}%)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth = depth_maps(frames, device)
    vals = depth[region] if region.any() else depth.ravel()
    lo, hi = np.percentile(vals, [2, 98])                     # clip-wide -> no flicker
    gray = (np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

    meshes = [None] * n if args.no_mesh else face_meshes(frames, tight, args.fps)
    print(f"face mesh found on {sum(m is not None for m in meshes)}/{n} frames")

    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            g = cv2.cvtColor(gray[i], cv2.COLOR_GRAY2BGR)
            if meshes[i] is not None:
                draw_mesh(g, meshes[i], args.face_flat)
            out = frames[i].copy()
            out[region[i]] = g[region[i]]
            cv2.imwrite(f"{td}/{i:05d}.png", out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{args.fps}",
                        "-i", f"{td}/%05d.png", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p",
                        str(args.out)], check=True)
    print(f"guide -> {args.out}")


if __name__ == "__main__":
    main()
