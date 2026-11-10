#!/usr/bin/env python3
"""Post-relight a swapped close-up so the new face sits in the shot's light.

Per frame (output and source are index-aligned, same timeline):
  face  (MediaPipe face hull, feathered):
     L  = high-freq(L_out)  +  low-freq(L_src) * relight + low-freq(L_out) * (1 - relight)
          -> the actress's shading/exposure (light direction) with our character's features
     ab = shifted toward the actress's mean skin chroma by --skin
  person (source tight mask, feathered):
     chroma * --sat, + --tint toward the scene light (white/bright pixels of the source frame),
     contrast * --contrast
  softness: blur the person region until its sharpness (Laplacian variance) matches the source
  edges: wide feather of the person mask so hair/face edges blend into the background
Outside the person mask the generated frame is untouched.
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
from make_guide import face_meshes, load_seq  # noqa: E402


def read_video(p: Path):
    cap, fr = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return np.stack(fr)


def hull_mask(pts, shape, feather):
    m = np.zeros(shape, np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 1.0, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), feather)


def sharpness(gray, m):
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap[m].var()) if m.sum() > 100 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="generated shot (e.g. final_720p.mp4)")
    ap.add_argument("--src-frames", type=Path, required=True, help="prepared source frames (guide/save_frames)")
    ap.add_argument("--src-tight", type=Path, required=True, help="target person masks (guide/save_tight)")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--relight", type=float, default=0.85, help="0..1 take the source's face shading")
    ap.add_argument("--skin", type=float, default=0.6, help="0..1 move skin chroma to the actress's")
    ap.add_argument("--sat", type=float, default=0.8, help="chroma scale for hair/clothes")
    ap.add_argument("--tint", type=float, default=0.35)
    ap.add_argument("--contrast", type=float, default=0.9)
    ap.add_argument("--edge-feather", type=float, default=6.0)
    ap.add_argument("--occluder-masks", type=Path, default=None,
                    help="masks of the foreground occluder (same timeline); its out-of-focus edge is "
                         "re-blended softly from the source instead of the hard mask cut")
    ap.add_argument("--occluder-soft", type=float, default=7.0, help="feather (px) of the occluder edge")
    args = ap.parse_args()

    out = read_video(args.video)
    n, H, W = out.shape[:3]
    src = load_seq(args.src_frames)[:n]
    tight = load_seq(args.src_tight, gray=True)[:n]
    src = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in src])
    tight = np.stack([cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in tight])

    occ = None
    if args.occluder_masks:
        occ = load_seq(args.occluder_masks, gray=True)[:n]
        occ = np.stack([cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in occ])
    m_src = face_meshes(src, tight, args.fps)
    m_out = face_meshes(out, tight, args.fps)
    print(f"{n} frames; face found src {sum(m is not None for m in m_src)}, out {sum(m is not None for m in m_out)}")

    # scene light colour from bright pixels of the source frames (whole clip)
    labs = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in src]
    bright = np.concatenate([l[..., 1:][l[..., 0] > np.percentile(l[..., 0], 92)] for l in labs])
    light = bright.mean(0)

    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            o = out[i]
            lo = cv2.cvtColor(o, cv2.COLOR_BGR2LAB).astype(np.float32)
            ls = labs[i]
            g = lo.copy()
            person = tight[i] > 127
            pts_o = m_out[i] if m_out[i] is not None else m_src[i]
            if pts_o is not None and m_src[i] is not None:
                face_w = float(np.ptp(pts_o[:, 0]))
                fm = hull_mask(pts_o, (H, W), max(4.0, face_w / 20))
                fm *= cv2.GaussianBlur(person.astype(np.float32), (0, 0), 4)   # never outside the person
                sig = max(6.0, face_w / 7)                                      # "lighting" scale
                lo_low = cv2.GaussianBlur(lo[..., 0], (0, 0), sig)
                ls_low = cv2.GaussianBlur(ls[..., 0], (0, 0), sig)
                relit = lo[..., 0] - lo_low + (ls_low * args.relight + lo_low * (1 - args.relight))
                g[..., 0] = lo[..., 0] * (1 - fm) + relit * fm
                sm = hull_mask(m_src[i], (H, W), 2) > 0.5
                om = fm > 0.5
                if sm.sum() > 100 and om.sum() > 100:
                    # match skin chroma mean AND spread to the actress (warm, natural variation)
                    cs, co = ls[..., 1:][sm], lo[..., 1:][om]
                    matched = (lo[..., 1:] - co.mean(0)) * (cs.std(0) / np.maximum(co.std(0), 1e-3)) + cs.mean(0)
                    g[..., 1:] = lo[..., 1:] * (1 - args.skin * fm[..., None]) + matched * args.skin * fm[..., None]
            else:
                fm = np.zeros((H, W), np.float32)
            # whole person: mute + scene tint + softer contrast (face chroma already handled)
            pm = cv2.GaussianBlur(person.astype(np.float32), (0, 0), args.edge_feather)
            rest = (pm * (1 - fm))[..., None]
            ab = 128 + (g[..., 1:] - 128) * args.sat + args.tint * (light - 128)
            g[..., 1:] = g[..., 1:] * (1 - rest) + ab * rest
            if person.sum() > 100:
                mean = g[..., 0][person].mean()
                L = (g[..., 0] - mean) * args.contrast + mean
                g[..., 0] = g[..., 0] * (1 - pm) + L * pm
            res = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
            # softness: blur until sharpness matches the source inside the person
            if person.sum() > 100:
                target = sharpness(cv2.cvtColor(src[i], cv2.COLOR_BGR2GRAY), person)
                best = res
                for s in (0.6, 0.9, 1.2, 1.6, 2.0, 2.5):
                    cand = cv2.GaussianBlur(res, (0, 0), s)
                    if sharpness(cv2.cvtColor(cand.astype(np.uint8), cv2.COLOR_BGR2GRAY), person) < target:
                        break
                    best = cand
                res = best
            final = o.astype(np.float32) * (1 - pm[..., None]) + res * pm[..., None]
            if occ is not None and (occ[i] > 127).any():
                # out-of-focus occluder overlaps softly: re-blend the source occluder with a feathered edge
                om_ = cv2.dilate((occ[i] > 127).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
                a = cv2.GaussianBlur(om_, (0, 0), args.occluder_soft)[..., None]
                final = final * (1 - a) + src[i].astype(np.float32) * a
            cv2.imwrite(f"{td}/{i:05d}.png", np.clip(final, 0, 255).astype(np.uint8))
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{args.fps}",
                        "-i", f"{td}/%05d.png", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                        str(args.dst)], check=True)
    print(f"relit -> {args.dst}")


if __name__ == "__main__":
    main()
