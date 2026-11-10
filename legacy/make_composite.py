#!/usr/bin/env python3
"""Rough per-frame composite: remove the source actress inside her mask and put the reference
character in her place (background kept). Used as <Video 1> so the model refines a picture of
OUR character instead of reading the actress's identity.

Per frame, inside the tight person mask only:
  1. actress pixels -> inpainted background (cv2 Telea)
  2. reference person (MediaPipe multiclass segmentation: hair/face/body/clothes) aligned to the
     actress's head with a similarity transform from face landmarks
  3. reference face warped triangle-by-triangle onto the actress's face mesh (expression follows
     the source), seamless-cloned for lighting
Frames without a detected face reuse the nearest detected head pose.

Usage: python make_composite.py --frames DIR --tight DIR --reference IMG --out composite.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Delaunay

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_guide import face_meshes, load_seq  # noqa: E402

MODELS_BASE = Path("/common/users/hn315/checkpoints/char_swap_models")
ROOT = Path(__file__).resolve().parent
SEG_MODEL = MODELS_BASE / "mediapipe/selfie_multiclass_256x256.tflite"
FACE_MODEL = MODELS_BASE / "mediapipe/face_landmarker.task"
# stable landmarks for the head similarity transform: eye corners, nose tip, mouth corners, chin, forehead
ANCHORS = [33, 133, 362, 263, 1, 61, 291, 152, 10]


def ref_landmarks(img_bgr: np.ndarray) -> np.ndarray:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
                                        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(img_bgr[..., ::-1])))
    if not res.face_landmarks:
        raise SystemExit("no face found in the reference image")
    h, w = img_bgr.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)


def ref_person_alpha(img_bgr: np.ndarray) -> np.ndarray:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(SEG_MODEL)),
                                        running_mode=vision.RunningMode.IMAGE, output_category_mask=True)
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(img_bgr[..., ::-1])))
    cat = res.category_mask.numpy_view()
    alpha = (cat > 0).astype(np.float32)                      # 0 = background
    alpha = cv2.GaussianBlur(alpha, (0, 0), 2.0)
    return cv2.resize(alpha, (img_bgr.shape[1], img_bgr.shape[0]))


class Segmenter:
    """MediaPipe selfie-multiclass: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others."""
    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(SEG_MODEL)),
                                            running_mode=vision.RunningMode.IMAGE,
                                            output_category_mask=True, output_confidence_masks=True)
        self.seg = vision.ImageSegmenter.create_from_options(opts)

    def probs(self, img_bgr: np.ndarray, cls: int) -> np.ndarray:
        import mediapipe as mp
        res = self.seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=np.ascontiguousarray(img_bgr[..., ::-1])))
        p = res.confidence_masks[cls].numpy_view().astype(np.float32)
        return cv2.resize(p, (img_bgr.shape[1], img_bgr.shape[0]))

    def hair_prob(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.probs(img_bgr, 1)


def hair_stats(img_bgr: np.ndarray, prob: np.ndarray):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = prob > 0.6
    if m.sum() < 200:
        raise SystemExit("no hair found in the reference image")
    v = lab[m]
    return v.mean(0), v.std(0)


def recolor_hair(frame: np.ndarray, prob: np.ndarray, ref_mean, ref_std, strength: float) -> np.ndarray:
    """Give the actress's hair the reference hair colour/brightness, keeping every strand and its motion."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = prob > 0.5
    if m.sum() < 100:
        return frame.astype(np.float32)
    v = lab[m]
    mean, std = v.mean(0), np.maximum(v.std(0), 1e-3)
    new = lab.copy()
    new[..., 0] = (lab[..., 0] - mean[0]) * (ref_std[0] / std[0]) + ref_mean[0]     # brightness + texture
    new[..., 1:] = (lab[..., 1:] - mean[1:]) * 0.5 + ref_mean[1:]                 # hue/chroma of the ref hair
    a = (np.clip((prob - 0.3) / 0.4, 0, 1) * strength)[..., None]
    out = lab * (1 - a) + new * a
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def recolor_clothes(img: np.ndarray, prob: np.ndarray, target_L: float, strength: float) -> np.ndarray:
    """Turn the actress's clothes toward the reference outfit tone (e.g. white jacket -> black top),
    keeping folds/shading (scaled) and motion."""
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    m = prob > 0.5
    if m.sum() < 100:
        return img
    mean = lab[..., 0][m].mean()
    new = lab.copy()
    new[..., 0] = (lab[..., 0] - mean) * 0.35 + target_L
    new[..., 1:] = 128 + (lab[..., 1:] - 128) * 0.3
    a = (np.clip((prob - 0.3) / 0.4, 0, 1) * strength)[..., None]
    out = lab * (1 - a) + new * a
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def warp_triangles(src: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, tris: np.ndarray,
                   out_shape) -> tuple[np.ndarray, np.ndarray]:
    h, w = out_shape
    out = np.zeros((h, w, 3), np.float32)
    cover = np.zeros((h, w), np.float32)
    for t in tris:
        s, d = src_pts[t].astype(np.float32), dst_pts[t].astype(np.float32)
        r = cv2.boundingRect(d)
        x, y, rw, rh = r
        if rw <= 0 or rh <= 0 or x >= w or y >= h or x + rw <= 0 or y + rh <= 0:
            continue
        M = cv2.getAffineTransform(s, (d - [x, y]).astype(np.float32))
        patch = cv2.warpAffine(src, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        tri_mask = np.zeros((rh, rw), np.float32)
        cv2.fillConvexPoly(tri_mask, np.int32(d - [x, y]), 1.0, cv2.LINE_AA)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + rw, w), min(y + rh, h)
        px, py = x0 - x, y0 - y
        m = tri_mask[py:py + y1 - y0, px:px + x1 - x0, None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - m) + patch[py:py + y1 - y0, px:px + x1 - x0] * m
        cover[y0:y1, x0:x1] = np.maximum(cover[y0:y1, x0:x1], m[..., 0])
    return out, cover


def fill_missing(meshes: list, allow_none: bool = False) -> list:
    idx = [i for i, m in enumerate(meshes) if m is not None]
    if not idx:
        if allow_none:          # e.g. a back view: hair/clothes recolour only, no face work
            return [(None, False)] * len(meshes)
        raise SystemExit("no face found in any frame - composite needs at least one")
    out = []
    for i in range(len(meshes)):
        j = min(idx, key=lambda k: abs(k - i))
        out.append((meshes[j], meshes[i] is not None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--tight", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--relight", type=float, default=0.0,
                    help="0..1 replace the face's large-scale shading with the actress's (light direction/exposure)")
    ap.add_argument("--skin", type=float, default=0.0,
                    help="0..1 match face skin chroma (mean+spread) to the actress's")
    ap.add_argument("--detail", type=float, default=0.0,
                    help="0..1: add the actress's high-frequency face detail (creases, tears, wet eyes = "
                         "expression) on top of the reference face; colour/identity stay the reference's")
    ap.add_argument("--alpha-out", type=Path, default=None,
                    help="write the per-frame region the composite changed (PNG dir) - used to grow the edit mask")
    ap.add_argument("--temporal", type=float, default=2.0,
                    help="temporal smoothing (frames, gaussian sigma) of landmarks, relight shading and skin "
                         "statistics -> no frame-to-frame flicker of the pasted face (0 = off)")
    ap.add_argument("--occ", type=Path, default=None,
                    help="occluder masks (in front of the target): pasted hair never covers them")
    ap.add_argument("--flow-max", type=float, default=40.0, help="paste-flow: max non-rigid hair displacement (px)")
    ap.add_argument("--hair-mode", default="paste", choices=["paste", "recolor", "paste-flow"],
                    help="paste: rigid cut-out of the reference head/hair (stiff hair motion); recolor: keep "
                         "the actress's own hair (natural motion) recoloured to the reference hair colour")
    ap.add_argument("--hair-strength", type=float, default=1.0)
    ap.add_argument("--clothes-L", type=float, default=-1,
                    help="recolour the actress's clothes to this LAB lightness (0..255, e.g. 35 = black top); "
                         "-1 = leave clothes")
    args = ap.parse_args()

    frames = load_seq(args.frames)
    tight = load_seq(args.tight, gray=True)[:len(frames)]
    frames = frames[:len(tight)]
    n, H, W = frames.shape[:3]

    ref = cv2.imread(str(args.reference))
    ref_pts = ref_landmarks(ref)[:468]
    ref_alpha = ref_person_alpha(ref)
    seg = Segmenter() if args.hair_mode == "recolor" else None
    if seg:
        ref_hair_mean, ref_hair_std = hair_stats(ref, seg.hair_prob(ref))
        print(f"reference hair LAB mean {ref_hair_mean.round(1)}")
    tris = Delaunay(ref_pts).simplices
    # face-only region of the reference (convex hull of the mesh), for the per-triangle expression warp
    meshes = face_meshes(frames, tight, args.fps)
    found = sum(m is not None for m in meshes)
    print(f"{n} frames {W}x{H}; actress face found on {found}/{n}")
    if any(m is None for m in meshes):
        # sensitive per-frame detection for partly hidden faces (no tracking state)
        from make_masks import face_hulls
        extra = face_hulls(frames, tight, 0.2)
        meshes = [m if m is not None else e for m, e in zip(meshes, extra)]
        print(f"after sensitive re-detection: face found on {sum(m is not None for m in meshes)}/{n}")
    if args.temporal > 0:
        # smooth landmarks inside each run of consecutive detections (keeps real motion, kills jitter)
        from scipy.ndimage import gaussian_filter1d
        i = 0
        while i < n:
            if meshes[i] is None:
                i += 1
                continue
            j = i
            while j < n and meshes[j] is not None:
                j += 1
            if j - i > 2:
                run = np.stack([m[:468] for m in meshes[i:j]]).astype(np.float32)
                run = gaussian_filter1d(run, args.temporal, axis=0, mode="nearest")
                for k in range(i, j):
                    meshes[k] = run[k - i]
            i = j
    poses = fill_missing(meshes, allow_none=args.hair_mode == "recolor")

    # temporally smoothed relight shading + skin statistics of the actress (per frame, then smoothed)
    s_low_all, skin_stats = None, None
    if args.relight > 0 or args.skin > 0:
        from scipy.ndimage import gaussian_filter1d
        widths = [float(np.ptp(p[0][:468, 0])) for p in poses if p[0] is not None]
        sig_l = max(6.0, (np.median(widths) if widths else 300) / 7)
        labs_src = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in frames]
        s_low_all = np.stack([cv2.GaussianBlur(l[..., 0], (0, 0), sig_l) for l in labs_src])
        if args.temporal > 0:
            s_low_all = gaussian_filter1d(s_low_all, args.temporal, axis=0, mode="nearest")
        skin_ids0 = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
                     397, 288, 361, 323, 454, 352, 346, 347, 348, 349, 350, 6, 121, 120, 119, 118, 117, 123]
        st = np.full((n, 4), np.nan, np.float32)
        for k in range(n):
            if poses[k][0] is None or not poses[k][1]:
                continue
            pk = poses[k][0][:468]
            sm = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(sm, cv2.convexHull(pk[skin_ids0].astype(np.int32)), 1)
            v = labs_src[k][..., 1:][sm > 0]
            if len(v) > 100:
                st[k] = [*v.mean(0), *v.std(0)]
        ok = ~np.isnan(st[:, 0])
        if ok.any():
            idx = np.arange(n)
            for c in range(4):
                st[:, c] = np.interp(idx, idx[ok], st[ok, c])
            if args.temporal > 0:
                st = gaussian_filter1d(st, args.temporal * 2, axis=0, mode="nearest")
            skin_stats = st
        del labs_src

    if args.alpha_out:
        args.alpha_out.mkdir(parents=True, exist_ok=True)
    occ = None
    if args.occ:
        occ = load_seq(args.occ, gray=True)[:n]
        occ = np.stack([cv2.resize(o, (W, H), interpolation=cv2.INTER_NEAREST) for o in occ])
    flow_ref = None
    if args.hair_mode == "paste-flow":
        # key frame: first frame with a detected face and the largest visible target area
        cand = [i for i in range(n) if poses[i][1]]
        k0 = max(cand[:max(1, len(cand) // 3)], key=lambda i: (tight[i] > 127).sum()) if cand else 0
        pk = poses[k0][0][:468].astype(np.float32)
        Mk, _ = cv2.estimateAffinePartial2D(ref_pts[ANCHORS], pk[ANCHORS], method=cv2.LMEDS)
        flow_ref = dict(k=k0, M=Mk,
                        layer=cv2.warpAffine(ref, Mk, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                        alpha=cv2.warpAffine(ref_alpha, Mk, (W, H), flags=cv2.INTER_LINEAR),
                        gray=cv2.cvtColor(frames[k0], cv2.COLOR_BGR2GRAY))
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        print(f"paste-flow key frame {k0}")

    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            frame = frames[i]
            m_t = (tight[i] > 127).astype(np.uint8)
            pts, detected = poses[i]
            pts = pts[:468].astype(np.float32) if pts is not None else None
            region_extra = None
            if flow_ref is not None and pts is not None:
                # reference hair, moved by the actress's own motion (optical flow frame i -> key frame)
                Mi, _ = cv2.estimateAffinePartial2D(ref_pts[ANCHORS], pts[ANCHORS], method=cv2.LMEDS)
                A = np.vstack([flow_ref["M"], [0, 0, 1]]) @ np.linalg.inv(np.vstack([Mi, [0, 0, 1]]))
                Rx = A[0, 0] * gx + A[0, 1] * gy + A[0, 2] - gx          # rigid head motion i -> k0
                Ry = A[1, 0] * gx + A[1, 1] * gy + A[1, 2] - gy
                F = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow_ref["gray"],
                                                 None, 0.5, 4, 31, 3, 7, 1.5, 0)
                Dx, Dy = F[..., 0] - Rx, F[..., 1] - Ry                  # non-rigid part (hair sway)
                valid = ((np.hypot(Dx, Dy) < args.flow_max) & (m_t > 0)).astype(np.float32)
                wv = cv2.GaussianBlur(valid, (0, 0), 10)
                Dx = cv2.GaussianBlur(Dx * valid, (0, 0), 10) / np.maximum(wv, 1e-3)
                Dy = cv2.GaussianBlur(Dy * valid, (0, 0), 10) / np.maximum(wv, 1e-3)
                conf = np.clip(wv * 2, 0, 1)
                mx = (gx + Rx + Dx * conf).astype(np.float32); my = (gy + Ry + Dy * conf).astype(np.float32)
                layer = cv2.remap(flow_ref["layer"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                alpha = cv2.remap(flow_ref["alpha"], mx, my, cv2.INTER_LINEAR, borderValue=0)
                if occ is not None:          # never paint over whatever is in front of her
                    o = cv2.dilate((occ[i] > 127).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32)
                    alpha = alpha * (1 - cv2.GaussianBlur(o, (0, 0), 3))
                alpha = alpha[..., None]
                hole = cv2.dilate(m_t, np.ones((9, 9), np.uint8))
                small = 4
                bg = cv2.inpaint(cv2.resize(frame, (W // small, H // small)),
                                 cv2.resize(hole, (W // small, H // small), interpolation=cv2.INTER_NEAREST),
                                 5, cv2.INPAINT_TELEA)
                bg = np.where(hole[..., None] > 0, cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC), frame)
                comp = bg.astype(np.float32) * (1 - alpha) + layer.astype(np.float32) * alpha
                region_extra = (alpha[..., 0] > 0.05).astype(np.uint8)
            elif seg is not None:
                # natural motion: the actress's own hair, recoloured; nothing rigid is pasted
                prob = seg.hair_prob(frame) * cv2.GaussianBlur(m_t.astype(np.float32), (0, 0), 2)
                comp = recolor_hair(frame, prob, ref_hair_mean, ref_hair_std, args.hair_strength)
                if args.clothes_L >= 0:
                    cp = seg.probs(frame, 4) * cv2.GaussianBlur(m_t.astype(np.float32), (0, 0), 2)
                    comp = recolor_clothes(comp, cp, args.clothes_L, 1.0)
            else:
                comp = None
            # 1. remove the actress: inpaint the background under a slightly dilated mask
            hole = cv2.dilate(m_t, np.ones((9, 9), np.uint8))
            small = 4
            bg = cv2.inpaint(cv2.resize(frame, (W // small, H // small)),
                             cv2.resize(hole, (W // small, H // small), interpolation=cv2.INTER_NEAREST),
                             5, cv2.INPAINT_TELEA)
            bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
            bg = np.where(hole[..., None] > 0, bg, frame)

            if comp is None:
                # 2. reference person aligned with a similarity transform on stable landmarks
                M, _ = cv2.estimateAffinePartial2D(ref_pts[ANCHORS], pts[ANCHORS], method=cv2.LMEDS)
                person = cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                alpha = cv2.warpAffine(ref_alpha, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
                comp = bg.astype(np.float32) * (1 - alpha) + person.astype(np.float32) * alpha

            # 3. expression: per-triangle warp of the reference face onto the actress mesh
            if detected:
                face, cover = warp_triangles(ref.astype(np.float32), ref_pts, pts, tris, (H, W))
                cover = cv2.erode(cover, np.ones((5, 5), np.uint8))
                fm = (cover > 0.5).astype(np.uint8) * 255
                if fm.any():
                    x, y, w, h = cv2.boundingRect(fm)
                    center = (x + w // 2, y + h // 2)
                    try:
                        comp = cv2.seamlessClone(np.clip(face, 0, 255).astype(np.uint8),
                                                 np.clip(comp, 0, 255).astype(np.uint8), fm, center,
                                                 cv2.NORMAL_CLONE).astype(np.float32)
                    except cv2.error:
                        comp = comp * (1 - cover[..., None]) + face * cover[..., None]

            # expression detail + lighting match inside the face (frequency separation)
            if detected and args.detail > 0:
                # central features only (brows, eyes, nose, mouth, chin): the actress's hair at the
                # face edges and forehead would otherwise ghost through
                feat = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300,          # brows
                        33, 133, 159, 145, 362, 263, 386, 374,                  # eyes
                        1, 98, 327, 61, 291, 0, 17, 181, 405, 152, 148, 377]    # nose, mouth, chin
                face_w = float(np.ptp(pts[:, 0]))
                feat_m = np.zeros((H, W), np.float32)
                cv2.fillConvexPoly(feat_m, cv2.convexHull(pts[feat].astype(np.int32)), 1.0, cv2.LINE_AA)
                feat_m = cv2.GaussianBlur(feat_m, (0, 0), max(3.0, face_w / 18))[..., None]  # wide feather
                full_m = np.zeros((H, W), np.float32)
                cv2.fillConvexPoly(full_m, cv2.convexHull(pts.astype(np.int32)), 1.0, cv2.LINE_AA)
                full_m = cv2.GaussianBlur(full_m, (0, 0), max(3.0, face_w / 25))
                sig = max(2.0, face_w / 45)
                src = frame.astype(np.float32)
                detail = src - cv2.GaussianBlur(src, (0, 0), sig)
                # match luminance mean/std of the whole composite face to the actress face (ref chroma kept)
                m = full_m > 0.5
                if m.sum() > 100:
                    lab_c = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                    lab_s = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
                    lc, ls = lab_c[..., 0][m], lab_s[..., 0][m]
                    L = (lab_c[..., 0] - lc.mean()) * (ls.std() / max(lc.std(), 1e-3)) + ls.mean()
                    lab_c[..., 0] = lab_c[..., 0] * (1 - full_m) + L * full_m
                    comp = cv2.cvtColor(np.clip(lab_c, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
                comp = comp + args.detail * detail * feat_m

            # relight: the actress's low-frequency shading + skin tone on our character's face
            if detected and (args.relight > 0 or args.skin > 0):
                face_w = float(np.ptp(pts[:, 0]))
                fm = np.zeros((H, W), np.float32)
                cv2.fillConvexPoly(fm, cv2.convexHull(pts.astype(np.int32)), 1.0, cv2.LINE_AA)
                hard = fm > 0.5
                fm = cv2.GaussianBlur(fm, (0, 0), max(3.0, face_w / 20))
                lc = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                ls = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
                if args.relight > 0:
                    sig = max(6.0, face_w / 7)
                    c_low = cv2.GaussianBlur(lc[..., 0], (0, 0), sig)
                    s_low = s_low_all[i] if s_low_all is not None else cv2.GaussianBlur(ls[..., 0], (0, 0), sig)
                    relit = lc[..., 0] - c_low + s_low * args.relight + c_low * (1 - args.relight)
                    lc[..., 0] = lc[..., 0] * (1 - fm) + relit * fm
                # skin statistics only on the inner face (below the eyes): bangs / side hair excluded
                skin_ids = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
                            397, 288, 361, 323, 454, 352, 346, 347, 348, 349, 350, 6, 121, 120, 119, 118, 117, 123]
                skin_m = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(skin_m, cv2.convexHull(pts[skin_ids].astype(np.int32)), 1)
                # exclude eyes/mouth interiors and lips (not skin)
                for ids in ([33, 133, 159, 145, 160, 144, 158, 153], [362, 263, 386, 374, 385, 380, 387, 373],
                            [61, 291, 0, 17, 37, 267, 84, 314]):
                    cv2.fillConvexPoly(skin_m, cv2.convexHull(pts[ids].astype(np.int32)), 0)
                hard = skin_m > 0
                if args.skin > 0 and hard.sum() > 100:
                    cs, cc = ls[..., 1:][hard], lc[..., 1:][hard]
                    cs_mean, cs_std = (skin_stats[i, :2], skin_stats[i, 2:]) if skin_stats is not None \
                        else (cs.mean(0), cs.std(0))
                    matched = (lc[..., 1:] - cc.mean(0)) * (cs_std / np.maximum(cc.std(0), 1e-3)) + cs_mean
                    # only below the eyebrows: bangs over the forehead must keep the hair colour
                    brow_low = float(pts[[70, 63, 105, 66, 107, 336, 296, 334, 293, 300], 1].max())
                    below = np.zeros((H, W), np.float32)
                    below[int(max(brow_low, 0)):] = 1
                    below = cv2.GaussianBlur(below, (0, 0), max(3.0, face_w / 40))
                    w = (args.skin * fm * below)[..., None]
                    lc[..., 1:] = lc[..., 1:] * (1 - w) + matched * w
                comp = cv2.cvtColor(np.clip(lc, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

            # only inside the actress's tight mask; outside stays the original frame
            region = m_t if region_extra is None else np.maximum(m_t, region_extra)
            soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.5)[..., None]
            if args.alpha_out:
                cv2.imwrite(str(args.alpha_out / f"{i:05d}.png"), region * 255)
            out = frame.astype(np.float32) * (1 - soft) + comp * soft
            cv2.imwrite(f"{td}/{i:05d}.png", np.clip(out, 0, 255).astype(np.uint8))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{args.fps}",
                        "-i", f"{td}/%05d.png", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p",
                        str(args.out)], check=True)
    if seg is not None:
        seg.seg.close()
    print(f"composite -> {args.out}")


if __name__ == "__main__":
    main()
