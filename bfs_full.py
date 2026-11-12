#!/usr/bin/env python3
"""Per-shot BFS v1.1 generation (run_bfs.py) over several GPUs: target check, shot split, padding, dispatch.

Per generated shot: frames resampled to 24 fps at the generation size, --lead sacrificial copies of the first
frame are prepended (the reference bleeds into the first latent group), the clip is ping-pong padded to the next
17n+5 length and generated. Results go to $CHAR_SWAP_OUT/<job>/shotNN/ (default outputs/); assembly is done by
assemble_bfs.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("CHAR_SWAP_OUT", ROOT / "outputs"))   # per-case output dir
FPS_SRC = 30000 / 1001    # replaced by the real source fps in main()
W, H = 1312, 736          # default generation size; --width/--height override


def read_frames(path: Path, start: int, end: int) -> list:
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or i >= end:
            break
        if i >= start:
            out.append(f)
        i += 1
    return out


def to24(frames: list, n24: int) -> list:
    return [frames[min(int(round(k * FPS_SRC / 24)), len(frames) - 1)] for k in range(n24)]


def write(frames: list, path: Path, fps: float = 24.0, audio_from: Path | None = None, ss: float = 0, t: float = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".noaudio.mp4")
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(tmp)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    if audio_from:
        # pad the audio with silence (never trim the video: frame count must stay exact)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ss", f"{ss}", "-t", f"{t}",
                        "-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
                        "-ar", "48000", "-af", "apad", "-t", f"{len(frames) / fps:.6f}", str(path)], check=True)
        tmp.unlink()
    else:
        tmp.rename(path)


def make_target_ref(frames: list, text: str, out: Path, url: str) -> Path | None:
    """SAM-detect `text` on the clearest real frame (largest detection) and save her cut-out on white."""
    sys.path.insert(0, str(ROOT))
    from utils import comfy_client as cs
    out.parent.mkdir(parents=True, exist_ok=True)
    picks = [frames[k] for k in sorted({0, len(frames) // 3, len(frames) // 2, (2 * len(frames)) // 3})]
    found = []            # (frame, mask, area) per pick, in time order
    c = cs.Comfy(url)
    for j, f in enumerate(picks):
        name = f"tref_{out.stem}_{j}.png"
        cv2.imwrite(str(ROOT / "ComfyUI" / "input" / name), f)
        wf = {"img": {"class_type": "LoadImage", "inputs": {"image": name}},
              "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
              "t": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": text}},
              "d": {"class_type": "SAM3_Detect", "inputs": {"model": ["sam", 0], "image": ["img", 0], "conditioning": ["t", 0],
                    "threshold": 0.4, "refine_iterations": 2, "individual_masks": False}},
              "mi": {"class_type": "MaskToImage", "inputs": {"mask": ["d", 0]}},
              "s": {"class_type": "SaveImage", "inputs": {"images": ["mi", 0], "filename_prefix": f"tref/{out.stem}_{j}"}}}
        h = c.wait(c.submit(wf), poll=2)
        mdir = out.parent / (out.stem + "_masks")
        mdir.mkdir(parents=True, exist_ok=True)
        mf = [p for p in c.download_outputs(h, mdir) if p.suffix == ".png"]
        if not mf:
            continue
        m = cv2.resize(cv2.imread(str(mf[0]), 0), (f.shape[1], f.shape[0])) > 127
        if m.any():
            found.append((f, m, int(m.sum())))
    # earliest frame with a solid detection (largest-mask frames can include an occluder merged into her mask)
    best = None
    if found:
        top = max(a for _, _, a in found)
        best = next((f, m) for f, m, a in found if a >= 0.5 * top)
    if best is None:
        print(f"{out.name}: SAM found nobody -> no <Picture 2>", flush=True)
        return None
    f, m = best
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)[..., None]
    cut = (f * a + 255 * (1 - a)).astype(np.uint8)
    ys, xs = np.where(m)
    pad = 24
    cut = cut[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    cv2.imwrite(str(out), cut)
    print(f"target reference -> {out} ({cut.shape[1]}x{cut.shape[0]})", flush=True)
    return out


def load_edits(path: str) -> dict:
    """Per-shot edits (web UI): {shot: "note"} or {shot: {"note", "take", "soft_face_scale", "keep_outfit",
    "relight"}}. Returns {int shot: dict}."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    for k, v in json.loads(Path(path).read_text()).items():
        out[int(k)] = v if isinstance(v, dict) else {"note": str(v)}
    return out


def soften_faces(inp: Path, out: Path, face_px: int = 20, min_face: int = 16, lead: int = 0,
                 small_scale: float = 0.35) -> Path | None:
    """Guide with the original person's face softened: every face (insightface) larger than `min_face` px is
    reduced to ~`face_px` px across and scaled back up, feathered. Position, head turn and the coarse expression
    (eyes / mouth open or shut) stay; the identity detail is gone, so BFS has to take the face from the references
    instead of copying the original (which it did in close-ups, depending on the seed). Only the GUIDE changes.
    Returns None when no face is large enough (the guide stays as it is)."""
    import logging
    logging.getLogger("insightface").setLevel(logging.ERROR)
    sys.path.insert(0, str(ROOT))
    from utils.face_refine import _face_app
    app = _face_app()
    frames = read_frames(inp, 0, 10 ** 6)
    boxes = []
    for f in frames:
        fs = [x for x in app.get(f) if x.bbox[2] - x.bbox[0] >= min_face]
        boxes.append([x.bbox for x in fs])
    if not any(boxes):
        return None
    for k in range(len(frames)):                 # fill single-frame detection gaps from the neighbours
        if not boxes[k]:
            boxes[k] = boxes[k - 1] if k and boxes[k - 1] else (boxes[k + 1] if k + 1 < len(boxes) else [])
    soft = []
    for f, bs in zip(frames, boxes):
        g = f.copy()
        for b in bs:
            x0, y0, x1, y1 = b.astype(int); w, h = x1 - x0, y1 - y0
            x0, y0 = max(0, x0 - w // 6), max(0, y0 - h // 6); x1, y1 = min(f.shape[1], x1 + w // 6), min(f.shape[0], y1 + h // 6)
            roi = f[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            sc = max(4.0, min(face_px, small_scale * roi.shape[1])) / max(1, roi.shape[1])   # small faces: a fraction
            small = cv2.resize(roi, (max(2, int(roi.shape[1] * sc)), max(2, int(roi.shape[0] * sc))), interpolation=cv2.INTER_AREA)
            blur = cv2.GaussianBlur(cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC), (0, 0), 2)
            m = np.zeros(roi.shape[:2], np.float32)
            cv2.ellipse(m, (roi.shape[1] // 2, roi.shape[0] // 2), (roi.shape[1] // 2, roi.shape[0] // 2), 0, 0, 360, 1, -1)
            m = cv2.GaussianBlur(m, (0, 0), max(2, roi.shape[1] / 12))[..., None]
            g[y0:y1, x0:x1] = (roi * (1 - m) + blur * m).astype(np.uint8)
        soft.append(g)
    write(soft, out, audio_from=inp, ss=0, t=len(soft) / 24)
    return out


def make_clean_guide(inp: Path, hide: str, thr: float, out: Path, url: str) -> Path:
    """SAM-track `hide` (e.g. the man) in every frame and inpaint him out -> guide where the target is the only
    person. Only the guide changes; the source video is untouched."""
    sys.path.insert(0, str(ROOT))
    import shutil
    from utils import comfy_client as cs
    name = f"hide_{inp.parent.name}_{inp.stem}.mp4"
    shutil.copy2(inp, ROOT / "ComfyUI" / "input" / name)
    wf = {"v": {"class_type": "LoadVideo", "inputs": {"file": name}},
          "p": {"class_type": "GetVideoComponents", "inputs": {"video": ["v", 0]}},
          "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
          "t": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": hide}},
          "tr": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["p", 0], "model": ["sam", 0],
                 "detection_threshold": thr, "max_objects": 2, "detect_interval": 1, "conditioning": ["t", 0]}},
          "m": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["tr", 0], "object_indices": ""}},
          "g": {"class_type": "GrowMask", "inputs": {"mask": ["m", 0], "expand": 12, "tapered_corners": True}},
          "mi": {"class_type": "MaskToImage", "inputs": {"mask": ["g", 0]}},
          "s": {"class_type": "SaveImage", "inputs": {"images": ["mi", 0], "filename_prefix": f"hide/{inp.stem}"}}}
    c = cs.Comfy(url)
    h = c.wait(c.submit(wf), poll=2)
    mdir = out.parent / (out.stem + "_masks")
    mdir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in c.download_outputs(h, mdir) if f.suffix == ".png")
    frames = read_frames(inp, 0, 10 ** 6)
    clean = []
    for k, f in enumerate(frames):
        m = cv2.imread(str(files[min(k, len(files) - 1)]), 0) if files else None
        if m is None or m.max() == 0:
            clean.append(f); continue
        m = cv2.resize(m, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
        small = 4
        fill = cv2.inpaint(cv2.resize(f, (f.shape[1] // small, f.shape[0] // small)),
                           cv2.resize(m, (f.shape[1] // small, f.shape[0] // small), interpolation=cv2.INTER_NEAREST),
                           7, cv2.INPAINT_TELEA)
        fill = cv2.GaussianBlur(cv2.resize(fill, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
        a = cv2.GaussianBlur((m > 127).astype(np.float32), (0, 0), 4)[..., None]
        clean.append((f * (1 - a) + fill * a).astype(np.uint8))
    write(clean, out)
    print(f"guide with '{hide}' removed -> {out} ({sum(1 for f in files)} masks)", flush=True)
    return out


def target_presence(video: Path, shots: list, face: Path | None, sex: str, sim: float = 0.35,
                    every: int = 2, min_score: float = 0.5) -> dict:
    """Per shot, the fraction of sampled frames in which the TARGET person (the one to replace) is visible.
    face: an image of the target's face (screenshot crop) -> identity match (insightface ArcFace cosine >= sim).
    sex:  'F' / 'M' fallback when no face image is given -> any face of that gender counts.
    A shot where the target is never seen from the front scores ~0: BFS would swap whoever is visible there
    (e.g. the man), so such shots are kept as source."""
    import logging
    logging.getLogger("insightface").setLevel(logging.ERROR)
    from insightface.app import FaceAnalysis
    mods = ["detection", "recognition"] if face else ["detection", "genderage"]
    app = FaceAnalysis(name="buffalo_l", root=str(ROOT / "ComfyUI" / "models" / "insightface"),
                       allowed_modules=mods, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    anchor = None
    if face:
        fs = app.get(cv2.imread(str(face)))
        if not fs:
            sys.exit(f"target check: no face found in {face}")
        anchor = max(fs, key=lambda f: f.det_score).normed_embedding

    def is_target(f) -> bool:
        if f.det_score < min_score:
            return False
        if anchor is not None:
            return float(np.dot(f.normed_embedding, anchor)) >= sim
        return f.sex == sex

    out = {}
    for i, (a, e) in enumerate(shots, 1):
        frames = read_frames(video, a, e)[::every]
        hit = sum(any(is_target(f) for f in app.get(fr)) for fr in frames)
        out[i] = hit / max(1, len(frames))
    return out


def person_frames(video: Path, conf: float = 0.35, min_area: float = 0.005, kp_conf: float = 0.5) -> tuple:
    """Per source frame: (person, head).
    person: YOLO11 detection, a body box >= min_area of the frame (catches backs, silhouettes, partial bodies);
            this defines the with/without segments.
    head:   YOLO11-pose, any head keypoint (nose, eyes, ears) with kp_conf - also works from behind. Body parts
            alone (a kimono hem, a sleeve) give person=1, head=0."""
    from ultralytics import YOLO
    d = ROOT / "ComfyUI" / "models" / "ultralytics" / "person"
    det, pose = YOLO(str(d / "yolo11m.pt")), YOLO(str(d / "yolo11m-pose.pt"))
    cap, person, head, light = cv2.VideoCapture(str(video)), [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        area = fr.shape[0] * fr.shape[1]
        r = det.predict(fr, classes=[0], conf=conf, verbose=False, device="cpu")[0]
        boxes = [b for b in r.boxes.xyxy.tolist() if (b[2] - b[0]) * (b[3] - b[1]) / area >= min_area]
        person.append(int(bool(boxes)))
        if boxes:                                    # median brightness inside the largest person box
            x0, y0, x1, y1 = map(int, max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
            light.append(float(np.median(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))))
        else:
            light.append(-1.0)
        k = pose.predict(fr, conf=0.3, verbose=False, device="cpu")[0].keypoints
        k = k.conf if k is not None else None
        head.append(int(k is not None and len(k) > 0 and bool((k[:, :5] > kp_conf).any())))
    return person, head, light


def split_by_presence(bounds: list, pf: list, gap: int, min_run: int, min_seg: int) -> tuple:
    """Split shots into segments WITH and WITHOUT the person.
    pf is smoothed per shot: presence blips shorter than min_run frames are dropped (detector noise) and absence
    stretches shorter than gap frames (also at the shot start/end) are filled (detector misses, e.g. a kimono under a newspaper). A shot is then
    cut wherever the smoothed presence changes, keeping segments of at least min_seg frames.
    Returns (new bounds, {segment index: with person?})."""
    nb, flag = [0], {}
    for a, e in zip(bounds[:-1], bounds[1:]):
        x = list(pf[a:e])
        def runs(v):
            r, i = [], 0
            while i < len(x):
                j = i
                while j < len(x) and x[j] == x[i]:
                    j += 1
                r.append((i, j, x[i])); i = j
            return r
        for i, j, v in runs(x):                                   # drop short presence blips
            if v == 1 and j - i < min_run and j - i < len(x):    # a flash shot (1-2 frames) is all presence
                x[i:j] = [0] * (j - i)
        if any(x):
            for i, j, v in runs(x):                               # fill short gaps (also at shot start/end)
                if v == 0 and j - i < gap:
                    x[i:j] = [1] * (j - i)
        segs = [[i, j, v] for i, j, v in runs(x)]
        merged = []                                               # absorb segments that are too short
        for sg in segs:
            if merged and (sg[1] - sg[0] < min_seg or merged[-1][1] - merged[-1][0] < min_seg):
                big = merged[-1] if merged[-1][1] - merged[-1][0] >= sg[1] - sg[0] else sg
                merged[-1] = [merged[-1][0], sg[1], big[2] if sg[2] != merged[-1][2] else sg[2]]
            else:
                merged.append(sg)
        for i, j, v in merged:
            if a + i != nb[-1]:
                nb.append(a + i)
            flag[len(nb)] = bool(v)
    nb.append(bounds[-1])
    return nb, flag


def face_fraction(video: Path, shots: list, every: int = 2, min_score: float = 0.5) -> dict:
    """Per shot, the fraction of sampled source frames with any detected face (insightface detection)."""
    import logging
    logging.getLogger("insightface").setLevel(logging.ERROR)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=str(ROOT / "ComfyUI" / "models" / "insightface"),
                       allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    out = {}
    for i, (a, e) in enumerate(shots, 1):
        frames = read_frames(video, a, e)[::every]
        out[i] = sum(any(f.det_score > min_score for f in app.get(fr)) for fr in frames) / max(1, len(frames))
    return out


def auto_cuts(video: Path, sensitivity: float = 1.5, min_change: float = 0.38, window: int = 6,
              same_scene: float = 0.15) -> str:
    """Scene cuts. A missed cut is the worst error (the person is drawn into a shot without her, or her face drifts
    across the change), so detection is sensitive and every candidate is then checked:
      1. candidates: PySceneDetect AdaptiveDetector at 1.5 (default 3.0), shots >= 8 frames - finds dissolves and
         cuts between similar-looking shots;
      2. colour check: Bhattacharyya distance of the HSV histograms of the `window` frames before / after >=
         min_change (fast motion of hair or arms trips step 1 but keeps the colours);
      3. snap: the cut moves to the sharpest frame change within +-3 frames (the detector can land 1-2 frames late;
         2 frames of the next shot made a sleeve-only segment count as "person visible" -> an invented face);
      4. same-scene check: ORB features matched across the cut (frame c-1 vs c+2, and c-4 vs c+5); a real cut shares
         almost none (<= 0.04 on 29 labelled cuts), an occluder crossing the same shot (smoke, a passer-by) keeps
         the background (0.23) -> dropped when >= same_scene;
      5. flash frames: 1-2 frames of another shot inserted into a scene get their own cuts (they differ from both
         neighbours, mean grey difference >= 2.5/255, while the neighbours match each other).
    On 3 hand-labelled clips (31 cuts incl. 2 flashes): every cut found, frame-exact except one fade frame, no extra cut."""
    from scenedetect import detect, AdaptiveDetector
    cand = [sc[0].frame_num for sc in detect(str(video), AdaptiveDetector(adaptive_threshold=sensitivity,
                                                                          min_scene_len=8))[1:]]
    cap = cv2.VideoCapture(str(video)); full = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        full.append(cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1]))) if f.shape[1] > 480 else f)
    fr = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV) for f in full]
    g = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in full]
    n = len(fr)
    pd = lambda i, j: float(np.abs(g[i] - g[j]).mean())

    def hist(fs):
        h = sum(cv2.calcHist([f], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256]) for f in fs)
        return cv2.normalize(h, None).flatten()

    orb, bfm = cv2.ORB_create(500), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def shared(i, j):           # fraction of ORB features of frame i that reappear in frame j
        ka, da = orb.detectAndCompute(cv2.cvtColor(full[i], cv2.COLOR_BGR2GRAY), None)
        kb, db = orb.detectAndCompute(cv2.cvtColor(full[j], cv2.COLOR_BGR2GRAY), None)
        if da is None or db is None or min(len(ka), len(kb)) < 10:
            return 0.0
        return len([m for m in bfm.match(da, db) if m.distance < 40]) / min(len(ka), len(kb))

    cuts = []
    for c in cand:
        d = cv2.compareHist(hist(fr[max(0, c - window):c]), hist(fr[c:c + window]), cv2.HISTCMP_BHATTACHARYYA)
        if d < min_change:
            print(f"  cut candidate {c}: colour change {d:.2f} -> ignored (motion)")
            continue
        best = max(range(max(1, c - 3), min(n - 1, c + 3) + 1), key=lambda k: pd(k - 1, k))
        sh = max(shared(max(0, best - 1 - o), min(n - 1, best + 2 + o)) for o in (0, 3))
        if sh >= same_scene:
            print(f"  cut candidate {c}: colour change {d:.2f}, but {sh:.0%} of the picture continues -> same scene")
            continue
        print(f"  cut {best}" + (f" (detector said {c})" if best != c else "") + f": colour change {d:.2f}, shared {sh:.0%}")
        cuts.append(best)
    cuts = sorted(set(cuts))
    k = 1
    while k < n - 1:
        for L in (1, 2):
            if k + L >= n:
                continue
            edge = min(pd(k - 1, k), pd(k + L - 1, k + L))
            if edge >= 2.5 and pd(k - 1, k + L) <= 0.5 * edge and all(abs(k - c) > 2 for c in cuts):
                print(f"  flash frames {k}-{k + L - 1}: cut around them")
                cuts += [k, k + L]
                k += L
                break
        k += 1
    return ",".join(map(str, sorted(set(cuts))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "hug_scene.mp4")
    ap.add_argument("--cuts", default="33,92,138,187,256")
    ap.add_argument("--keep", default="", help="shots always kept as source")
    ap.add_argument("--min-target", type=float, default=0.1,
                    help="target check: a shot is generated only if a female face is seen in at least this fraction "
                         "of its frames; otherwise it stays exactly as the source (0 = check off)")
    ap.add_argument("--target-face", type=Path, default=None,
                    help="image of the face of the person to REPLACE (screenshot crop); shots are generated only "
                         "where this identity is visible")
    ap.add_argument("--target-sex", default="F", help="fallback target check without --target-face: F or M")
    ap.add_argument("--min-person-light", type=float, default=6.0,
                    help="presence=head: segments whose person is darker than this (median, 0-255) are silhouettes "
                         "and stay source")
    ap.add_argument("--min-head", type=float, default=0.3,
                    help="presence=head: a segment is generated only if the head is seen in this fraction of it")
    ap.add_argument("--presence", default="face", choices=["face", "person", "head"],
                    help="face: target identity/sex must be seen (multi-person videos); person: any person visible "
                         "(single-person videos: also back views, silhouettes, partial bodies); head: like person, but segments "
                         "showing only body parts (hem, sleeve) stay source - nothing there to swap")
    ap.add_argument("--face-sim", type=float, default=0.35, help="ArcFace cosine needed to count as the target")
    ap.add_argument("--keep-outfit", action="store_true", help="face-only reference: keep the guide's clothing")
    ap.add_argument("--face-only", action="store_true", help="swap only the face (combine with --subject face)")
    ap.add_argument("--ref-extra", default="", help="comma-separated extra character views (character sheet)")
    ap.add_argument("--ref-color-scene", action="store_true", help="reference colour from the scene only")
    ap.add_argument("--ref-color-mode", default="scene", help="scene | scene_lum | skin (see run_bfs.py)")
    ap.add_argument("--upscale", type=float, default=0.0, help="two-stage latent upscale factor (0 = off)")
    ap.add_argument("--realism-stage", default="both", help="with --upscale: Realism LoRA in stage 1, 2 or both")
    ap.add_argument("--ref-face", default="",
                    help="comma-separated face references (face photo, face close-up); added only to shots where a "
                         "face is visible in the source - in back views they pull the framing to a face portrait")
    ap.add_argument("--ref-back", default="", help="back-view reference: becomes <Picture 1> in shots without a face")
    ap.add_argument("--min-face", type=float, default=0.2, help="--ref-face: face visible in >= this fraction")
    ap.add_argument("--check-only", action="store_true", help="run the target check, print decisions, exit")
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--details", required=True)
    ap.add_argument("--shot-notes", default="", help="JSON {shot: extra instruction} appended to --details per shot")
    ap.add_argument("--target", default="the woman with long black hair in the white jacket")
    ap.add_argument("--keep-others", default="the man in the black vest and white shirt")
    ap.add_argument("--subject", default="", help="SAM text of the person to replace -> per-frame edit mask")
    ap.add_argument("--mask-grow", type=int, default=24)
    ap.add_argument("--protect", default="", help="SAM text of people to cut out of her edit mask, e.g. 'man'")
    ap.add_argument("--protect-only", action="store_true", help="mask only the protected people; no mask on her")
    ap.add_argument("--job-suffix", default="")
    ap.add_argument("--light-match", action="store_true")
    ap.add_argument("--realism", type=float, default=0.0, help="Realism People LoRA strength (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref-color", default="", help="METHOD:STRENGTH for ColorTransfer on the reference, per shot")
    ap.add_argument("--width", type=int, default=1312)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--target-ref-text", default="",
                    help="SAM text of the person to replace; a cut-out of her from the shot becomes <Picture 2>")
    ap.add_argument("--soft-face-scale", type=float, default=0.15, help="small faces are reduced to this fraction")
    ap.add_argument("--face-lead", action="store_true", help="in shots with a visible face the face photo is <Picture 1>")
    ap.add_argument("--soften-face", action="store_true",
                    help="soften the original person's face in the guide (identity removed, pose/expression kept)")
    ap.add_argument("--hide", default="", help="SAM text of people to remove from the GUIDE only, e.g. 'man'")
    ap.add_argument("--hide-threshold", type=float, default=0.7)
    ap.add_argument("--lead", type=int, default=5, help="sacrificial copies of the first frame (one latent group)")
    ap.add_argument("--only", default="", help="regenerate only these shots (others reuse existing results)")
    ap.add_argument("--solo-shots", default="4,6", help="shots where she is the only person (no target note)")
    ap.add_argument("--comfy-urls", default="http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190")
    ap.add_argument("--job", default="bfs_full")
    ap.add_argument("--out", type=Path, default=None, help="unused (kept for old commands)")
    args = ap.parse_args()
    global W, H, FPS_SRC
    W, H = args.width, args.height
    _c = cv2.VideoCapture(str(args.video)); FPS_SRC = _c.get(cv2.CAP_PROP_FPS) or FPS_SRC; _c.release()

    if args.cuts == "auto":
        args.cuts = auto_cuts(args.video)
        print(f"auto cuts: {args.cuts}", flush=True)
    cap = cv2.VideoCapture(str(args.video)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    b = [0] + [int(c) for c in args.cuts.split(",")] + [total]
    shots = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
    keep = {int(k) for k in args.keep.split(",") if k}
    (OUT_ROOT / args.job).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / args.job / "cuts.txt").write_text(args.cuts + "\n")    # for assemble_bfs.py
    if args.min_target > 0 and args.presence in ("person", "head"):
        pf, hf, lf = person_frames(args.video)
        b, flag = split_by_presence(b, pf, gap=int(round(FPS_SRC * 0.5)), min_run=2, min_seg=4)
        shots = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
        args.cuts = ",".join(str(x) for x in b[1:-1])
        (OUT_ROOT / args.job / "cuts.txt").write_text(args.cuts + "\n")
        print(f"segments with/without the person: cuts {args.cuts}", flush=True)
        if args.presence == "head":                    # keep body-part-only segments (no head) as source
            for i, (a, e) in enumerate(shots, 1):
                hfrac = sum(hf[a:e]) / max(1, e - a)
                if flag.get(i) and hfrac < args.min_head:
                    print(f"segment {i}: person but head in only {hfrac:.0%} of frames -> body parts only, keep",
                          flush=True)
                    flag[i] = False
            for i, (a, e) in enumerate(shots, 1):         # silhouettes: identity invisible, keep the original look
                lv = [x for x in lf[a:e] if x >= 0]
                if flag.get(i) and lv and float(np.mean(lv)) < args.min_person_light:
                    print(f"segment {i}: person median brightness {np.mean(lv):.1f} -> silhouette, keep", flush=True)
                    flag[i] = False
    if args.min_target > 0:
        pres = ({i: (sum(pf[a:e]) / max(1, e - a) if flag.get(i) else 0.0) for i, (a, e) in enumerate(shots, 1)}
                if args.presence in ("person", "head") else
                target_presence(args.video, shots, args.target_face, args.target_sex.upper()[:1], args.face_sim))
        for i, fr in pres.items():
            ok = fr >= args.min_target if args.presence == "face" else bool(flag.get(i))
            print(f"target check shot {i}: target visible in {fr:.0%} of frames -> {'generate' if ok else 'KEEP SOURCE'}",
                  flush=True)
            if not ok:
                keep.add(i)
        (OUT_ROOT / args.job).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / args.job / "target_check.json").write_text(
            json.dumps({str(k): {"target_frac": round(v, 3),
                                 "generate": (v >= args.min_target) if args.presence == "face" else bool(flag.get(k)),
                                 "frames": list(shots[k - 1]),
                                 "by": args.presence if args.presence in ("person", "head") else
                                       (str(args.target_face) if args.target_face else f"sex={args.target_sex}")}
                        for k, v in pres.items()}, indent=1))
        if args.check_only:
            return
    solo = {int(k) for k in args.solo_shots.split(",") if k}
    if not args.protect:
        solo = set(range(1, len(shots) + 1))          # nobody else to keep: no "keep the others" prompt note
    work = OUT_ROOT / args.job
    work.mkdir(parents=True, exist_ok=True)
    edits = load_edits(args.shot_notes)

    plan = {}
    for i, (a, e) in enumerate(shots, 1):
        src = read_frames(args.video, a, e)
        n24 = int(round(len(src) * 24 / FPS_SRC))
        f24 = [cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4) for f in to24(src, n24)]
        plan[i] = dict(a=a, e=e, n=n24, frames=f24)
        if i in keep:
            continue
        seq = [f24[0]] * args.lead + f24           # sacrificial leading frames (reference bleeds into them)
        L = len(seq)
        target = 5 + 17 * max(1, -(-(L - 5) // 17))  # next 17n+5 >= L
        pp = f24[::-1][1:] + f24[1:]               # ping-pong continuation
        k = 0
        while len(seq) < target:
            seq.append(pp[k % len(pp)] if pp else f24[-1]); k += 1
        inp = work / f"shot{i:02d}_in.mp4"
        write(seq, inp, audio_from=args.video, ss=a / FPS_SRC, t=len(seq) / 24)
        plan[i].update(inp=inp, length=target, lead=args.lead)
        if args.target_ref_text and (not args.only or str(i) in args.only.split(",")):
            plan[i]["tref"] = make_target_ref(f24, args.target_ref_text, work / f"shot{i:02d}_target.png",
                                              args.comfy_urls.split(",")[0])
        if args.hide and (not args.only or str(i) in args.only.split(",")):
            plan[i]["guide"] = make_clean_guide(inp, args.hide, args.hide_threshold, work / f"shot{i:02d}_guide.mp4",
                                                args.comfy_urls.split(",")[0])
        if args.soften_face and "guide" not in plan[i] and (not args.only or str(i) in args.only.split(",")):
            sg = soften_faces(inp, work / f"shot{i:02d}_guide_softface.mp4",
                              small_scale=float(edits.get(i, {}).get("soft_face_scale", args.soft_face_scale)),
                              face_px=int(edits.get(i, {}).get("soft_face_px", 20)))
            if sg:
                plan[i]["guide"] = sg
                print(f"shot {i}: original face softened in the guide", flush=True)
        print(f"shot {i}: {n24} frames -> generate {target} (incl. sacrificial frame 0)")

    face_frac = {}
    if args.ref_face:
        todo_ = [i for i in plan if i not in keep]
        ff = face_fraction(args.video, shots)
        for i in todo_:
            face_frac[i] = ff[i]
            print(f"shot {i}: face visible in {ff[i]:.0%} -> face refs {'on' if ff[i] >= args.min_face else 'off'}",
                  flush=True)

    urls = queue.Queue()
    for u in args.comfy_urls.split(","):
        urls.put(u.strip())

    def details_for(i: int) -> str:      # per-shot instruction (web UI "edit this shot") appended to the description
        note = str(edits.get(i, {}).get("note", "")).strip()
        return args.details if not note else f"{args.details.rstrip('. ')}. In this shot: {note}"

    def run(i):
        u = urls.get()
        try:
            job = f"{args.job}/shot{i:02d}"
            cmd = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "run_bfs.py"), "--video", str(plan[i]["inp"]),
                   "--ref", str(args.ref), "--details", details_for(i), "--width", str(W), "--height", str(H),
                   "--frames", str(plan[i]["length"]), "--job", job, "--comfy-url", u]
            if i not in solo and args.target:
                cmd += ["--target", args.target, "--keep-others", args.keep_others]
            if args.protect_only and args.protect:
                cmd += ["--protect", args.protect, "--protect-only"]
            elif args.subject:
                cmd += ["--subject", args.subject, "--mask-grow", str(args.mask_grow)]
                if args.protect:
                    cmd += ["--protect", args.protect]
            if plan[i].get("guide"):
                cmd += ["--guide-video", str(plan[i]["guide"])]
            if args.light_match:
                cmd += ["--light-match"]
            if args.realism > 0:
                cmd += ["--realism", str(args.realism)]
            if args.ref_color:
                cmd += ["--ref-color", args.ref_color]
                if args.ref_color_scene:
                    cmd += ["--ref-color-scene", "--ref-color-mode", args.ref_color_mode]
            cmd += ["--seed", str(args.seed + int(edits.get(i, {}).get("take", 0)))]      # "take": a new variation
            if edits.get(i, {}).get("keep_outfit", args.keep_outfit):
                cmd += ["--keep-outfit"]
            if args.face_only:
                cmd += ["--face-only"]
            extra = [x for x in args.ref_extra.split(",") if x]
            if args.ref_face and face_frac.get(i, 0) >= args.min_face:
                fr_ = [x for x in args.ref_face.split(",") if x]
                if args.face_lead:            # face visible: her real face photo leads the references as <Picture 1>
                    lead_ = fr_[-1]
                    extra = [str(args.ref)] + extra + [x for x in fr_ if x != lead_]
                    cmd[cmd.index("--ref") + 1] = lead_
                    print(f"shot {i}: face visible -> face photo leads the references", flush=True)
                else:
                    extra += fr_
            elif args.ref_back and args.ref_face and face_frac.get(i, 0) < args.min_face:
                # no face in the shot (back view, distant figure): the back view leads as <Picture 1>; with the front
                # view first the model drew a large frontal portrait instead of a small figure walking away
                extra = [str(args.ref)] + [x for x in extra if x != args.ref_back]
                cmd[cmd.index("--ref") + 1] = args.ref_back
                print(f"shot {i}: no face -> back view leads the references", flush=True)
            if extra:
                cmd += ["--ref-extra", ",".join(extra)]
            if args.upscale and args.upscale > 1.0:
                cmd += ["--upscale", str(args.upscale), "--realism-stage", args.realism_stage]
                src_job = os.environ.get("CHAR_SWAP_FROM_JOB", "")      # stage 2 only, on an existing job's results
                if src_job:
                    gens = sorted((OUT_ROOT / src_job / f"shot{i:02d}").glob("save_*.mp4"), key=lambda p: p.stat().st_mtime)
                    if gens:
                        cmd += ["--from-gen", str(gens[-1])]
            if plan[i].get("tref"):
                cmd += ["--target-ref", str(plan[i]["tref"])]
            print(f"shot {i}: generating on {u}", flush=True)
            sd = OUT_ROOT / args.job / f"shot{i:02d}"
            sd.mkdir(parents=True, exist_ok=True)
            (sd / "lead.txt").write_text(str(plan[i]["lead"]))     # how many leading frames to drop
            with open(work / f"shot{i:02d}.log", "w") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"shot {i}: {'done' if rc == 0 else 'FAILED'}", flush=True)
            return i, rc
        finally:
            urls.put(u)

    only = {int(k) for k in args.only.split(",") if k}
    todo = sorted([i for i in plan if i not in keep and (not only or i in only)], key=lambda i: -plan[i]["length"])
    with ThreadPoolExecutor(max_workers=urls.qsize()) as ex:
        failed = [i for i, rc in ex.map(run, todo) if rc != 0]
    if failed:
        sys.exit(f"failed shots: {failed}")

    print(f"generation finished -> {OUT_ROOT / args.job} (assemble with assemble_bfs.py)")


if __name__ == "__main__":
    main()
