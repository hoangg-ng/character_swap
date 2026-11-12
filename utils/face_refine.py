#!/usr/bin/env python3
"""Face-refine pass (Carasibana/ComfyUI-H3-FaceRefine) over an assembled swap clip, shot by shot.

For every shot of the clip (cut points as in bfs_full.py):
  H3 Face Track + Crop  -> tracks HER face only (insightface identity match against --face-ref, so the man is
                           never picked), crops it to a 768-capped canvas
  H3 Inject Video Latent + H3 Per-Frame Denoise -> img2img re-render of the crop with H3 R2V (+ <Picture 1> = her
                           face), low denoise, stronger where the face is small
  H3 Face Stitch Back   -> pastes only the face box back, colour-matched and feathered
  identity guard        -> per frame, ArcFace similarity to --face-ref of the refined vs the unrefined face; where
                           the refined face is worse by more than --guard-margin (or lost), that frame cross-fades
                           back to the unrefined one (FaceRefine occasionally distorts a face at a shot edge)
Shots shorter than --min-shot frames are not refined (too short for a stable track).
Optionally stacks fal/MiniMax-H3-Realism-People-LoRA in the refine pass (--realism).
Shots listed in --keep are copied through (shots without the person). Output keeps the input size; original audio.

Shot mode (--shots-job): refine every generated shot clip <job>/shotNN/save_*.mp4 as one shot and write
<--out>/shotNN/save_fr_shotNN_00001_.mp4 (a job folder that CHAR_SWAP_FROM_JOB can upscale), no assembly.

Usage:
  python utils/face_refine.py --clip cases/x/output/final/swapped.mp4 --source cases/x/input/video.mp4 \
      --face-ref cases/x/input/face.png --cuts 37,163,... --keep 6 --out cases/x/output/final_fr \
      [--realism 0.8] [--denoise 0.4]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue

ROOT = Path(__file__).resolve().parent.parent   # project root
sys.path.insert(0, str(ROOT))
from utils import comfy_client as cs  # noqa: E402

PROMPT = ("{trigger}Close-up of <Subject 1>, the person shown in <Picture 1>: exactly the same face and identity as "
          "<Picture 1>{details}, skin with natural texture and fine pores. Natural micro-expressions, natural blinking, "
          "realistic eyes and teeth, lighting and colour exactly as in the source footage, cinematic, photorealistic.")


def build(video: str, face: str, prompt: str, denoise: float, steps: int, seed: int, realism: float,
          prefix: str, mult_large: float = 0.35) -> dict:
    wf = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "attn": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["shift", 0], "attention": "pytorch attention"}},
        "turbo": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["attn", 0], "lora_name": "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors",
                             "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "face": {"class_type": "LoadImage", "inputs": {"image": face}},
        "vid": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["vid", 0]}},
        "track": {"class_type": "H3FaceTrackCrop",
                  "inputs": {"images": ["parts", 0], "detector": "face_yolov8m.pt", "confidence": 0.35,
                             "crop_factor": 3.0, "canvas_width": 768, "canvas_height": 768,
                             "canvas_mode": "auto_capped_768", "smooth_window": 21, "size_smooth_window": 51,
                             "smooth_method": "gaussian", "size_mode": "per_frame",
                             "identity_reference": ["face", 0], "identity_track": True, "identity_threshold": 0.0,
                             "select": "largest_face", "identity_model": "insightface", "cut_detection": "none",
                             "absent_shots": "by_identity"}},   # skip a shot if her face never matches
        "r2v": {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["clip", 0], "prompt": prompt, "width": ["track", 4], "height": ["track", 5],
                           "length": ["track", 6], "ref_image_size": "match", "vae": ["vvae", 0],
                           "audio_vae": ["avae", 0], "ref_images.ref_image_0": ["face", 0]}},
        "inject": {"class_type": "H3InjectVideoLatent",
                   "inputs": {"av_latent": ["r2v", 1], "images": ["track", 0], "vae": ["vvae", 0]}},
        "pfd": {"class_type": "H3PerFrameDenoise",
                "inputs": {"model": ["turbo", 0], "av_latent": ["inject", 0], "transform": ["track", 1],
                           "denoise_multiplier_small_face": 1.0, "denoise_multiplier_large_face": mult_large,
                           "scale_mode": "absolute_px", "face_px_small": 30.0, "face_px_large": 120.0,
                           "gamma": 1.0, "smooth_frames": 9}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["pfd", 2], "conditioning": ["r2v", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["pfd", 2], "scheduler": "simple",
                                                             "steps": steps, "denoise": denoise}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["pfd", 0]}},
        "sep": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sample", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sep", 0], "vae": ["vvae", 0]}},
        "stitch": {"class_type": "H3FaceStitch",
                   "inputs": {"base_images": ["parts", 0], "refined_crops": ["dec", 0], "transform": ["track", 1],
                              "paste_region": "face_only", "mask_dilation": 24, "feather": 24, "colour_match": 1.0,
                              "blend": 1.0, "undetected_frames": "fade_out"}},
        "mk": {"class_type": "CreateVideo", "inputs": {"images": ["stitch", 0], "fps": 24.0}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["mk", 0], "filename_prefix": prefix,
                                                       "format": "mp4", "codec": "h264"}},
        "rep": {"class_type": "PreviewAny", "inputs": {"source": ["track", 3]}},
    }
    if realism > 0:
        wf["real"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": ["turbo", 0], "lora_name": "h3-realism-people-t2v-i2v-r2v.safetensors",
                                 "strength_model": realism}}
        wf["pfd"]["inputs"]["model"] = ["real", 0]
    return wf


_FACE_APP = None


def _face_app():
    global _FACE_APP
    if _FACE_APP is None:
        import logging
        logging.getLogger("insightface").setLevel(logging.ERROR)
        from insightface.app import FaceAnalysis
        _FACE_APP = FaceAnalysis(name="buffalo_l", root=str(ROOT / "ComfyUI" / "models" / "insightface"),
                                 allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
        _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _FACE_APP


def _read_frames(path: Path) -> list:
    import cv2
    cap = cv2.VideoCapture(str(path)); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


def _best_sim(app, frame, ref) -> float:
    """Highest ArcFace cosine of any face in the frame to the reference; -1 if no face is found."""
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    sc = 960 / max(h, w)
    small = cv2.resize(frame, (int(w * sc), int(h * sc))) if sc < 1 else frame
    return max((float(np.dot(f.normed_embedding, ref)) for f in app.get(small)), default=-1.0)


def identity_guard(unrefined: Path, refined: Path, ref, out: Path, margin: float, lock, log: Path | None = None
                   ) -> tuple[int, int]:
    """Cross-fade refined frames whose identity dropped (vs the unrefined frame) back to the unrefined frame.
    Returns (frames touched, total frames)."""
    import numpy as np
    a, b = _read_frames(unrefined), _read_frames(refined)
    n = min(len(a), len(b))
    with lock:                                   # insightface on CPU, one shot at a time
        app = _face_app()
        sa = np.array([_best_sim(app, a[k], ref) for k in range(n)])
        sb = np.array([_best_sim(app, b[k], ref) for k in range(n)])
    bad = (sa > 0) & (sb < sa - margin)          # refined face clearly worse than before, or lost
    keep = np.ones(n)
    for k in np.flatnonzero(bad):                # revert the bad frame and its neighbours, ramp the switch
        keep[max(0, k - 2):k + 3] = 0
    alpha = np.convolve(np.pad(keep, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    alpha[bad] = 0.0
    if log is not None:
        log.write_text("frame,sim_before,sim_after,alpha\n" +
                       "".join(f"{k},{sa[k]:.3f},{sb[k]:.3f},{alpha[k]:.2f}\n" for k in range(n)))
    h, w = b[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "12",
                          "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for k in range(n):
        f = b[k] if alpha[k] >= 1 else (alpha[k] * b[k] + (1 - alpha[k]) * a[k]).astype(np.uint8)
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    return int((alpha < 1).sum()), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, default=None, help="assembled swap clip (24 fps, any size)")
    ap.add_argument("--source", type=Path, default=None, help="the source video (for the original audio)")
    ap.add_argument("--face-ref", type=Path, required=True, help="face image of the new character")
    ap.add_argument("--cuts", default="", help="shot boundaries in the 24 fps clip")
    ap.add_argument("--keep", default="", help="shots copied through untouched (no person / no visible face)")
    ap.add_argument("--details", default="", help="short face/hair description of the character")
    ap.add_argument("--denoise", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mult-large", type=float, default=0.35,
                    help="denoise multiplier for large faces (>120 px); small faces use 1.0")
    ap.add_argument("--realism", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guard-margin", type=float, default=0.08,
                    help="revert a frame whose ArcFace similarity drops more than this after refining (<0: off)")
    ap.add_argument("--min-shot", type=int, default=24, help="shots shorter than this (frames) are not refined")
    ap.add_argument("--shots-job", type=Path, default=None,
                    help="shot mode: refine each <job>/shotNN/save_*.mp4 (shots listed in --only, or all)")
    ap.add_argument("--only", default="", help="shot mode: shot numbers to refine")
    ap.add_argument("--only-needed", action="store_true",
                    help="refine a shot only if it needs it: median face smaller than --need-face px, or identity to "
                         "--face-ref below --need-id; large faces that already match are left as generated")
    ap.add_argument("--need-face", type=float, default=120.0)
    ap.add_argument("--need-id", type=float, default=0.40)
    ap.add_argument("--guard-only", type=Path, default=None,
                    help="skip ComfyUI: apply the guard to already refined shots s<N>.mp4 in this folder")
    ap.add_argument("--comfy-urls", default="http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    inp = ROOT / "ComfyUI" / "input"
    tag = out.name
    fname = f"fr_{tag}_face{args.face_ref.suffix}"
    shutil.copy2(args.face_ref, inp / fname)
    prompt = PROMPT.format(trigger="r34l1sm, " if args.realism > 0 else "",
                           details=f" ({args.details.strip().rstrip('.')})" if args.details else "")
    (out / "prompt.txt").write_text(prompt + "\n")

    def nframes(v: Path) -> int:
        return int(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                                            str(v)]).decode().strip())

    keep = {int(k) for k in args.keep.split(",") if k}
    shots, length = {}, {}
    if args.shots_job:                          # shot mode: the generated clips are the shots
        only = {int(k) for k in args.only.split(",") if k}
        for d in sorted(args.shots_job.glob("shot[0-9][0-9]")):
            i = int(d.name[4:]); gens = sorted(d.glob("save_*.mp4"), key=lambda q: q.stat().st_mtime)
            if gens and (not only or i in only):
                p = out / f"in_s{i}.mp4"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(gens[-1]), "-r", "24", "-an",
                                "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(p)], check=True)
                shots[i] = p; length[i] = nframes(p)
    else:
        if not (args.clip and args.source and args.cuts):
            sys.exit("--clip, --source and --cuts are required (or use --shots-job)")
        b = [0] + [int(c) for c in args.cuts.split(",")] + [nframes(args.clip)]
        for i in range(len(b) - 1):     # split the clip into shots (frame-exact, re-encoded)
            p = out / f"in_s{i + 1}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.clip), "-vf",
                            f"select='between(n\\,{b[i]}\\,{b[i + 1] - 1})',setpts=N/24/TB", "-r", "24", "-an",
                            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(p)], check=True)
            shots[i + 1] = p; length[i + 1] = b[i + 1] - b[i]
    import threading
    import time
    import numpy as np
    t_start = time.time()
    guard_lock = threading.Lock()
    ref = None
    if args.guard_margin >= 0 or args.guard_only or args.only_needed:
        import cv2
        fs = _face_app().get(cv2.imread(str(args.face_ref)))
        if not fs:
            sys.exit(f"no face found in {args.face_ref}")
        ref = max(fs, key=lambda f: f.det_score).normed_embedding

    urls = queue.Queue()
    for u in args.comfy_urls.split(","):
        urls.put(u)

    def run(i: int) -> Path:
        res = out / f"s{i}.mp4"
        if args.only_needed and i not in keep and length[i] >= args.min_shot:
            import cv2
            fsz, ids = [], []
            with guard_lock:
                app = _face_app(); cap = cv2.VideoCapture(str(shots[i])); k = 0
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    if k % 4 == 0:
                        sc = 960 / max(f.shape[:2]); sm = cv2.resize(f, None, fx=sc, fy=sc) if sc < 1 else f
                        fs = app.get(sm)
                        if fs:
                            b = max(fs, key=lambda x: float(np.dot(x.normed_embedding, ref)))
                            ids.append(float(np.dot(b.normed_embedding, ref))); fsz.append((b.bbox[3] - b.bbox[1]) / min(sc, 1))
                    k += 1
            if fsz and np.median(fsz) >= args.need_face and np.median(ids) >= args.need_id:
                print(f"shot {i}: face {np.median(fsz):.0f}px, identity {np.median(ids):.2f} -> good already, not refined",
                      flush=True)
                shutil.copy2(shots[i], out / f"s{i}.mp4")
                return out / f"s{i}.mp4"
        if i in keep or length[i] < args.min_shot:
            if i not in keep:
                print(f"shot {i}: {length[i]} frames < --min-shot -> unrefined", flush=True)
            shutil.copy2(shots[i], res)
            return res
        if args.guard_only:
            nrev, nf = identity_guard(shots[i], args.guard_only / f"s{i}.mp4", ref, res, args.guard_margin,
                                      guard_lock, out / f"guard_s{i}.csv")
            print(f"shot {i}: identity guard reverted {nrev}/{nf} frames", flush=True)
            return res
        u = urls.get()
        try:
            vname = f"fr_{tag}_s{i}.mp4"
            shutil.copy2(shots[i], inp / vname)
            wf = build(vname, fname, prompt, args.denoise, args.steps, args.seed, args.realism, f"fr/{tag}_s{i}",
                       args.mult_large)
            (out / f"workflow_s{i}.json").write_text(json.dumps(wf, indent=2))
            c = cs.Comfy(u)
            pid = c.submit(wf)
            print(f"shot {i}: refining on {u}", flush=True)
            hist = c.wait(pid)
            sd = out / f"gen_s{i}"; sd.mkdir(exist_ok=True)
            files = [f for f in c.download_outputs(hist, sd) if str(f).endswith(".mp4")]
            for node in hist.get("outputs", {}).values():          # tracker report, for diagnosis
                for t in node.get("text", []) if isinstance(node, dict) else []:
                    (out / f"report_s{i}.txt").write_text(str(t))
            if args.guard_margin >= 0:
                nrev, nf = identity_guard(shots[i], Path(files[-1]), ref, res, args.guard_margin, guard_lock,
                                          out / f"guard_s{i}.csv")
                print(f"shot {i}: done, identity guard reverted {nrev}/{nf} frames", flush=True)
            else:
                shutil.copy2(files[-1], res)
                print(f"shot {i}: done", flush=True)
            return res
        except (Exception, SystemExit) as e:  # Comfy.wait sys.exit()s on a node error; keep the unrefined shot
            msg = str(e)
            why = ("her face never matched (only his face visible)" if "not containing the subject" in msg
                   else msg[:300])
            print(f"shot {i}: SKIPPED/FAILED ({why}) -> unrefined", flush=True)
            shutil.copy2(shots[i], res)
            return res
        finally:
            urls.put(u)

    with ThreadPoolExecutor(max_workers=len(args.comfy_urls.split(","))) as ex:
        parts = list(ex.map(run, sorted(shots, key=lambda i: -length[i])))
    if args.shots_job:                          # shot mode: a job folder of refined shots, no assembly
        for i in shots:
            d = out / f"shot{i:02d}"; d.mkdir(exist_ok=True)
            shutil.move(str(out / f"s{i}.mp4"), d / f"save_fr_shot{i:02d}_00001_.mp4")
            shots[i].unlink(missing_ok=True)
        print(f"FINAL -> {out}")
        return
    parts = [out / f"s{i}.mp4" for i in sorted(shots)]
    (out / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
    final = out / "swapped.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(out / "list.txt"),
                    "-i", str(args.source), "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final)],
                   check=True)
    for p in shots.values():
        p.unlink(missing_ok=True)
    print(f"FINAL -> {final}  ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
