#!/usr/bin/env python3
"""Face-refine pass (Carasibana/ComfyUI-H3-FaceRefine) over an assembled swap clip, shot by shot.

For every shot of the clip (cut points as in bfs_full.py):
  H3 Face Track + Crop  -> tracks HER face only (insightface identity match against --face-ref, so the man is
                           never picked), crops it to a 768-capped canvas
  H3 Inject Video Latent + H3 Per-Frame Denoise -> img2img re-render of the crop with H3 R2V (+ <Picture 1> = her
                           face), low denoise, stronger where the face is small
  H3 Face Stitch Back   -> pastes only the face box back, colour-matched and feathered
Optionally stacks fal/MiniMax-H3-Realism-People-LoRA in the refine pass (--realism).
Shots listed in --keep (default 1,3: back view / source shot) are copied through. Original audio on the result.

Usage:
  python face_refine.py --clip outputs/final_bfs_manlock/hug_scene_swapped_720p.mp4 --out outputs/final_bfs_manlock_fr \
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

PROMPT = ("{trigger}Close-up of <Subject 1>, the young woman shown in <Picture 1>: same face, same identity, fair skin "
          "with natural texture and fine pores, dark brown eyes, small nose, soft pink lips, long straight wine-red "
          "hair with blunt bangs. Natural micro-expressions, natural blinking, realistic eyes and teeth, lighting and "
          "colour exactly as in the source footage, cinematic, photorealistic.")


def build(video: str, face: str, prompt: str, denoise: float, steps: int, seed: int, realism: float,
          prefix: str) -> dict:
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
                           "denoise_multiplier_small_face": 1.0, "denoise_multiplier_large_face": 0.35,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True, help="assembled 1280x720 swap clip (24 fps)")
    ap.add_argument("--source", type=Path, required=True, help="the source video (for the original audio)")
    ap.add_argument("--face-ref", type=Path, required=True, help="face image of the new character")
    ap.add_argument("--cuts", default="26,73,110,149,204", help="shot boundaries in the 24 fps clip")
    ap.add_argument("--keep", default="1,3",
                    help="shots copied through untouched (1: only her back is visible, the one face in shot is his)")
    ap.add_argument("--denoise", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--realism", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--comfy-urls", default="http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    inp = ROOT / "ComfyUI" / "input"
    tag = out.name
    fname = f"fr_{tag}_face{args.face_ref.suffix}"
    shutil.copy2(args.face_ref, inp / fname)
    prompt = PROMPT.format(trigger="r34l1sm, " if args.realism > 0 else "")
    (out / "prompt.txt").write_text(prompt + "\n")

    # split the clip into shots (frame-exact, re-encoded)
    n_total = int(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                           "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                                           str(args.clip)]).decode().strip())
    b = [0] + [int(c) for c in args.cuts.split(",")] + [n_total]
    keep = {int(k) for k in args.keep.split(",") if k}
    shots = {}
    for i in range(len(b) - 1):
        p = out / f"in_s{i + 1}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.clip), "-vf",
                        f"select='between(n\\,{b[i]}\\,{b[i + 1] - 1})',setpts=N/24/TB", "-r", "24", "-an",
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(p)], check=True)
        shots[i + 1] = p

    urls = queue.Queue()
    for u in args.comfy_urls.split(","):
        urls.put(u)

    def run(i: int) -> Path:
        res = out / f"s{i}.mp4"
        if i in keep:
            shutil.copy2(shots[i], res)
            return res
        u = urls.get()
        try:
            vname = f"fr_{tag}_s{i}.mp4"
            shutil.copy2(shots[i], inp / vname)
            wf = build(vname, fname, prompt, args.denoise, args.steps, args.seed, args.realism, f"fr/{tag}_s{i}")
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
        parts = list(ex.map(run, sorted(shots)))
    (out / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
    final = out / "hug_scene_swapped_720p.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(out / "list.txt"),
                    "-i", str(args.source), "-map", "0:v", "-map", "1:a", "-vf", "scale=1280:720",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final)],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.clip), "-i", str(final), "-filter_complex",
                    "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]hstack", "-map", "1:a", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", str(out / "before_after.mp4")], check=True)
    for p in shots.values():
        p.unlink(missing_ok=True)
    print(f"FINAL -> {final}")


if __name__ == "__main__":
    main()
