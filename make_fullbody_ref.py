#!/usr/bin/env python3
"""Make a full-body character reference from a face-only photo with MiniMax H3 reference-to-video.

BFS copies the framing of its reference: a face selfie turns every shot into a frontal portrait. This renders a
short clip of the person (<Picture 1> = face photo) standing full body on a plain white background, in the outfit
you describe, and saves its frames; pick the best one as CHAR_REF.

  python make_fullbody_ref.py --face cases/x/input/face_cutout.png --outfit "red floral kimono ..." \
      --out cases/x/input/fullbody --seeds 1,2,3
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import comfy_client as cs  # noqa: E402

PROMPT = ("r34l1sm, full-body studio photo of the woman shown in <Picture 1>: exactly her face, identity, hairstyle, "
          "hair color and bangs. She stands still facing the camera, whole body visible from head to feet, arms "
          "relaxed at her sides, neutral calm expression. She wears {outfit}. Plain pure white seamless background, "
          "soft even front lighting, sharp focus, static camera, no movement.")


def build(face: str, prompt: str, w: int, h: int, n: int, seed: int, prefix: str) -> dict:
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "attn": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["shift", 0], "attention": "pytorch attention"}},
        "turbo": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["attn", 0], "lora_name": "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors",
                             "strength_model": 1.0}},
        "real": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["turbo", 0], "lora_name": "h3-realism-people-t2v-i2v-r2v.safetensors",
                            "strength_model": 0.8}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "face": {"class_type": "LoadImage", "inputs": {"image": face}},
        "r2v": {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["clip", 0], "prompt": prompt, "width": w, "height": h, "length": n,
                           "ref_image_size": "match", "vae": ["vvae", 0], "audio_vae": ["avae", 0],
                           "ref_images.ref_image_0": ["face", 0]}},
        "guider": {"class_type": "CFGGuider",
                   "inputs": {"model": ["real", 0], "positive": ["r2v", 0], "negative": ["r2v", 0], "cfg": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["real", 0], "scheduler": "simple",
                                                             "steps": 8, "denoise": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["r2v", 1]}},
        "sep": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sample", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sep", 0], "vae": ["vvae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--outfit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--comfy-urls", default="http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fname = "fullbody_face" + args.face.suffix
    shutil.copy2(args.face, ROOT / "ComfyUI" / "input" / fname)
    prompt = PROMPT.format(outfit=args.outfit)
    (args.out / "prompt.txt").write_text(prompt + "\n")
    urls = args.comfy_urls.split(",")

    def run(k_seed):
        k, seed = k_seed
        c = cs.Comfy(urls[k % len(urls)])
        h = c.wait(c.submit(build(fname, prompt, args.width, args.height, args.frames, seed, f"fullbody/s{seed}")),
                   poll=3)
        d = args.out / f"seed{seed}"
        d.mkdir(exist_ok=True)
        return [p for p in c.download_outputs(h, d) if p.suffix == ".png"]

    seeds = [int(s) for s in args.seeds.split(",")]
    with ThreadPoolExecutor(len(seeds)) as ex:
        for seed, files in zip(seeds, ex.map(run, enumerate(seeds))):
            print(f"seed {seed}: {len(files)} frames -> {args.out}/seed{seed}")
    (ROOT / "ComfyUI" / "input" / fname).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
