#!/usr/bin/env python3
"""Run Alissonerdx's BFS Character Swap v1.1 recipe (native-node core) on one clip.

Graph (rebuilt from workflows/minimax_h3_character_swap_workflow_native_nodes_v1.1.json, stock nodes only):
  UNET ref2va pruned int8 -> SigmaShift 12/3 -> attention backend -> DMD ref2va 8-step turbo (1.0)
  -> BFS character swap v1.1 (1.0) -> CFGGuider(cfg 1)
  ReferenceToVideo(prompt, <Picture 1> = character) -> AddGuide(frame 0, guide = the original clip + audio)
  latent = EmptyMiniMaxH3LatentAV (whole frame generated) ; euler, simple, 8 steps
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils import comfy_client as cs  # noqa: E402

LIGHT_MATCH = ("Match the lighting of the guide video: <Subject 1> is lit only by the same light sources as the guide "
               "video, with the same light direction, color temperature, exposure, contrast, shadows and highlights on "
               "the face, hair and clothing as the guide subject; do not use the lighting, white balance or background "
               "of <Picture 1>. ")

REALISM_LORA = "h3-realism-people-t2v-i2v-r2v.safetensors"
TEMPLATE = ("body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, "
            "head, hair or fur, skin, body shape, proportions and clothing define the character in the target video. "
            "The guide video provides the scene and the performance: its background, environment, lighting, camera, "
            "framing, body pose, hand and arm movement, head rotation, gaze direction, lip movement and facial "
            "expression, frame by frame; only the main subject of the guide video is replaced: its face, hair, body "
            "and clothing become <Subject 1>.  summary: [video generation + reference] Replace the guide subject with "
            "<Subject 1>, keeping the scene, camera and the complete motion and performance of the guide video.  "
            "detailed_description: [Shot 1] <Subject 1> takes the exact place of the guide subject and moves with the "
            "same timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the "
            "complete identity from <Picture 1>: facial structure, hair or fur, skin, body shape and clothing stay "
            "stable in every frame. The background, lighting, colors and camera of the guide video are preserved. "
            "Prevent identity drift, clothing changes, texture crawling and flicker.")


KEEP_OUTFIT = [   # identity from <Picture 1>, clothing and styling stay exactly as worn in the guide video
    ("whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the character in "
     "the target video.",
     "whose identity, face, head, hair, skin and body shape define the character in the target video. The clothing "
     "of <Picture 1> is NOT used: her clothing, accessories and styling are taken from the guide subject and stay "
     "exactly as worn in the guide video, with the same garment, pattern, colors and fit."),
    ("its face, hair, body and clothing become <Subject 1>.",
     "its face, head, hair and body become <Subject 1>, while its clothing and accessories are kept unchanged."),
    ("facial structure, hair or fur, skin, body shape and clothing stay stable in every frame.",
     "facial structure, hair, skin and body shape stay stable in every frame, and the clothing of the guide "
     "subject is preserved exactly."),
]


FACE_ONLY = [   # only the face is swapped (SAM "face" edit mask): hair, head shape, clothing, body stay as in the guide
    ("whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the character in "
     "the target video.",
     "whose face and facial identity (eyes, eyebrows, nose, lips, face shape, skin tone and makeup) define the "
     "character in the target video; her hair, hairstyle, clothing, accessories and body are NOT taken from "
     "<Picture 1> and stay exactly as the guide subject's."),
    ("its face, hair, body and clothing become <Subject 1>.",
     "only its face becomes the face of <Subject 1>; its hair, clothing and body stay unchanged."),
    ("facial structure, hair or fur, skin, body shape and clothing stay stable in every frame.",
     "facial structure, eyes, nose, lips and skin tone stay stable in every frame, while the guide subject's "
     "hair, expression, eye state (open or closed), head angle and makeup lighting are followed exactly."),
]


def build(video: str, ref: str, prompt: str, w: int, h: int, n: int, seed: int, steps: int,
          lora: str, lora_strength: float, prefix: str) -> dict:
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "attn": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["shift", 0], "attention": "pytorch attention"}},
        "turbo": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["attn", 0], "lora_name": "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors",
                             "strength_model": 1.0}},
        "swap": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["turbo", 0], "lora_name": lora, "strength_model": lora_strength}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": ref}},
        "vid": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["vid", 0]}},
        "r2v": {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["clip", 0], "prompt": prompt, "width": w, "height": h, "length": n,
                           "ref_image_size": "match", "vae": ["vvae", 0], "audio_vae": ["avae", 0],
                           "ref_images.ref_image_0": ["ref", 0]}},
        "empty": {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": w, "height": h, "length": n}},
        "guide": {"class_type": "MiniMaxH3AddGuide",
                  "inputs": {"positive": ["r2v", 0], "latent": ["empty", 0], "frame_idx": 0, "vae": ["vvae", 0],
                             "audio_vae": ["avae", 0], "image": ["parts", 0], "audio": ["parts", 1]}},
        "guider": {"class_type": "CFGGuider",
                   "inputs": {"model": ["swap", 0], "positive": ["guide", 0], "negative": ["guide", 0], "cfg": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["swap", 0], "scheduler": "simple",
                                                             "steps": steps, "denoise": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["empty", 0]}},
        "sep": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sample", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sep", 0], "vae": ["vvae", 0]}},
        "mk": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": 24.0, "audio": ["parts", 1]}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["mk", 0], "filename_prefix": prefix,
                                                       "format": "mp4", "codec": "h264"}},
    }


UPSCALER = "minimax_h3_latent_upscaler_2d_conv_v1_fp16.safetensors"
STAGE2_SIGMAS = "0.9035, 0.6316, 0.3158, 0.0000"      # refine pass after the latent upscale (as in the upscaler's
                                                      # own R2V example workflow)


def add_upscale(wf: dict, scale: float, w: int, h: int, n: int, stage2_model: list, sharpness: int = 1,
                chunk_frames: int = 73) -> tuple:
    """Two-stage generation: stage 1 as built (w x h), then the video latent is upscaled by the learned H3 latent
    upscaler (LBH-123-AI, 2D v1), re-joined with the audio latent and refined at the high resolution with a short
    3-step pass, conditioned on the same references and guide video re-encoded at the new size."""
    w2, h2 = int(round(w * scale / 32)) * 32, int(round(h * scale / 32)) * 32
    wf["s1_sep"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sample", 0]}}
    wf["up"] = {"class_type": "MinimaxH3LatentUpscalerNode2D",
                "inputs": {"latent": ["s1_sep", 0], "model_name": UPSCALER, "mode": "scale by multiplier",
                           "mode.scale": float(scale), "sharpness": sharpness, "align": 32,
                           "temporal_chunking": "single pass", "force_unload": True, "device": "cuda",
                           "precision": "fp16"}}
    wf["s2_cat"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["up", 0], "audio_latent": ["s1_sep", 1]}}
    r2 = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": dict(wf["r2v"]["inputs"])}
    r2["inputs"].update({"width": w2, "height": h2})
    wf["r2v2"] = r2
    wf["empty2"] = {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": w2, "height": h2, "length": n}}
    g2 = {"class_type": "MiniMaxH3AddGuide", "inputs": dict(wf["guide"]["inputs"])}
    g2["inputs"].update({"positive": ["r2v2", 0], "latent": ["empty2", 0]})
    wf["guide2"] = g2
    wf["sigmas2"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE2_SIGMAS}}
    # refine in overlapping time chunks (MMH3SplitUpscale: H3-grid chunks, cross-fade, motion/identity anchors,
    # per-chunk guide keyframes) so long shots fit in GPU memory at high resolution
    wf["split2"] = {"class_type": "MMH3TemporalSplitParamsV10",
                    "inputs": {"chunk_frames": chunk_frames, "temporal_overlap_frames": 22, "anchor_strength": 0.999,
                               "motion_anchor_frames": "22", "identity_anchor_frames": 24}}
    wf["sample2"] = {"class_type": "MMH3SplitUpscale",
                     "inputs": {"model": stage2_model, "conditioning": ["guide2", 0], "negative": ["guide2", 0],
                                "latent": ["s2_cat", 0], "noise": ["noise", 0], "sampler": ["sampler", 0],
                                "sigmas": ["sigmas2", 0], "cfg": 1.0, "temporal_split_param": ["split2", 0],
                                "seam_polish": "off", "color_match": True}}
    wf["sep"]["inputs"]["av_latent"] = ["sample2", 0]           # decode the refined, upscaled result
    return w2, h2


def add_sam_mask(wf: dict, subject: str, threshold: float, grow: int, prefix: str,
                 protect: str = "", protect_threshold: float = 0.7) -> dict:
    """Confine generation to the target person: SAM 3.1 tracks `subject` in every frame of the guide; the
    (grown) mask becomes the edit mask - inside it the model generates, outside it the source latent is kept.
    Frames where SAM finds nobody get an empty mask and stay exactly as the source."""
    wf = dict(wf)
    wf["sam"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}}
    wf["sam_txt"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": subject}}
    wf["sam_track"] = {"class_type": "SAM3_VideoTrack",
                       "inputs": {"images": ["parts", 0], "model": ["sam", 0], "detection_threshold": threshold,
                                  "max_objects": 1, "detect_interval": 1, "conditioning": ["sam_txt", 0]}}
    wf["sam_mask"] = {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["sam_track", 0], "object_indices": ""}}
    wf["sam_grow"] = {"class_type": "GrowMask", "inputs": {"mask": ["sam_mask", 0], "expand": grow, "tapered_corners": True}}
    final_mask = ["sam_grow", 0]
    if protect:
        # people who must not change (e.g. the man): tracked separately and cut out of her edit mask
        wf["prot_txt"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": protect}}
        wf["prot_track"] = {"class_type": "SAM3_VideoTrack",
                            "inputs": {"images": ["parts", 0], "model": ["sam", 0],
                                       "detection_threshold": protect_threshold, "max_objects": 2,
                                       "detect_interval": 1, "conditioning": ["prot_txt", 0]}}
        wf["prot_mask"] = {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["prot_track", 0],
                                                                         "object_indices": ""}}
        wf["sam_final"] = {"class_type": "MaskComposite",
                           "inputs": {"destination": ["sam_grow", 0], "source": ["prot_mask", 0],
                                      "x": 0, "y": 0, "operation": "subtract"}}
        final_mask = ["sam_final", 0]
    wf["src_enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["parts", 0], "vae": ["vvae", 0]}}
    wf["edit"] = {"class_type": "H3RelayApplyEditMask",
                  "inputs": {"latent": ["empty", 0], "source_latent": ["src_enc", 0], "mask": final_mask}}
    wf["sample"]["inputs"]["latent_image"] = ["edit", 0]
    wf["mask_img"] = {"class_type": "MaskToImage", "inputs": {"mask": final_mask}}
    wf["mask_save"] = {"class_type": "SaveImage", "inputs": {"images": ["mask_img", 0], "filename_prefix": prefix + "_mask"}}
    return wf


def add_protect_only(wf: dict, protect: str, threshold: float, grow: int, prefix: str) -> dict:
    """Mask ONLY the people who must not change (e.g. the man): their latent is locked to the source, everything
    else - the target woman and the background - is generated freely as one image (no mask edge on her).
    Frames where SAM finds nobody to protect are generated in full."""
    wf = dict(wf)
    wf["sam"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}}
    wf["prot_txt"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": protect}}
    wf["prot_track"] = {"class_type": "SAM3_VideoTrack",
                        "inputs": {"images": ["parts", 0], "model": ["sam", 0], "detection_threshold": threshold,
                                   "max_objects": 2, "detect_interval": 1, "conditioning": ["prot_txt", 0]}}
    wf["prot_mask"] = {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["prot_track", 0], "object_indices": ""}}
    wf["prot_grow"] = {"class_type": "GrowMask", "inputs": {"mask": ["prot_mask", 0], "expand": grow, "tapered_corners": True}}
    wf["edit_free"] = {"class_type": "InvertMask", "inputs": {"mask": ["prot_grow", 0]}}
    wf["src_enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["parts", 0], "vae": ["vvae", 0]}}
    wf["edit"] = {"class_type": "H3RelayApplyEditMask",
                  "inputs": {"latent": ["empty", 0], "source_latent": ["src_enc", 0], "mask": ["edit_free", 0]}}
    wf["sample"]["inputs"]["latent_image"] = ["edit", 0]
    wf["prot_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ["prot_grow", 0]}}
    wf["prot_save"] = {"class_type": "SaveImage", "inputs": {"images": ["prot_img", 0], "filename_prefix": prefix + "_protect"}}
    return wf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="24 fps clip, 17n+5 frames, 32-aligned size")
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--details", required=True, help="comma-separated head-to-toe description, starting with gender")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--light-match", action="store_true", help="add the lighting-match instruction to the prompt")
    ap.add_argument("--target-ref", type=Path, default=None,
                    help="image of the person to replace, cut from the video (becomes <Picture 2>)")
    ap.add_argument("--guide-video", type=Path, default=None,
                    help="separate guide for AddGuide (e.g. other people removed); --video stays the source")
    ap.add_argument("--subject", default="",
                    help="SAM text for the person to replace; enables the per-frame SAM edit mask")
    ap.add_argument("--sam-threshold", type=float, default=0.5)
    ap.add_argument("--mask-grow", type=int, default=24, help="px the SAM mask is grown (room for new hair)")
    ap.add_argument("--protect", default="", help="SAM text of people that must stay unchanged, e.g. 'man'")
    ap.add_argument("--protect-only", action="store_true",
                    help="mask only the --protect people (locked); the target and background are generated freely")
    ap.add_argument("--protect-grow", type=int, default=4)
    ap.add_argument("--keep-others", default="",
                    help="who must stay unchanged, e.g. 'the man in the black vest'")
    ap.add_argument("--target", default="",
                    help="which guide subject to replace when several people appear, e.g. 'the woman in the white jacket'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lora", default="minimax_h3_character_swap_v1.1_ref2va_fro.safetensors")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--ref-color", default="",
                    help="METHOD:STRENGTH, e.g. reinhard_lab:0.6 - recolour the character reference to this shot's "
                         "colour theme with ComfyUI's ColorTransfer node (ref = the shot's frames) before it is "
                         "used as <Picture 1>; the recoloured reference is saved as refcolor_*.png")
    ap.add_argument("--ref-color-scene", action="store_true",
                    help="with --ref-color reinhard_lab: take the colour statistics from the scene only (all people "
                         "masked out with YOLO11-seg, utils/scene_color.py) instead of the whole frames, so the "
                         "replaced person's clothes do not tint the character")
    ap.add_argument("--ref-color-mode", default="scene", choices=["scene", "scene_lum", "skin"],
                    help="with --ref-color-scene: scene = scene without people; scene_lum = the same, but colour "
                         "(a/b) only at 30%% so the skin hue stays the character's; skin = light on the people's "
                         "face skin, mapped from the character's face skin")
    ap.add_argument("--realism", type=float, default=0.0,
                    help="strength of fal/MiniMax-H3-Realism-People-LoRA stacked after BFS (0 = off; 0.6-1.0); "
                         "also prefixes the prompt with its trigger word r34l1sm")
    ap.add_argument("--ref-extra", default="",
                    help="comma-separated extra views of the same character (character sheet) -> <Picture 2..N>")
    ap.add_argument("--face-only", action="store_true",
                    help="swap only the face (use with --subject face): hair, clothing, body follow the guide")
    ap.add_argument("--keep-outfit", action="store_true",
                    help="face-only reference: swap face/head/hair, keep the guide subject's clothing")
    ap.add_argument("--upscale", type=float, default=0.0,
                    help="two-stage: upscale the stage-1 latent by this factor with the H3 latent upscaler and refine "
                         "at the higher resolution (0 = off, e.g. 1.5)")
    ap.add_argument("--from-gen", type=Path, default=None,
                    help="with --upscale: skip stage 1 and upscale/refine this existing generated clip of the shot "
                         "(re-encoded with the video VAE)")
    ap.add_argument("--chunk-frames", type=int, default=73,
                    help="with --upscale: stage-2 refine chunk length in frames (H3 grid 5+17n; overlap 22)")
    ap.add_argument("--realism-stage", default="both", choices=["1", "2", "both"],
                    help="with --upscale: apply the Realism LoRA in stage 1, stage 2 or both")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    out = Path(os.environ.get("CHAR_SWAP_OUT", ROOT / "outputs")) / args.job
    out.mkdir(parents=True, exist_ok=True)
    inp = ROOT / "ComfyUI" / "input"
    safe = args.job.replace("/", "_")
    vname, rname = f"bfs_{safe}.mp4", f"bfs_{safe}{args.ref.suffix}"
    shutil.copy2(args.video, inp / vname)
    shutil.copy2(args.ref, inp / rname)
    prompt = TEMPLATE.format(details=args.details.strip().rstrip("."))
    if args.face_only:
        for a, b in FACE_ONLY:
            assert a in prompt, a
            prompt = prompt.replace(a, b)
    elif args.keep_outfit:
        for a, b in KEEP_OUTFIT:
            assert a in prompt, a
            prompt = prompt.replace(a, b)
    if args.light_match:
        prompt = prompt.replace("Prevent identity drift", LIGHT_MATCH + "Prevent identity drift", 1)
    if args.target:
        # name the target everywhere the template says "main subject" / "guide subject"
        t = args.target
        prompt = prompt.replace("only the main subject of the guide video is replaced",
                                f"only {t} in the guide video is replaced")
        prompt = prompt.replace("Replace the guide subject with", f"Replace {t} with")
        prompt = prompt.replace("takes the exact place of the guide subject", f"takes the exact place of {t}")
        others = args.keep_others or "every other person"
        prompt = prompt.replace("  summary:", f" {others[0].upper() + others[1:]} in the guide video is not replaced "
                                              f"and stays exactly as in the guide video.  summary:", 1)
    (out / "prompt.txt").write_text(prompt + "\n")
    wf = build(vname, rname, prompt, args.width, args.height, args.frames, args.seed, args.steps,
               args.lora, args.lora_strength, f"bfs/{safe}")
    if args.target_ref:
        tname = f"bfs_{safe}_target{args.target_ref.suffix}"
        shutil.copy2(args.target_ref, inp / tname)
        wf["tref"] = {"class_type": "LoadImage", "inputs": {"image": tname}}
        wf["r2v"]["inputs"]["ref_images.ref_image_1"] = ["tref", 0]          # <Picture 2>
        note = (" The guide subject to replace is the woman shown in <Picture 2>, exactly as she appears in the "
                "guide video: she alone becomes <Subject 1>. <Picture 2> only identifies who to replace; her face, "
                "hair and clothing are not used. Every other person in the guide video stays exactly as it is.  summary:")
        prompt = prompt.replace("  summary:", note, 1)
        wf["r2v"]["inputs"]["prompt"] = prompt
        (out / "prompt.txt").write_text(prompt + "\n")
    if args.protect_only and args.protect:
        wf = add_protect_only(wf, args.protect, 0.7, args.protect_grow, f"bfs/{safe}")
    elif args.subject:
        wf = add_sam_mask(wf, args.subject, args.sam_threshold, args.mask_grow, f"bfs/{safe}", args.protect)
    if args.guide_video:
        gname = f"bfs_{safe}_guide.mp4"
        shutil.copy2(args.guide_video, inp / gname)
        wf["gvid"] = {"class_type": "LoadVideo", "inputs": {"file": gname}}
        wf["gparts"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["gvid", 0]}}
        wf["guide"]["inputs"]["image"] = ["gparts", 0]         # the model follows the cleaned guide
        # source latent, SAM target mask and output audio keep using the original video ("parts")
    import json
    scene_st = None
    if args.ref_color and args.ref_color_scene:
        from utils import scene_color as scol
        mode = args.ref_color_mode
        src_st, abf = None, (0.3 if mode == "scene_lum" else 1.0)
        scene_st = scol.skin_stats(args.video) if mode == "skin" else None
        if mode == "skin" and scene_st is not None:
            import cv2 as _cv2                      # source = the character's face skin (largest face in her refs)
            import numpy as np
            cands = [inp / rname] + [Path(x) for x in args.ref_extra.split(",") if x]
            best = None
            for cp in cands:
                pxs = scol.face_pixels(_cv2.imread(str(cp)))
                if pxs is not None and (best is None or len(pxs) > len(best)):
                    best = pxs
            src_st = (best.mean(0), np.maximum(best.std(0), 1e-3)) if best is not None else None
            if src_st is None:
                scene_st = None
        if scene_st is None:
            mode = "scene" if mode == "skin" else mode
            scene_st = scol.scene_stats(args.video)
        print(f"reference colour mode {mode} ({scene_st[2]:.0%} of sampled pixels used)", flush=True)

        def scene_recolor(src_path: Path, name: str) -> str:
            _, st_ = (args.ref_color.split(":") + ["0.6"])[:2]
            o = scol.recolor(src_path, scene_st, float(st_), out / f"refscene_{name}.png", ab_factor=abf,
                             src_stats=src_st)
            shutil.copy2(o, inp / f"bfs_{safe}_scene_{name}.png")
            return f"bfs_{safe}_scene_{name}.png"

        wf["ref"]["inputs"]["image"] = scene_recolor(inp / rname, "ref")
    if args.ref_color and scene_st is None:
        method, strength = (args.ref_color.split(":") + ["0.6"])[:2]
        wf["refct"] = {"class_type": "ColorTransfer",
                       "inputs": {"image_target": ["ref", 0], "image_ref": ["parts", 0], "method": method,
                                  "source_stats": "uniform", "strength": float(strength)}}
        wf["r2v"]["inputs"]["ref_images.ref_image_0"] = ["refct", 0]
        wf["refct_save"] = {"class_type": "SaveImage",
                            "inputs": {"images": ["refct", 0], "filename_prefix": f"bfs/refcolor_{safe}"}}
    extras = [Path(x) for x in args.ref_extra.split(",") if x]
    if extras:
        for j, xp in enumerate(extras, 1):
            xname = f"bfs_{safe}_view{j}{xp.suffix}"
            shutil.copy2(xp, inp / xname)
            if scene_st is not None:
                xname = scene_recolor(xp, f"view{j}")
            wf[f"xref{j}"] = {"class_type": "LoadImage", "inputs": {"image": xname}}
            src = [f"xref{j}", 0]
            if args.ref_color and scene_st is None:
                wf[f"xct{j}"] = {"class_type": "ColorTransfer",
                                 "inputs": {"image_target": src, "image_ref": ["parts", 0], "method": method,
                                            "source_stats": "uniform", "strength": float(strength)}}
                src = [f"xct{j}", 0]
            wf["r2v"]["inputs"][f"ref_images.ref_image_{j}"] = src
        n = len(extras) + 1
        pr = wf["r2v"]["inputs"]["prompt"]
        pr = pr.replace("shown in <Picture 1>,", f"shown in <Picture 1> to <Picture {n}>, which show the same person "
                        "from different angles (front, three-quarter, side and back views),", 1)
        pr = pr.replace("complete identity from <Picture 1>", f"complete identity from <Picture 1> to <Picture {n}>", 1)
        wf["r2v"]["inputs"]["prompt"] = pr
        (out / "prompt.txt").write_text(pr + "\n")
    if args.realism > 0:
        wf["real"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": ["swap", 0], "lora_name": REALISM_LORA, "strength_model": args.realism}}
        wf["guider"]["inputs"]["model"] = ["real", 0]
        wf["sched"]["inputs"]["model"] = ["real", 0]
        wf["r2v"]["inputs"]["prompt"] = "r34l1sm, " + wf["r2v"]["inputs"]["prompt"]
        (out / "prompt.txt").write_text(wf["r2v"]["inputs"]["prompt"] + "\n")
    if args.upscale and args.upscale > 1.0:
        base = ["swap", 0]
        real = ["real", 0] if args.realism > 0 else base
        s1 = real if args.realism_stage in ("1", "both") else base
        s2 = real if args.realism_stage in ("2", "both") else base
        wf["guider"]["inputs"]["model"] = s1
        wf["sched"]["inputs"]["model"] = s1
        w2, h2 = add_upscale(wf, args.upscale, args.width, args.height, args.frames, s2, chunk_frames=args.chunk_frames)
        if args.from_gen:
            # stage 2 only: the stage-1 result is an existing clip; encode it and use it instead of a new stage 1
            gname = f"bfs_{safe}_stage1.mp4"
            shutil.copy2(args.from_gen, inp / gname)
            wf["s1vid"] = {"class_type": "LoadVideo", "inputs": {"file": gname}}
            wf["s1parts"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["s1vid", 0]}}
            wf["s1enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["s1parts", 0], "vae": ["vvae", 0]}}
            wf["s1empty_sep"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["empty", 0]}}
            wf["up"]["inputs"]["latent"] = ["s1enc", 0]
            wf["s2_cat"]["inputs"]["audio_latent"] = ["s1empty_sep", 1]
            print(f"stage 2 only, from {args.from_gen}", flush=True)
        print(f"two-stage: {args.width}x{args.height} -> latent upscale x{args.upscale} -> {w2}x{h2}, "
              f"realism in stage {args.realism_stage}", flush=True)
    (out / "workflow_api.json").write_text(json.dumps(wf, indent=2))
    c = cs.Comfy(args.comfy_url)
    pid = c.submit(wf)
    print(f"submitted {pid}", flush=True)
    hist = c.wait(pid)
    print()
    for f in c.download_outputs(hist, out):
        print("saved", f)


if __name__ == "__main__":
    main()
