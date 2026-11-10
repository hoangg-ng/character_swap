#!/usr/bin/env python3
"""Character swap with MiniMax H3 + akatz-ai/MiniMax-H3-Character-Swap-LoRA (V1.5).

Pipeline
  1. probe + normalise the source video (trim, h264/aac mp4, scene-cut check)
  2. normalise the character reference image (PNG)
  3. build the H3 prompt and the SAM 3.1 subject text
  4. patch the official H3 Relay V2 API workflow (workflows/H3-Character-Swap-Relay-V2.api.json)
  5. submit to a running ComfyUI (start_comfy.sh), wait, download the result + mask preview

Example
  python character_swap.py --video inputs/clip.mp4 --character inputs/girl.png \
      --subject "man in the blue shirt" \
      --character-desc "a young woman with long black hair, wearing a white blouse and jeans"

Only the Python standard library is required (plus ffmpeg/ffprobe on PATH).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKFLOW = ROOT / "workflows" / "H3-Character-Swap-Relay-V2.api.json"
PROMPT_TEMPLATE = ROOT / "prompts" / "swap_prompt.txt"
PROMPT_TEMPLATE_DEPTH = ROOT / "prompts" / "swap_prompt_depth.txt"
PROMPT_TEMPLATE_CN = ROOT / "prompts" / "swap_prompt_cn.txt"
CONTROLNET = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"

CHARSWAP_LORA = "h3_character_swap_v1.5.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"


# --------------------------------------------------------------------------- utils
def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout + r.stderr


def probe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", str(path)])
    info = json.loads(out[out.index("{"):])
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        sys.exit(f"{path}: no video stream")
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    return {
        "width": int(v["width"]), "height": int(v["height"]),
        "fps": float(num) / float(den) if float(den) else 0.0,
        "duration": float(info["format"].get("duration", 0.0)),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def count_scene_cuts(path: Path, threshold: float) -> list[float]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-an",
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", out)]


# --------------------------------------------------------------------------- inputs
def target_size(w: int, h: int, short: int) -> tuple[int, int]:
    """32-aligned size whose short side is closest to `short`, keeping aspect ratio."""
    a32 = lambda x: max(32, int(x / 32.0 + 0.5) * 32)
    up32 = lambda x: -(-int(x) // 32) * 32          # short side rounds UP -> final step only downsamples
    if w >= h:
        th = up32(short); tw = a32(w * th / h)
    else:
        tw = up32(short); th = a32(h * tw / w)
    return tw, th


def prepare_video(src: Path, dst: Path, start: float, duration: float,
                  size: tuple[int, int] | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start}"]
    cmd += ["-i", str(src)]
    if duration > 0:
        cmd += ["-t", f"{duration}"]
    # even dims for h264; H3 Relay itself resamples to 24 fps and aligns to 32 px
    vf = f"scale={size[0]}:{size[1]}:flags=lanczos,setsar=1" if size else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd += ["-vf", vf, "-c:v", "libx264", "-crf", "12",
            "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(dst)]
    run(cmd)


def prepare_reference(src: Path, dst: Path, max_side: int) -> None:
    # flatten any alpha onto white (ComfyUI LoadImage would treat alpha as a mask)
    vf = (f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))',"
          "format=rgba,split[a][b];[a]drawbox=c=white:t=fill[bg];[bg][b]overlay=format=auto,format=rgb24")
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-frames:v", "1", "-filter_complex", vf, str(dst)])


def build_prompt(args) -> str:
    if args.prompt:
        return args.prompt.strip()
    tpl = {"depth-mesh": PROMPT_TEMPLATE_DEPTH, "depth-mesh-cn": PROMPT_TEMPLATE_CN}.get(
        args.guide, PROMPT_TEMPLATE).read_text()
    desc = args.character_desc.strip().rstrip(".") if args.character_desc else ""
    return tpl.format(
        character_noun=args.character_noun,
        character_desc_sentence=f" The reference character is {desc}." if desc else "",
        emotion=args.emotion,
    ).strip()


# --------------------------------------------------------------------------- workflow
def patch_workflow(wf: dict, *, video_name: str, ref_name: str, prompt: str, args) -> dict:
    wf = copy.deepcopy(wf)

    def node(nid: str) -> dict:
        if nid not in wf:
            sys.exit(f"workflow has no node '{nid}' - did the upstream JSON change?")
        return wf[nid]["inputs"]

    node("source")["file"] = video_name
    node("reference")["image"] = ref_name
    node("reference_conditioning")["prompt"] = prompt
    node("subject")["text"] = args.subject                     # SAM 3.1 text query
    node("detect")["threshold"] = args.sam_threshold
    node("select")["selection"] = args.selection
    node("select")["index"] = args.index
    node("expand_mask")["expansion_px"] = args.mask_expand
    node("prepare")["resolution"] = args.budget
    node("prepare")["duration_seconds"] = 0.0                  # trimming is done by ffmpeg
    node("plan")["seed"] = args.seed
    node("adapter")["lora_name"] = args.charswap_lora
    node("adapter")["strength_model"] = args.lora_strength
    node("turbo")["lora_name"] = args.turbo_lora
    if args.turbo == "larryvrh":
        # larryvrh/MiniMax-H3-Turbo-Lora: own LoRA node (re-injects time conditioning on the pruned
        # base) + own few-step sampler; scheduler 'simple'. Needs ComfyUI-MiniMax-H3-Turbo nodes.
        wf["turbo"] = {"class_type": "MiniMaxH3TurboLoRA",
                       "inputs": {"model": ["attention", 0], "lora_name": args.turbo_lora_larryvrh,
                                  "strength": args.turbo_strength, "low_vram": False}}
        wf["sampler"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
        node("schedule")["scheduler"] = "simple"
    node("schedule")["steps"] = args.steps or (6 if args.turbo == "larryvrh" else 8)
    node("sampling_memory")["activation_memory_gib"] = args.activation_memory_gib
    node("text_device")["device"] = args.text_device
    node("attention")["attention"] = args.attention
    node("save")["filename_prefix"] = f"character-swap/{args.job}"
    return wf


MASK_NODES = ["source", "prepare", "sam", "subject", "first_frame", "detect", "select", "track",
              "mask", "recover", "expand_mask"]


OCC_NODES = ["occ_subject", "occ_track", "occ_mask", "occ_grow", "tight_final", "exp_final"]


def add_occluder(wf: dict, text: str, threshold: float, grow: int) -> dict:
    """Track a foreground occluder (e.g. the man passing in front) with SAM 3.1 text detection on
    every frame and subtract it from the target masks. Where the occluder is in front, nothing is
    regenerated (source latent kept) and nothing is pasted in the composite."""
    wf = copy.deepcopy(wf)
    wf["occ_subject"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": text}}
    wf["occ_track"] = {"class_type": "SAM3_VideoTrack",
                       "inputs": {"images": ["prepare", 0], "model": ["sam", 0], "detection_threshold": threshold,
                                  "max_objects": 2, "detect_interval": 1, "conditioning": ["occ_subject", 0]}}
    wf["occ_mask"] = {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["occ_track", 0],
                                                                    "object_indices": ""}}
    wf["occ_grow"] = {"class_type": "GrowMask", "inputs": {"mask": ["occ_mask", 0], "expand": grow,
                                                           "tapered_corners": True}}
    for nid, dst in (("tight_final", ["recover", 0]), ("exp_final", ["expand_mask", 0])):
        wf[nid] = {"class_type": "MaskComposite", "inputs": {"destination": dst, "source": ["occ_grow", 0],
                                                             "x": 0, "y": 0, "operation": "subtract"}}
    wf["window"]["inputs"]["mask"] = ["exp_final", 0]
    return wf


def mask_workflow(wf: dict, prefix: str) -> dict:
    """Stage A: only source prep + SAM 3.1 tracking; save prepared frames and both masks as PNGs."""
    occ = "tight_final" in wf
    m = {k: copy.deepcopy(wf[k]) for k in MASK_NODES + (OCC_NODES if occ else [])}
    m["tight_img"] = {"class_type": "MaskToImage",
                      "inputs": {"mask": ["tight_final" if occ else "recover", 0]}}
    m["exp_img"] = {"class_type": "MaskToImage",
                    "inputs": {"mask": ["exp_final" if occ else "expand_mask", 0]}}
    m["raw_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ["recover", 0]}}
    saves = [("save_frames", ["prepare", 0]), ("save_tight", ["tight_img", 0]),
             ("save_exp", ["exp_img", 0]), ("save_raw", ["raw_img", 0])]
    if occ:
        m["occ_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ["occ_mask", 0]}}
        saves.append(("save_occ", ["occ_img", 0]))
    for nid, src in saves:
        m[nid] = {"class_type": "SaveImage",
                  "inputs": {"images": src, "filename_prefix": f"{prefix}/{nid}"}}
    return m


def add_guide(wf: dict, guide_name: str) -> dict:
    """Stage B: feed the depth/mesh guide video as <Video 1>; the latent outside the mask
    is still encoded from the original source (node 'encode_source')."""
    wf = copy.deepcopy(wf)
    wf["guide_source"] = {"class_type": "LoadVideo", "inputs": {"file": guide_name}}
    wf["guide_prepare"] = {"class_type": "H3RelayPrepareSource",
                           "inputs": {**wf["prepare"]["inputs"], "video": ["guide_source", 0]}}
    gw = copy.deepcopy(wf["window"]["inputs"])
    gw["source"] = ["guide_prepare", 0]
    gw.pop("mask", None)
    wf["guide_window"] = {"class_type": "H3RelayWindowInputs", "inputs": gw}
    wf["reference_conditioning"]["inputs"]["ref_videos.ref_video_0"] = ["guide_window", 5]
    return wf


def add_custom_edit_mask(wf: dict, file_name: str) -> dict:
    """Replace the workflow's internal edit mask by our own mask sequence (lossless video)."""
    wf = copy.deepcopy(wf)
    wf["cm_video"] = {"class_type": "LoadVideo", "inputs": {"file": file_name}}
    wf["cm_comp"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["cm_video", 0]}}
    wf["cm_mask"] = {"class_type": "ImageToMask", "inputs": {"image": ["cm_comp", 0], "channel": "red"}}
    wf["window"]["inputs"]["mask"] = ["cm_mask", 0]
    return wf


def add_ref_video(wf: dict, file_name: str, prefix: str = "comp") -> dict:
    """Use `file_name` (same timeline as the source) as <Video 1> instead of the original footage."""
    wf = copy.deepcopy(wf)
    wf[f"{prefix}_source"] = {"class_type": "LoadVideo", "inputs": {"file": file_name}}
    wf[f"{prefix}_prepare"] = {"class_type": "H3RelayPrepareSource",
                               "inputs": {**wf["prepare"]["inputs"], "video": [f"{prefix}_source", 0]}}
    w = copy.deepcopy(wf["window"]["inputs"])
    w["source"] = [f"{prefix}_prepare", 0]
    w.pop("mask", None)
    wf[f"{prefix}_window"] = {"class_type": "H3RelayWindowInputs", "inputs": w}
    wf["reference_conditioning"]["inputs"]["ref_videos.ref_video_0"] = [f"{prefix}_window", 5]
    return wf


def add_controlnet(wf: dict, guide_name: str, strength: float, end: float) -> dict:
    """ControlNet mode: <Video 1> stays the ORIGINAL footage; the depth + face-mesh video drives
    the H3 Fun ControlNet Union as a control signal (not as pixels to reproduce)."""
    wf = copy.deepcopy(wf)
    wf["guide_source"] = {"class_type": "LoadVideo", "inputs": {"file": guide_name}}
    wf["guide_prepare"] = {"class_type": "H3RelayPrepareSource",
                           "inputs": {**wf["prepare"]["inputs"], "video": ["guide_source", 0]}}
    wf["cn_patch"] = {"class_type": "ModelPatchLoader", "inputs": {"name": CONTROLNET}}
    wf["controlnet"] = {"class_type": "MiniMaxH3FunControlNetApply",
                        "inputs": {"model": ["adapter", 0], "model_patch": ["cn_patch", 0],
                                   "vae": ["video_vae", 0], "strength": strength,
                                   "start_percent": 0.0, "end_percent": end,
                                   "control_video": ["guide_prepare", 0]}}
    wf["sampling_memory"]["inputs"]["model"] = ["controlnet", 0]
    return wf


# --------------------------------------------------------------------------- comfy API
class Comfy:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _req(self, path: str, data: dict | None = None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.url + path, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def alive(self) -> bool:
        try:
            self._req("/system_stats")
            return True
        except Exception:
            return False

    def submit(self, wf: dict) -> str:
        try:
            res = json.loads(self._req("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex}))
        except urllib.error.HTTPError as e:
            sys.exit(f"ComfyUI rejected the workflow:\n{e.read().decode()[:4000]}")
        if res.get("node_errors"):
            sys.exit(f"node errors: {json.dumps(res['node_errors'], indent=2)}")
        return res["prompt_id"]

    def wait(self, pid: str, poll: float = 10.0) -> dict:
        t0 = time.time()
        while True:
            hist = json.loads(self._req(f"/history/{pid}"))
            if pid in hist:
                h = hist[pid]
                st = h.get("status", {})
                if st.get("status_str") == "error":
                    msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                    sys.exit(f"execution error:\n{json.dumps(msgs, indent=2)[:4000]}")
                if st.get("completed", True):
                    return h
            print(f"\r  running... {time.time() - t0:6.0f}s", end="", flush=True)
            time.sleep(poll)

    def download_outputs(self, hist: dict, out_dir: Path) -> list[Path]:
        saved = []
        for nid, outs in hist.get("outputs", {}).items():
            for items in outs.values():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not (isinstance(it, dict) and "filename" in it):
                        continue
                    q = urllib.parse.urlencode({"filename": it["filename"],
                                                "subfolder": it.get("subfolder", ""),
                                                "type": it.get("type", "output")})
                    dst = out_dir / f"{nid}_{Path(it['filename']).name}"
                    dst.write_bytes(self._req(f"/view?{q}"))
                    saved.append(dst)
        return saved


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--video", required=True, type=Path, help="source video (one continuous shot)")
    g.add_argument("--character", required=True, type=Path, help="reference image of the new character")
    g.add_argument("--subject", default="person",
                   help="SAM 3.1 text query for the person to replace, e.g. 'man in red jacket'")
    g.add_argument("--start", type=float, default=0.0, help="trim start (s)")
    g.add_argument("--duration", type=float, default=0.0, help="trim length (s); 0 = full clip")

    g = p.add_argument_group("prompt")
    g.add_argument("--character-noun", default="woman", help="noun used in the prompt template")
    g.add_argument("--character-desc", default="",
                   help="appearance of the reference character (hair, outfit, ...); strengthens identity")
    g.add_argument("--prompt", default="", help="full prompt override (use <Video 1> / <Picture 1>)")
    g.add_argument("--emotion", default="the same emotion as the original performance",
                   help="expression wording for the depth-mesh prompt, e.g. 'crying, tearful eyes, trembling lips'")
    g.add_argument("--cn-strength", type=float, default=0.6, help="ControlNet strength (depth-mesh-cn)")
    g.add_argument("--cn-end", type=float, default=1.0, help="ControlNet end percent (depth-mesh-cn)")
    g.add_argument("--face-flat", action="store_true",
                   help="guide: flat face + expression contours only (less identity leakage from the source face)")
    g.add_argument("--guide", default="none", choices=["none", "depth-mesh", "depth-mesh-cn"],
                   help="depth-mesh: replace the person inside the mask in <Video 1> by a Depth Anything V2 "
                        "depth map + MediaPipe face mesh, so the model gets motion/expression, not identity")
    g.add_argument("--ref-video", default="original", choices=["original", "composite"],
                   help="<Video 1>: original footage, or 'composite' = actress removed (inpainted) and the "
                        "reference character pasted in her place, face warped to her face mesh")
    g.add_argument("--composite-ref", type=Path, default=None,
                   help="image used for the composite (default: --character); a full/upper-body photo works best")
    g.add_argument("--composite-relight", type=float, default=0.0,
                   help="composite: 0..1 actress's face shading (light direction/exposure) on our character")
    g.add_argument("--composite-skin", type=float, default=0.0,
                   help="composite: 0..1 match skin tone to the actress's")
    g.add_argument("--composite-hair", default="paste", choices=["paste", "recolor", "paste-flow"],
                   help="composite hair: rigid paste of the reference (stiff) or the actress's own hair recoloured "
                        "(natural motion)")
    g.add_argument("--composite-clothes-L", type=float, default=-1,
                   help="composite (recolor mode): recolour the actress's clothes to this LAB lightness, "
                        "e.g. 35 for a black top; -1 = leave")
    g.add_argument("--custom-masks", action="store_true",
                   help="build occlusion-aware masks (SAM target + detected face - opaque core of SAM "
                        "occluder/near-depth) and use them for composite, guide, edit mask and bg restore")
    g.add_argument("--occ-erode", type=int, default=14)
    g.add_argument("--near-depth", type=float, default=0.15)
    g.add_argument("--restore-bg", type=int, default=0,
                   help="after generation, restore the source outside the person mask + this many px "
                        "(needs a guide/composite mode; removes the 32px-block background halo)")
    g.add_argument("--composite-detail", type=float, default=0.0,
                   help="composite: 0..1 actress high-frequency face detail (expression) over the reference face")
    g.add_argument("--guide-gpu", default="0",
                   help="nvidia-smi index for the depth/face-mesh models (default 0 = the A4500)")

    g = p.add_argument_group("mask")
    g.add_argument("--occluder", default="",
                   help="SAM text for a foreground person/object passing IN FRONT of the target (e.g. 'man'); "
                        "it is tracked every frame and subtracted from the target mask")
    g.add_argument("--occluder-threshold", type=float, default=0.7,
                   help="keep high: lower values make SAM match the target person too")
    g.add_argument("--occluder-grow", type=int, default=6, help="px dilation of the occluder mask")
    g.add_argument("--selection", default="largest", choices=["largest", "leftmost", "rightmost", "index"],
                   help="which SAM detection to replace when several people are found")
    g.add_argument("--index", type=int, default=0, help="mask index for index-based selection")
    g.add_argument("--sam-threshold", type=float, default=0.35)
    g.add_argument("--mask-expand", type=int, default=64, help="mask dilation in px")

    g = p.add_argument_group("generation")
    g.add_argument("--resolution", default="768",
                   help="'480' (~0.5 MP budget), '768' (~1 MP budget), or a short-side size such as "
                        "720: the source is pre-scaled to a 32-aligned size near it (e.g. 1280x736), "
                        "generated under the matching budget, and the result rescaled to exactly 720p")
    g.add_argument("--seed", type=int, default=904239)
    g.add_argument("--lora-strength", type=float, default=1.0)
    g.add_argument("--charswap-lora", default=CHARSWAP_LORA)
    g.add_argument("--turbo-lora", default=TURBO_LORA)
    g.add_argument("--turbo", default="lightx2v", choices=["lightx2v", "larryvrh"],
                   help="distillation LoRA: lightx2v FL2V 8-step (what the swap LoRA was trained with) or "
                        "larryvrh/MiniMax-H3-Turbo-Lora (4-8 steps; needs the ComfyUI-MiniMax-H3-Turbo nodes)")
    g.add_argument("--turbo-lora-larryvrh", default="minimax_h3_turbo_v4_step600_ema.safetensors")
    g.add_argument("--turbo-strength", type=float, default=1.0)
    g.add_argument("--steps", type=int, default=0, help="sampling steps (0 = 8 for lightx2v, 6 for larryvrh)")
    g.add_argument("--activation-memory-gib", type=float, default=17.0)
    g.add_argument("--max-ref-side", type=int, default=1536)
    g.add_argument("--attention", default="pytorch attention",
                   choices=["pytorch attention", "comfy kitchen attention"],
                   help="'comfy kitchen attention' (INT8, faster) needs an NVIDIA driver with CUDA >= 13; "
                        "this machine's driver 570 / CUDA 12.8 only supports 'pytorch attention'")
    g.add_argument("--text-device", default="default",
                   help="text-encoder device: 'default', 'cpu', or 'gpu:N' when ComfyUI sees several GPUs")

    g = p.add_argument_group("runtime")
    g.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    g.add_argument("--comfy-dir", type=Path, default=ROOT / "ComfyUI")
    g.add_argument("--job", default="", help="job name (default: <video>_<timestamp>)")
    g.add_argument("--out", type=Path, default=ROOT / "outputs")
    g.add_argument("--dry-run", action="store_true", help="prepare inputs + workflow only, don't submit")
    args = p.parse_args()

    for f in (args.video, args.character):
        if not f.is_file():
            sys.exit(f"not found: {f}")
    args.job = args.job or f"{args.video.stem}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    job_dir = args.out / args.job
    job_dir.mkdir(parents=True, exist_ok=True)

    # 1. video
    src = probe(args.video)
    print(f"[1/5] source: {src['width']}x{src['height']} @ {src['fps']:.2f} fps, "
          f"{src['duration']:.2f}s, audio={src['has_audio']}")
    video = job_dir / "source.mp4"
    gen_size, final_size = None, None
    if args.resolution in ("480", "768"):
        args.budget = args.resolution
    elif args.resolution.isdigit():
        short = int(args.resolution)
        gen_size = target_size(src["width"], src["height"], short)
        px = gen_size[0] * gen_size[1]
        if px > 1_000_000:
            sys.exit(f"{gen_size[0]}x{gen_size[1]} exceeds the H3 Relay 1 MP budget; use a smaller --resolution")
        args.budget = "480" if px <= 500_000 else "768"
        if src["width"] >= src["height"]:
            final_size = (int(round(src["width"] * short / src["height"] / 2)) * 2, short)
        else:
            final_size = (short, int(round(src["height"] * short / src["width"] / 2)) * 2)
        print(f"  custom resolution {short}: generate at {gen_size[0]}x{gen_size[1]} "
              f"(budget {args.budget}), deliver {final_size[0]}x{final_size[1]}")
    else:
        sys.exit("--resolution must be 480, 768 or a number like 720")
    prepare_video(args.video, video, args.start, args.duration, gen_size)
    vinfo = probe(video)
    if vinfo["duration"] < 5.2:
        print(f"  WARNING: {vinfo['duration']:.2f}s is shorter than one H3 window (124 frames @ 24fps "
              "~5.2s). Use a clip >= 5.2s.")
    if vinfo["duration"] > 15.0:
        print("  note: >15s -> H3 Relay will generate several continuation windows (slower, joins may drift)")
    cuts = count_scene_cuts(video, 0.4)
    if cuts:
        print(f"  WARNING: possible scene cuts at {', '.join(f'{c:.2f}s' for c in cuts)}. "
              "The LoRA expects ONE continuous shot - trim with --start/--duration.")

    # 2. reference
    ref = job_dir / "reference.png"
    prepare_reference(args.character, ref, args.max_ref_side)
    print(f"[2/5] reference -> {ref}")

    # 3. prompt
    prompt = build_prompt(args)
    (job_dir / "prompt.txt").write_text(prompt + "\n")
    print(f"[3/5] prompt:\n      " + prompt.replace("\n", "\n      "))
    print(f"      SAM subject: {args.subject!r}")

    # 4. workflow
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.job)
    video_name, ref_name = f"charswap_{safe}.mp4", f"charswap_{safe}.png"
    wf = patch_workflow(json.loads(WORKFLOW.read_text()), video_name=video_name,
                        ref_name=ref_name, prompt=prompt, args=args)
    if args.occluder:
        wf = add_occluder(wf, args.occluder, args.occluder_threshold, args.occluder_grow)
    (job_dir / "workflow_api.json").write_text(json.dumps(wf, indent=2))
    (job_dir / "job.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "source_probe": src,
         "prepared_probe": vinfo, "scene_cuts": cuts, "prompt": prompt}, indent=2))
    print(f"[4/5] workflow -> {job_dir / 'workflow_api.json'}")
    if args.dry_run:
        print("dry run: stopping before submission")
        return

    # 5. run
    comfy = Comfy(args.comfy_url)
    if not comfy.alive():
        sys.exit(f"ComfyUI not reachable at {args.comfy_url}. Run: bash start_comfy.sh")
    input_dir = args.comfy_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, input_dir / video_name)
    shutil.copy2(ref, input_dir / ref_name)
    if args.guide in ("depth-mesh", "depth-mesh-cn") or args.ref_video == "composite" or args.custom_masks:
        gdir = job_dir / "guide"
        if gdir.exists():
            shutil.rmtree(gdir)
        raw = gdir / "raw"
        raw.mkdir(parents=True)
        print("[5a] SAM mask pass ...", flush=True)
        hist_a = comfy.wait(comfy.submit(mask_workflow(wf, f"guide/{safe}")))
        print()
        for f in comfy.download_outputs(hist_a, raw):
            for k in ("save_frames", "save_tight", "save_exp", "save_raw", "save_occ"):
                if f.name.startswith(k + "_"):
                    (gdir / k).mkdir(exist_ok=True)
                    f.rename(gdir / k / f.name)
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.guide_gpu}
        if args.custom_masks:
            print("[5m] occlusion-aware masks ...", flush=True)
            cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "make_masks.py"),
                   "--frames", str(gdir / "save_frames"), "--raw", str(gdir / "save_raw"),
                   "--out", str(gdir / "masks"), "--expand", str(args.mask_expand),
                   "--occ-erode", str(args.occ_erode), "--near", str(args.near_depth)]
            if (gdir / "save_occ").is_dir():
                cmd += ["--occ", str(gdir / "save_occ")]
            if subprocess.run(cmd, env=env).returncode != 0:
                sys.exit("mask generation failed")
            # downstream steps use the occlusion-aware person mask as the "tight" mask
            shutil.rmtree(gdir / "save_tight", ignore_errors=True)
            shutil.copytree(gdir / "masks" / "person", gdir / "save_tight")
            em_name = f"charswap_{safe}_editmask.mp4"
            shutil.copy2(gdir / "masks" / "edit.mp4", input_dir / em_name)
            wf = add_custom_edit_mask(wf, em_name)
            (job_dir / "workflow_api.json").write_text(json.dumps(wf, indent=2))
        if args.ref_video == "composite":
            comp = gdir / "composite.mp4"
            print("[5c] composite: actress removed, reference character inserted ...", flush=True)
            r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "make_composite.py"),
                                "--frames", str(gdir / "save_frames"), "--tight", str(gdir / "save_tight"),
                                "--reference", str(args.composite_ref or args.character), "--out", str(comp),
                                "--detail", str(args.composite_detail),
                                "--relight", str(args.composite_relight), "--skin", str(args.composite_skin),
                                "--hair-mode", args.composite_hair,
                                "--clothes-L", str(args.composite_clothes_L),
                                *(["--occ", str(gdir / "masks" / "occ"), "--alpha-out", str(gdir / "comp_alpha")]
                                  if args.custom_masks else [])],
                               env=env)
            if r.returncode != 0:
                sys.exit("composite generation failed")
            if args.custom_masks and args.composite_hair == "paste-flow" and (gdir / "comp_alpha").is_dir():
                # grow person/edit masks to cover the pasted reference hair, then refresh downstream inputs
                print("[5m+] growing masks to the reference hair ...", flush=True)
                cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "make_masks.py"),
                       "--frames", str(gdir / "save_frames"), "--raw", str(gdir / "save_raw"),
                       "--extra", str(gdir / "comp_alpha"), "--out", str(gdir / "masks"),
                       "--expand", str(args.mask_expand), "--occ-erode", str(args.occ_erode),
                       "--near", str(args.near_depth)]
                if (gdir / "save_occ").is_dir():
                    cmd += ["--occ", str(gdir / "save_occ")]
                if subprocess.run(cmd, env=env).returncode != 0:
                    sys.exit("mask regrow failed")
                shutil.rmtree(gdir / "save_tight", ignore_errors=True)
                shutil.copytree(gdir / "masks" / "person", gdir / "save_tight")
                shutil.copy2(gdir / "masks" / "edit.mp4", input_dir / f"charswap_{safe}_editmask.mp4")
            comp_name = f"charswap_{safe}_composite.mp4"
            shutil.copy2(comp, input_dir / comp_name)
            wf = add_ref_video(wf, comp_name)
            (job_dir / "workflow_api.json").write_text(json.dumps(wf, indent=2))
    if args.guide in ("depth-mesh", "depth-mesh-cn"):
        guide = gdir / "guide.mp4"
        print("[5b] depth + face mesh guide ...", flush=True)
        r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "make_guide.py"),
                            "--frames", str(gdir / "save_frames"), "--tight", str(gdir / "save_tight"),
                            "--expanded", str(gdir / "save_exp"), "--out", str(guide),
                            *(["--full-frame"] if args.guide == "depth-mesh-cn" else []),
                            *(["--face-flat"] if args.face_flat else [])], env=env)
        if r.returncode != 0:
            sys.exit("guide generation failed")
        guide_name = f"charswap_{safe}_guide.mp4"
        shutil.copy2(guide, input_dir / guide_name)
        wf = (add_controlnet(wf, guide_name, args.cn_strength, args.cn_end)
              if args.guide == "depth-mesh-cn" else add_guide(wf, guide_name))
        (job_dir / "workflow_api.json").write_text(json.dumps(wf, indent=2))
    pid = comfy.submit(wf)
    print(f"[5/5] submitted prompt_id={pid}")
    hist = comfy.wait(pid)
    print()
    files = comfy.download_outputs(hist, job_dir)
    for f in files:
        print(f"  saved {f}")
    if final_size:
        for f in files:
            if f.suffix == ".mp4" and f.name.startswith("save_"):
                dst = job_dir / f"final_{final_size[1] if final_size[0] >= final_size[1] else final_size[0]}p.mp4"
                run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(f),
                     "-vf", f"scale={final_size[0]}:{final_size[1]}:flags=lanczos,setsar=1",
                     "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dst)])
                print(f"  rescaled -> {dst}")
                gdir = job_dir / "guide"
                if args.restore_bg > 0 and (gdir / "save_tight").is_dir():
                    raw = dst.with_name(dst.stem + "_raw.mp4")
                    dst.rename(raw)
                    r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "restore_bg.py"),
                                        "--video", str(raw), "--src-frames", str(gdir / "save_frames"),
                                        "--tight", str(gdir / "save_tight"), "--dst", str(dst),
                                        "--margin", str(args.restore_bg)])
                    if r.returncode != 0:
                        raw.rename(dst)
                        print("  WARNING: background restore failed; kept the raw result")
    print(f"done -> {job_dir}")


if __name__ == "__main__":
    main()
