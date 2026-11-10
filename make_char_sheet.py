#!/usr/bin/env python3
"""Character sheet with Qwen-Image-2.1 (+ lightx2v Lightning 4-step LoRA): the character from a photo, dressed in
the outfit from the video, rendered from several angles on white. The views become MiniMax H3 references
(CHAR_REF + CHAR_REF_EXTRA in run_swap.sh), so BFS sees the identity and the outfit from front, side and back.

  python make_char_sheet.py --person cases/x/input/full_body_cutout.png --outfit cases/x/input/outfit_ref.jpg \
      --face cases/x/input/face_cutout.png --out cases/x/input/sheet
Writes <out>/view_<name>.png per angle and <out>/sheet.png (all views side by side, for viewing).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import character_swap as cs  # noqa: E402

VIEWS = {   # name -> camera description (Chinese, like the model's own multi-reference examples)
    "front": "正面视角，人物面向镜头",
    "front_left": "左前方45度视角，人物身体转向画面左侧，脸部仍然可见",
    "left": "左侧面视角，人物完全侧身朝向画面左侧，可见侧脸轮廓",
    "back": "背面视角，人物背对镜头，看不到脸，可见后脑的发型和服装背面",
    "front_right": "右前方45度视角，人物身体转向画面右侧，脸部仍然可见",
}

# stage 1: dress the person (one image, front view). Image 2 is the garment only (head cropped off).
DRESS = ("图1提供人物身份（五官、脸型、发型、发色和体型），图2只提供服装（不提供人物），图3提供面部细节。"
         "画面中只有一个人：图1的人物脱下她原来的衣服，换上图2的服装，{view}。"
         "服装完全保持图2的款式、剪裁、花纹和颜色：{outfit}；不保留图1原来的上衣和短裤。"
         "保持图1和图3人物的五官、发型和发色不变。人物自然站立，双臂自然下垂，从头到脚完整入镜，"
         "纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
         "每张输入图只提供指定的信息，不引入其余背景或物体。只生成一个人物。输出一张完整图片。")
# stage 2: the same person and outfit from another camera angle (image 1 = stage-1 result).
ROTATE = ("图1是一个人物的全身照。生成同一个人物、同一套服装的全身照，改为{view}。"
          "保持图1人物的五官、发型、发色、体型以及服装的款式、花纹、颜色和穿法完全不变，只改变相机角度。"
          "人物自然站立，从头到脚完整入镜，纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
          "画面中只有一个人物。输出一张完整图片。")


def build(images: list, prompt: str, seed: int, prefix: str) -> dict:
    wf = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "attn": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["unet", 0], "attention": "pytorch attention"}},
        "lightning": {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": ["attn", 0], "lora_name": "qwen_image_2.1_lightning_4steps_lightx2v.safetensors",
                                 "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "enc": {"class_type": "TextEncodeQwenImage21",
                "inputs": {"clip": ["clip", 0], "prompt": prompt, "negative_prompt": "", "vae": ["vae", 0],
                           "resolution": 1024}},
        "ks": {"class_type": "KSampler",
               "inputs": {"model": ["lightning", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "positive": ["enc", 0], "negative": ["enc", 1],
                          "latent_image": ["enc", 2], "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}},
    }
    for i, name in enumerate(images, 1):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["enc"]["inputs"][f"images.image_{i}"] = [f"img{i}", 0]
    return wf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", type=Path, required=True, help="full-body photo of the character (cut-out on white)")
    ap.add_argument("--outfit", type=Path, required=True, help="image of the outfit to wear (e.g. a video frame)")
    ap.add_argument("--face", type=Path, required=True, help="face close-up of the character")
    ap.add_argument("--outfit-text", default="红色和服，红白相间的波浪花纹，单肩露出，宽大的白色衣领，花纹腰带，垂下的黄色系带")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reuse-front", action="store_true", help="keep an existing <out>/view_front.png (stage 1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--comfy-urls", default="http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inp = ROOT / "ComfyUI" / "input"
    names = {}
    for key, p in (("p", args.person), ("o", args.outfit), ("f", args.face)):
        names[key] = f"sheet_{key}{p.suffix}"
        shutil.copy2(p, inp / names[key])
    urls = args.comfy_urls.split(",")
    views = [v for v in args.views.split(",") if v != "front"]

    def gen(k, images, prompt, view):
        c = cs.Comfy(urls[k % len(urls)])
        h = c.wait(c.submit(build(images, prompt, args.seed, f"sheet/{view}")), poll=2)
        tmp = args.out / f"_{view}"
        tmp.mkdir(parents=True, exist_ok=True)
        files = [f for f in c.download_outputs(h, tmp) if f.suffix == ".png"]
        dst = args.out / f"view_{view}.png"
        shutil.move(str(files[-1]), dst)
        shutil.rmtree(tmp, ignore_errors=True)
        (args.out / f"prompt_{view}.txt").write_text(prompt + "\n")
        print(f"{view}: {dst}", flush=True)
        return dst

    # stage 1: front view, dressed
    front = args.out / "view_front.png"
    if not (args.reuse_front and front.exists()):
        front = gen(0, [names["p"], names["o"], names["f"]], DRESS.format(view=VIEWS["front"], outfit=args.outfit_text),
                    "front")
    names["front"] = "sheet_front.png"
    shutil.copy2(front, inp / names["front"])
    # stage 2: rotate the dressed front view
    with ThreadPoolExecutor(len(urls)) as ex:
        rest = list(ex.map(lambda kv: gen(kv[0], [names["front"]], ROTATE.format(view=VIEWS[kv[1]]), kv[1]),
                           enumerate(views)))
    outs = [front] + rest
    ims = [cv2.imread(str(p)) for p in outs]
    H = max(i.shape[0] for i in ims)
    ims = [cv2.resize(i, (int(i.shape[1] * H / i.shape[0]), H)) for i in ims]
    cv2.imwrite(str(args.out / "sheet.png"), np.hstack(ims))
    for v in names.values():
        (inp / v).unlink(missing_ok=True)
    print(f"sheet: {args.out / 'sheet.png'}")


if __name__ == "__main__":
    main()
