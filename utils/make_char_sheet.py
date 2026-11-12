#!/usr/bin/env python3
"""Character sheet with Qwen-Image-2.1 (+ lightx2v Lightning 4-step LoRA): the character from a photo, dressed in
the outfit from the video, rendered from several angles on white. The views become MiniMax H3 references
(CHAR_REF + CHAR_REF_EXTRA in run_swap.sh), so BFS sees the identity and the outfit from front, side and back.

  python utils/make_char_sheet.py --person cases/x/input/full_body_cutout.png --outfit cases/x/input/outfit_ref.jpg \
      --face cases/x/input/face_cutout.png --out cases/x/input/sheet
Writes <out>/view_<name>.png per angle and <out>/sheet.png (all views side by side, for viewing).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # project root
sys.path.insert(0, str(ROOT))
from utils import comfy_client as cs  # noqa: E402

VIEWS = {   # name -> camera description (Chinese, like the model's own multi-reference examples)
    "front": "正面视角，人物面向镜头",
    "front_left": "相机向右绕人物转45度：人物的身体、脸和脚尖都朝向画面左前方，呈四分之三侧面，能看到她的右侧脸和右肩",
    "left": "相机绕人物转90度，拍摄完全的侧面：人物的身体、脸和脚尖都朝向画面左侧，只看到她的侧脸轮廓、一只眼睛和身体侧面",
    "back": "相机绕到人物正后方拍摄：画面中只看到人物的后背、后脑勺和背后的头发，完全看不到脸和身体正面，脚跟朝向镜头",
    "front_right": "相机向左绕人物转45度：人物的身体、脸和脚尖都朝向画面右前方，呈四分之三侧面",
    "face": "正面胸部以上的特写肖像：人物面向镜头，脸部占据画面大部分，清晰呈现五官、皮肤细节、发型和刘海",
}

# stage 1: dress the person (one image, front view). Image 2 is the garment only (head cropped off).
DRESS = ("图1提供人物身份（五官、脸型、发型、发色和体型），图2只提供服装（不提供人物），图3提供面部细节。"
         "画面中只有一个人：图1的人物脱下她原来的衣服，换上图2的服装，{view}。"
         "服装完全保持图2的款式、剪裁、花纹和颜色：{outfit}；不保留图1原来的服装。"
         "保持图1和图3人物的五官、发型和发色不变。人物自然站立，双臂自然下垂，从头到脚完整入镜，"
         "纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
         "每张输入图只提供指定的信息，不引入其余背景或物体。只生成一个人物。输出一张完整图片。")
# stage 1 without an outfit image: keep the person's own clothes, just a clean neutral front view.
NEUTRAL = ("图1提供人物（五官、脸型、发型、发色、体型和服装），图2提供面部细节。"
           "画面中只有一个人：生成图1人物的全身照，{view}，穿着图1中她自己的服装（保持款式、剪裁、花纹和颜色）。"
           "保持图1和图2人物的五官、发型和发色不变。人物自然站立，双臂自然下垂放在身体两侧，手中不拿任何物品，"
           "从头到脚完整入镜，纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
           "每张输入图只提供指定的信息，不引入其余背景或物体。只生成一个人物。输出一张完整图片。")
# stage 2: the same person and outfit from another camera angle (image 1 = stage-1 result).
ROTATE = ("图1是一个人物的全身照。生成同一个人物、同一套服装的全身照，改为{view}。"
          "保持图1人物的五官、发型、发色、体型以及服装的款式、花纹、颜色和穿法完全不变，只改变相机角度。"
          "人物自然站立，从头到脚完整入镜，纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
          "画面中只有一个人物。输出一张完整图片。")


CLOSEUP = ("图1是一个人物的全身照。生成同一个人物的{view}。保持图1人物的五官、脸型、发型、发色、妆容和服装领口完全不变，"
           "只改变相机距离。纯白色背景，柔和均匀的影棚光，高清写实照片，皮肤纹理清晰。\n\n画面中只有一个人物。输出一张完整图片。")


# one-shot character sheet: a 3x2 grid in a single generation (default mode)
GRID_PROMPT = ("Create a character reference sheet of the person in the reference image. "
               "3x2 grid on a plain light grey background: "
               "1. close-up portrait of the face (neutral expression, looking at camera), "
               "2. full-body front view, "
               "3. full-body back view, "
               "4. full-body side view, "
               "5. three-quarter front view, "
               "6. three-quarter back view. "
               "Keep the exact same face, hair, body proportions, clothing, and colors in every panel. "
               "Clean studio lighting, no background elements.")
GREY = (217, 217, 217)


# face-only close-up added to every sheet (identity detail for close-up shots)
FACE_ONLY_PROMPT = ("图1是一个人物头部的局部截图。把它生成为同一个人的高清面部特写照片：脸部占满画面，"
                    "人物正面直视镜头，表情自然平静，清晰呈现五官、皮肤纹理、发际线、刘海和耳朵。"
                    "保持图1人物的五官、脸型、肤色、发型、发色和妆容完全不变，不改变构图。纯浅灰色背景，"
                    "柔和均匀的影棚光，高清写实照片。\n\n画面中只有一个人物的脸。输出一张完整图片。")


def face_crop(src: Path, dst: Path, size: int = 1024) -> bool:
    """Square crop around the face (head and hair) of an image, scaled to size x size."""
    import logging
    logging.getLogger("insightface").setLevel(logging.ERROR)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=str(ROOT / "ComfyUI" / "models" / "insightface"),
                       allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    im = cv2.imread(str(src))
    faces = app.get(im)
    if not faces:
        return False
    x0, y0, x1, y1 = max(faces, key=lambda f: f.det_score).bbox
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2 - 0.1 * (y1 - y0), 1.9 * max(x1 - x0, y1 - y0)
    H, W = im.shape[:2]
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    cv2.imwrite(str(dst), cv2.resize(im[b:d, a:c], (size, size), interpolation=cv2.INTER_CUBIC))
    return True


def pad_to_aspect(src: Path, dst: Path, aspect: float = 1.5) -> None:
    """Put the reference on a light grey canvas of the grid's aspect (the latent follows the first image)."""
    im = cv2.imread(str(src))
    h, w = im.shape[:2]
    W, H = (int(h * aspect), h) if w / h < aspect else (w, int(w / aspect))
    canvas = np.full((H, W, 3), GREY, np.uint8)
    y, x = (H - h) // 2, (W - w) // 2
    canvas[y:y + h, x:x + w] = im
    cv2.imwrite(str(dst), canvas)


def _runs(profile: np.ndarray, min_len: int) -> list:
    """[start, end) runs where profile is True, at least min_len long."""
    out, i, n = [], 0, len(profile)
    while i < n:
        if profile[i]:
            j = i
            while j < n and profile[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _segments(prof: np.ndarray, n: int) -> list:
    """Content segments along one axis: thin full-length border lines removed, smoothed coverage > 3%, gaps
    narrower than 0.4% merged, segments narrower than 3% (specks, panel numbers) dropped."""
    prof = prof.copy()
    for a, b in _runs(prof > 0.97, 1):              # thin, (almost) full-length lines = panel borders
        if b - a <= 6:
            prof[a:b] = 0
    sm = np.convolve(prof, np.ones(5) / 5, mode="same")
    runs = _runs(sm > 0.03, 1)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.004 * n:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= 0.03 * n]


def split_grid(grid: Path, out: Path, pad: int = 16) -> list:
    """Cut the sheet into its panels, whatever layout the model chose (3x2, 1x4, ...): figures are separated by
    plain background, panel border lines are too thin to count. Writes view_panel<k>.png (row-major)."""
    im = cv2.imread(str(grid))
    H, W = im.shape[:2]
    bg = np.median(np.concatenate([im[:8].reshape(-1, 3), im[-8:].reshape(-1, 3)]), axis=0)
    fg = np.abs(im.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 28
    outs, k = [], 0
    for x0, x1 in _segments(fg.mean(axis=0), W):
        for y0, y1 in _segments(fg[:, x0:x1].mean(axis=1), H):
            k += 1
            dst = out / f"view_panel{k}.png"
            cv2.imwrite(str(dst), im[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)])
            outs.append(dst)
    return outs


def build(images: list, prompt: str, seed: int, prefix: str, resolution: int = 1536) -> dict:
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
                           "resolution": resolution}},
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


def write_preview(views: list, dst: Path, height: int = 1024) -> None:
    """Labelled side-by-side preview of the views (JPEG, for looking at; not used by the pipeline)."""
    tiles = []
    for p in views:
        im = cv2.imread(str(p))
        im = cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height), interpolation=cv2.INTER_AREA)
        name = Path(p).stem.replace("view_", "")
        cv2.putText(im, name, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(im, name, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    cv2.imwrite(str(dst), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", type=Path, required=True, help="full-body photo of the character (cut-out on white)")
    ap.add_argument("--outfit", type=Path, default=None,
                    help="image of the outfit to wear (e.g. a video crop without the head); omit to keep her own clothes")
    ap.add_argument("--face", type=Path, required=True, help="face close-up of the character")
    ap.add_argument("--outfit-text", default="与图2完全相同的服装和配饰",
                    help="short description of the outfit image, Chinese works best, e.g. the tiktok kimono: "
                         "红色和服，红白相间的波浪花纹，单肩露出，宽大的白色衣领，花纹腰带，垂下的黄色系带")
    ap.add_argument("--views", default="front,left,back,face")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", default="grid", choices=["grid", "views"],
                    help="grid: one 3x2 sheet in a single generation, split into panels (default); "
                         "views: one generation per camera angle (--views)")
    ap.add_argument("--resolution", type=int, default=1536,
                    help="views mode: Qwen render size per view (~resolution^2 pixels; 1536 -> ~1248x1920)")
    ap.add_argument("--grid-resolution", type=int, default=2048,
                    help="grid mode: render size of the whole sheet (2048 -> ~2496x1664, panels ~830 px)")
    ap.add_argument("--reuse-grid", action="store_true",
                    help="with --reuse-front: keep an existing sheet_grid.png (re-split it, redo only the face close-up)")
    ap.add_argument("--no-face", action="store_true", help="grid mode: skip the face-only close-up (view_face.png)")
    ap.add_argument("--front-resolution", type=int, default=1024,
                    help="stage 1 (dressing) render size; higher values tend to draw the person twice")
    ap.add_argument("--reuse-front", action="store_true", help="keep an existing <out>/view_front.png (stage 1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--comfy-urls", default="http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inp = ROOT / "ComfyUI" / "input"
    names = {}
    for key, p in (("p", args.person), ("o", args.outfit), ("f", args.face)):
        if p is None:
            continue
        names[key] = f"sheet_{os.getpid()}_{key}{p.suffix}"
        shutil.copy2(p, inp / names[key])
    urls = args.comfy_urls.split(",")
    views = [v for v in args.views.split(",") if v != "front"]

    def gen(k, images, prompt, view, res=None):
        c = cs.Comfy(urls[k % len(urls)])
        h = c.wait(c.submit(build(images, prompt, args.seed, f"sheet/{view}", res or args.resolution)), poll=2)
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
    front = args.out / "front_reference.png"                # grid mode keeps stage 1 here
    if not front.exists():
        front = args.out / "view_front.png"
    if not (args.reuse_front and front.exists()):
        if args.outfit:
            front = gen(0, [names["p"], names["o"], names["f"]],
                        DRESS.format(view=VIEWS["front"], outfit=args.outfit_text), "front", args.front_resolution)
        else:
            front = gen(0, [names["p"], names["f"]], NEUTRAL.format(view=VIEWS["front"]), "front",
                        args.front_resolution)
    names["front"] = f"sheet_{os.getpid()}_front.png"
    shutil.copy2(front, inp / names["front"])
    if args.mode == "grid":
        # stage 2 (grid): the whole sheet in one generation from the front view, then split into panels
        ref = args.out / "_ref_front_padded.png"
        pad_to_aspect(front, ref)
        names["grid_ref"] = f"sheet_{os.getpid()}_gridref.png"
        shutil.copy2(ref, inp / names["grid_ref"])
        ref.unlink()
        front_ref = args.out / "front_reference.png"          # keep the stage-1 image, the panels replace view_front
        if front != front_ref:
            shutil.copy2(front, front_ref)
        existing = args.out / "sheet_grid.png"
        if args.reuse_front and args.reuse_grid and existing.exists():
            grid = existing                           # keep the sheet, only (re)make the face close-up
        else:
            grid = gen(0, [names["grid_ref"]], GRID_PROMPT, "sheet", args.grid_resolution)
        (args.out / "sheet.png").unlink(missing_ok=True)
        if Path(grid) != existing:
            grid = Path(shutil.move(str(grid), existing))
        outs = split_grid(grid, args.out)
        prev = cv2.imread(str(grid))
        if not args.no_face:                          # face-only close-up from the same front reference
            fc = args.out / "_face_crop.png"
            # identity from the real face photo (--face); the generated front view drifts toward a generic face
            if face_crop(args.face, fc) or face_crop(front, fc):
                names["face_crop"] = f"sheet_{os.getpid()}_facecrop.png"
                shutil.copy2(fc, inp / names["face_crop"])
                fc.unlink()
                face = gen(1, [names["face_crop"]], FACE_ONLY_PROMPT, "face", 1024)
                outs.append(face)
            else:
                face = None
        if not args.no_face and face is not None:
            fim = cv2.imread(str(face))
            fim = cv2.resize(fim, (int(fim.shape[1] * prev.shape[0] / fim.shape[0]), prev.shape[0]))
            prev = np.hstack([prev, fim])
        cv2.imwrite(str(args.out / "sheet.jpg"), cv2.resize(prev, (int(prev.shape[1] * 1024 / prev.shape[0]), 1024)),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        for v in names.values():
            (inp / v).unlink(missing_ok=True)
        print(f"sheet: {grid} -> panels {', '.join(p.name for p in outs)}")
        return
    # stage 2: rotate the dressed front view
    with ThreadPoolExecutor(len(urls)) as ex:
        rest = list(ex.map(lambda kv: gen(kv[0], [names["front"]],
                                                (CLOSEUP if kv[1] == "face" else ROTATE).format(view=VIEWS[kv[1]]), kv[1]),
                           enumerate(views)))
    outs = [front] + rest
    write_preview(outs, args.out / "sheet.jpg")
    for v in names.values():
        (inp / v).unlink(missing_ok=True)
    print(f"sheet preview: {args.out / 'sheet.jpg'}  (the pipeline uses the view_*.png files)")


if __name__ == "__main__":
    main()
