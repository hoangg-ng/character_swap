# Character swap: MiniMax H3 + BFS + Realism

This pipeline replaces **one chosen person** in a video with a new character. Everyone else and the
background stay as in the source. Each video is a **case** folder with its own inputs, settings and outputs:

```
cases/<name>/input/     video, character image(s), face crop of the person to replace
cases/<name>/case.env   settings (paths relative to the case folder)
cases/<name>/output/    gen_seed<N>/shotNN/...  (per-shot generations)   final_seed<N>/swapped.mp4  (result)
```
Run a case with `CASE=cases/<name> bash run_swap.sh`.

| case | video | status |
|---|---|---|
| `cases/hug_scene` | K-drama hug scene, 1280x720, 6 shots, actress + man | **done**: final videos `output/swapped_seed42.mp4`, `output/swapped_seed7.mp4` (section 6) |
| `cases/tiktok_trend` | 948x576, 30 fps, 25.6 s, one woman in a red kimono | new character `face.jpg` / `full_body.jpg`, 3 variants compared (section 7) |

The step table below uses `hug_scene` as the example.

### Step by step

| # | Step | Component | What it does |
|---|---|---|---|
| 0 | **Target check** | `bfs_full.py` (insightface `buffalo_l`) | For each shot, counts the source frames in which the **person to replace** is visible: an identity match against `TARGET_FACE` (ArcFace cosine ≥ 0.35), or any face of `TARGET_SEX` as a fallback. If they are seen in less than `MIN_TARGET` (10%) of the frames, the shot is **not generated** and stays exactly as the source: BFS always swaps the most visible person, so it would turn someone else into the character. On this clip, shots 1 (her back) and 3 (only the man) score 0% and are kept; shots 2, 4, 5 and 6 score 17–100% and are generated. The result is saved as `output/<job>/target_check.json`. |
| 1 | **Shot split** | `bfs_full.py` | Cuts the clip at its hard cuts (`CUTS`, or `auto` with PySceneDetect), converts from the source fps to 24 fps and resizes to the generation size: short side 480 (736 with `RES=720`), long side from the source aspect, 32-aligned (hug_scene 864x480, tiktok_trend 800x480). It adds 9 sacrificial lead frames, because H3 leaks the reference pose into the first latent group. Then it pads to H3's 17n+5 frame grid, with audio padded by `apad`. The shots run in parallel on 3 A6000s. |
| 2 | **Colour-match the reference** | ComfyUI `ColorTransfer` | Recolours `CHAR_REF` toward this shot's frames (`reinhard_lab`, stats pooled over the shot, strength **1.0**). She enters the model with the scene's tint and lighting instead of a studio look. The result is saved as `shotNN/refcolor_*.png`. |
| 3 | **Lock the others** (`PROTECT`) | SAM 3.1 (`SAM3_Detect` → `SAM3_VideoTrack`) | Tracks `PROTECT` (here "man") (threshold 0.7) in every frame, grows the mask 4 px and inverts it, then applies it as the latent edit mask (`H3RelayApplyEditMask`). He cannot change. **She has no mask**, so her body, hair and clothes are generated freely and blend with the scene. |
| 4 | **Conditioning** | MiniMax H3 `MiniMaxH3ReferenceToVideo` + `MiniMaxH3AddGuide` | `<Picture 1>` is the recoloured character. The prompt is `r34l1sm, ` (Realism trigger word) followed by the BFS `body_swap:` template with her head-to-toe description. The original shot is the guide video, at frame 0. |
| 5 | **Model stack** | MiniMax H3 + 3 LoRAs | H3 ref2va (int8) with shift 12/3 and pytorch attention, then: DMD ref2va 8-step turbo (speed), **BFS v1.1** (Alissonerdx; swaps the subject and follows the guide's motion, camera and lighting), **Realism People LoRA** 0.8 (fal; natural skin, hair strands, hands and expressions). |
| 6 | **Sampling** | `SamplerCustomAdvanced` | euler, simple scheduler, 8 steps, CFG 1. The whole frame is generated from noise, guided by the original shot; the man's latent is kept. |
| 7 | **Assembly** | `assemble_bfs.py --raw --letterbox` | Drops the lead frames, trims to the shot's real length and scales back to the source size. It uses the **pure generated frames**: the man is *not* pasted back from the source, because the generation-time lock (step 3) already keeps him, and pasting only added a visible box or halo where the generated man sits a few pixels off the source man. It then re-applies the source's black letterbox bars (detected per shot; hug_scene rows <52 and >667) to remove the dark line BFS draws at the bar edge. |
| 8 | **Final cut** | `run_swap.sh` | Generated shots and source shots (from the target check) are concatenated with the source audio, plus a `side_by_side.mp4`. Optional `SPLICE="N=clip.mp4"` replaces shot N with a prepared clip (hug_scene: `input/shot1_backview_v5.mp4`, her back swapped by the older akatz pipeline). |

### Why each piece is there
- **BFS v1.1 (not the akatz LoRA):** it generates the whole frame from the guide, so she gets new
  hair motion and lighting instead of a pasted-in look.
- **Man lock instead of a mask on her:** with two people in shot, BFS sometimes swaps the man.
  A mask on her fixed that, but left a visible colour zone. Locking him fixes the target with
  nothing visible on her.
- **No pixel paste of the man:** the SAM man mask is needed only *inside generation*, as the latent edit
  mask; that is what stops BFS from swapping him. Copying his pixels back from the source afterwards
  left a visible box edge, so it was dropped. Fresh runs with seeds 42 and 7 have no box around the man.
- **Realism People LoRA:** the most visible quality gain in a 4-way test (baseline / FaceRefine / Realism / both):
  hair strands, skin and hands.
- **ColorTransfer on the reference:** the reference photo is studio-lit on white, and the scene is
  night, warm, with green bokeh. Pre-tinting the reference narrows the colour gap the model has to bridge.
  The final config is `reinhard_lab` at **1.0**. On the still reference, 1.0 looks dark and muddy, but
  BFS re-lights her from the scene, and in the video 1.0 gives warmer, less pink-white skin that sits
  better in the night light than 0.6. The trade-off: the hair drifts slightly from wine-red toward
  copper. `mkl_lab` was rejected: it turns the red
  hair magenta or green.
- **FaceRefine (`Carasibana/ComfyUI-H3-FaceRefine`) is not in the main pipeline.** `face_refine.py` is
  kept as an optional post-pass, but on this clip:
  - when only one face is in frame its identity check never runs, and in shot 1 it re-rendered the man's face;
  - her face is already large (147 to 580 px), so the gain is small.
  The script now skips shot 1, and any shot where her face never matches, such as shot 5.

### Settings: define who to replace
`run_swap.sh` reads `cases/<name>/case.env`. Any setting can be overridden on the command line
(`CASE=cases/hug_scene SEED=7 bash run_swap.sh`).

| setting | meaning |
|---|---|
| `VIDEO` | the source video (any size and fps) |
| `CUTS` | shot boundaries in source frames, or `auto` (PySceneDetect) |
| `CHAR_REF` / `CHAR_DETAILS` | the new character (cut-out on white) and its head-to-toe description, starting with gender |
| `TARGET_FACE` | a face crop of the person to **replace**. For each shot, insightface (ArcFace) checks whether this identity is visible (cosine ≥ 0.35 in ≥ `MIN_TARGET` = 10% of frames). Only those shots are generated; every other shot stays exactly as the source. Without it, `TARGET_SEX=F/M` counts any face of that gender. |
| `TARGET_TEXT` | how the BFS prompt names the person to replace |
| `PROTECT` / `PROTECT_TEXT` | SAM 3.1 text of the people to **keep** (locked in the latent during generation), and how the prompt names them; empty = nobody |
| `PRESENCE` | `face` (default): a shot is generated only where the target's face is seen; use it when several people appear. `person`: for single-person videos. YOLO11 detects her body in every frame (backs, silhouettes, partial bodies), and shots are **split into segments with and without her**: gaps under 0.5 s and blips under 2 frames are smoothed out. Segments with her are generated; empty segments (scenery, objects) stay source. `head`: like `person`, plus two rules that keep a segment as source: (1) her head (YOLO11-pose nose/eye/ear keypoints, which also work from behind) is seen in under `--min-head` = 30% of it, i.e. only body parts such as a hem or a sleeve, where BFS would invent a whole person; (2) she is a silhouette (median brightness inside her body box < 6/255), where no identity is visible and BFS would light her up. |
| `TONE_PERSON` (0.8) | post-process in assembly (`tone_match.py`): in shots where the generated person is more than 1.25x brighter than the original (dark, low-key shots), her LAB tone is matched to the original person inside her box, feathered. Normally lit shots are left alone so hair/skin colours keep the new identity |
| `LIGHT_MATCH=1` | adds a lighting-match sentence to the prompt; tested on the dark spin and it did not help (brighter, and she turned to the camera), so it is off by default |
| `KEEP_OUTFIT=1` | face-only character reference: the prompt swaps face, head and hair from `<Picture 1>` and keeps the clothing the person wears in the video |
| `SOLO_SHOTS` | shots where the target is the only person (the prompt drops the "keep the others" note); automatic for all shots when `PROTECT` is empty |
| `REALISM` (0.8), `REF_COLOR` (`reinhard_lab:1.0`), `RES` (480), `SEED` (42) | quality settings, as in the step table |
| `JOB` / `OUT` | folder names under `output/` (default `gen_seed<SEED>`, `final_seed<SEED>`) |
| `ONLY="2,4"` | generate only these shots (the others reuse existing results in `JOB`) |
| `SPLICE="N=clip.mp4"` | replace shot N with a prepared clip |
| `CHECK_ONLY=1` | print the per-shot target-check decision and stop (no GPU needed) |
| `ASSEMBLE_ONLY=1` | re-assemble `JOB` without generating |

A face crop for `TARGET_FACE` can be cut from any frame where the person faces the camera; insightface
(`buffalo_l`) finds the face. On hug_scene the check works both ways: with the actress's face it
generates shots 2, 4, 5 and 6; with the man's face it picks exactly shots 1 and 3.

## 1. Setup (once)
Accept the MiniMax H3 licence on Hugging Face first, then:
```bash
HF_TOKEN=hf_xxx bash download_models.sh && HF_TOKEN=hf_xxx bash setup.sh
# extras for this flow
.venv/bin/hf download Alissonerdx/Minimax-H3-ComfyUI minimax_h3_character_swap_v1.1_ref2va_fro.safetensors --local-dir ComfyUI/models/loras   # BFS v1.1
.venv/bin/hf download fal/MiniMax-H3-Realism-People-LoRA h3-realism-people-t2v-i2v-r2v.safetensors --local-dir ComfyUI/models/loras
.venv/bin/hf download Bingsu/adetailer face_yolov8m.pt --local-dir ComfyUI/models/ultralytics/bbox
git clone https://github.com/Carasibana/ComfyUI-H3-FaceRefine ComfyUI/custom_nodes/ComfyUI-H3-FaceRefine
uv pip install --python .venv/bin/python ultralytics insightface "scenedetect>=0.7"
uv pip uninstall --python .venv/bin/python onnxruntime onnxruntime-gpu && uv pip install --python .venv/bin/python onnxruntime-gpu
```
Required models: `minimax_h3_ref2va_pruned_int8_convrot`, `qwen3vl_32b_minimax_h3_nvfp4_awq`, H3 video/audio VAEs,
`sam3.1_multiplex_fp16`, `minimax_h3_dmd_ref2va_8step_turbo_pruned`, `minimax_h3_character_swap_v1.1_ref2va_fro`,
`h3-realism-people-t2v-i2v-r2v`, `face_yolov8m.pt`, `yolo11m.pt` (person presence, `ComfyUI/models/ultralytics/person/`).
Character sheet: Qwen-Image-2.1 from `Comfy-Org/Qwen-Image-2.1` (`qwen_image_2.1_int8_convrot`,
`qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16`) and the 4-step LoRA `lightx2v/Qwen-Image-2.1-Lightning`
(saved as `qwen_image_2.1_lightning_4steps_lightx2v.safetensors`).

**Where the models are.** Base models (diffusion models, text encoders, VAEs, SAM, ControlNet patches,
depth, MediaPipe) live on the checkpoint storage **`/common/users/hn315/checkpoints/char_swap_models/`**, which
is not counted against the `/research/cbim/vast` quota. Only the LoRAs stay in the project (`models/loras/`,
~3.5 GB). `ComfyUI/extra_model_paths.yaml` registers both; `setup.sh` and `download_models.sh` download base
models there (override with `MODELS_BASE=...`).

**Two pitfalls.** Keep only `onnxruntime-gpu` installed: if the CPU `onnxruntime` package is also
there, it hides the GPU build and insightface runs on the CPU. The driver is CUDA 12.8, so every
graph uses `pytorch attention`.

## 2. Start ComfyUI (one server per A6000)
```bash
GPU=1 PORT=8188 bash start_comfy.sh &   # nvidia-smi numbering, CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU=2 PORT=8189 bash start_comfy.sh &
GPU=3 PORT=8190 bash start_comfy.sh &
```

## 3. Run
```bash
CASE=cases/hug_scene CHECK_ONLY=1 bash run_swap.sh       # which shots will be generated
CASE=cases/hug_scene bash run_swap.sh                    # -> output/gen_seed42, output/final_seed42/swapped.mp4
CASE=cases/hug_scene SEED=7 bash run_swap.sh             # another seed
CASE=cases/hug_scene RES=720 JOB=gen720 OUT=final720 bash run_swap.sh
# variants: REALISM=0 (no realism LoRA), REF_COLOR="" (original reference),
#           REF_COLOR=reinhard_lab:0.6 (other method/strength: reinhard_lab | mkl_lab | histogram)
CASE=cases/hug_scene ASSEMBLE_ONLY=1 bash run_swap.sh   # re-assemble output/gen_seed42 without generating
# optional FaceRefine post-pass on an assembled clip (not in the main pipeline)
.venv/bin/python face_refine.py --clip cases/hug_scene/output/swapped_seed42.mp4 \
    --source cases/hug_scene/input/hug_scene.mp4 --face-ref cases/hug_scene/input/phuc_face.jpg \
    --out cases/hug_scene/output/seed42_fr [--realism 0.8] [--denoise 0.4]
```
Each run writes to `cases/<name>/output/`:
- `<OUT>/swapped.mp4` (with the source audio), `<OUT>/side_by_side.mp4` (source | result)
- `<OUT>/s*.mp4` per shot, `<OUT>/target_check.json` (the per-shot decision)
- `<JOB>/shotNN/`: the generation (workflow, prompt, recoloured reference, lock masks)

## 4. Files
| file | role |
|---|---|
| `run_swap.sh` | **entry point** (`CASE=cases/<name>`): target check → BFS + Realism + ColorTransfer with the SAM lock → raw assembly + letterbox |
| `cases/<name>/` | one folder per video: `input/`, `case.env`, `output/` |
| `bfs_full.py` | target check (`--target-face` / `--target-sex`, `--check-only`), cuts shots (`--cuts auto`), pads (lead frames, 17n+5, audio `apad`), dispatches `run_bfs.py` over the GPUs; writes to `$CHAR_SWAP_OUT/<job>` |
| `run_bfs.py` | builds and submits the BFS v1.1 graph for one shot (`--protect man --protect-only`, `--realism`, `--ref-color`) |
| `assemble_bfs.py` | per-shot assembly: `--raw` (pure generation, used), `--letterbox` (source bars back), optional man restore / colour match; source size and fps, concat, source audio |
| `make_char_sheet.py` | character sheet with Qwen-Image-2.1 + Lightning (4 steps): stage 1 dresses the person (`--person`) in the outfit (`--outfit`, a video crop without the head) as one front view; stage 2 rotates that view (front-left, side, back, ...). Writes `view_*.png` + `sheet.png`; use the views as `CHAR_REF` + `CHAR_REF_EXTRA` |
| `tone_match.py` | per-frame person tone matching (YOLO11 person boxes, LAB Reinhard), used by `assemble_bfs.py --tone-person` |
| `make_fullbody_ref.py` | full-body reference generated by H3 from a face photo (weaker than a real photo or a sheet) |
| `face_refine.py` | optional FaceRefine post-pass (not in the main pipeline), her face only (`--realism`, `--denoise`, `--keep`) |
| `comfy_client.py` | minimal ComfyUI HTTP client (submit, wait, download) used by the scripts |

## 5. What we learned on hug_scene (why this flow)
- BFS generates the whole frame and looks more natural than the akatz masked edit.
- With two people in shot, BFS tends to swap the more prominent one (it swapped the man in shots 1
  and 5). A SAM mask on **her** forces the right person but leaves a visible colour zone. Locking
  **the man** instead (edit mask = NOT man) fixes the target in shots 2, 4, 5 and 6 without any mask on her.
- Shot 1 is a back view and shot 3 has no actress. Generating them with BFS (seeds 42 and 7) turned the man
  into the character, even with the man lock. That is why
  the target check keeps any shot where her face is not seen.
- Generating at 480p fixed the shot 6 failure, and the faces are close-ups anyway; 720p is optional.
- The raised arm in the reference photo leaks into the first frames of shot 2. An arm-free reference is the next fix.

## 6. Case hug_scene: results
| output (in `cases/hug_scene/`) | what |
|---|---|
| `output/final_sheet_seed42/swapped.mp4` | **best (case default)**: references = character sheet `input/sheet/view_{front,left,back}.png` (Qwen-Image-2.1 from `phuc_cutout.png` + `phuc_face.jpg`, her own outfit, arms down). Shots 2, 4, 5, 6 swapped, 1 and 3 kept by the target check, man locked, no box |
| `output/compare_sheet_vs_previous.mp4` | original / previous best (single photo) / character sheet |
| `output/swapped_seed42.mp4`, `swapped_seed7.mp4` | previous best: single reference photo `phuc_cutout.png` |
| `input/` | `hug_scene.mp4`, `phuc*.jpg/png` (character photos), `target_face.jpg` (actress face crop), `sheet/` (character sheet), `shot1_backview_v5.mp4` (optional shot-1 splice) |

With the single photo, her raised arm from the photo leaked into the first frames of shot 2. The sheet's
views have the arms down, and that pose leak is gone; identity, hair and outfit are unchanged.
For this sheet the front-left view would not rotate with any seed, so it uses front, side and back.

To reproduce: `CASE=cases/hug_scene bash run_swap.sh`.

## 7. Case tiktok_trend
- `input/tiktok_trend.mp4`: 948x576, 30 fps, 25.6 s, one woman in a red kimono, no other people
  (`PROTECT` is empty). Generation size 800x480.
- `input/face.jpg`: the new character (a selfie). `input/face_cutout.png` is her SAM 3.1 cut-out on white,
  used as `CHAR_REF`. Since it shows only the face, `KEEP_OUTFIT=1`: face, head and hair are swapped,
  and the red kimono of the video is kept.
- `input/target_face.jpg`: the original woman's face, cropped from frame 405 (used by `PRESENCE=face`).
- Shots: PySceneDetect cuts with the last cut corrected (728 → 727). `PRESENCE=head` splits shot 15 into
  her 6-frame flash (727–732) and the foggy train, giving 16 segments:

  | generated (her head visible) | kept as source |
  |---|---|
  | 3 (walking away), 4 (walking toward camera), 6, 8 (close-ups), 11 (dance), 13 (spin), 15 (flash) | 1 (silhouette), 2 (kimono hem and feet), 7, 10 (sleeve under the newspaper), 5, 9, 12, 14, 16 (no one) |

  Segment 2 (≈1.2–2.5 s) broke every variant before this rule: a face, a sideways head or a second whole
  woman was invented in a shot of only a hem and feet.
- Results (seed 42, Realism 0.8 + ColorTransfer 1.0, clothing kept from the video), compared in
  `output/compare_4_variants.mp4` (original | 1 | 2 / 3 | 4 | the sheet):

  | variant | reference | settings | output | verdict |
  |---|---|---|---|---|
  | 1. Face only | `face_cutout.png` (from `face.jpg`) | `FACE_ONLY=1` (SAM "face" edit mask, face composited on the source) | `output/final_face_seed42/` | most faithful to the video: only the face changes; light-brown bangs sit on the black hair |
  | 2. Body inferred | `fullbody_ref.png` (H3-generated from `face.jpg`, `make_fullbody_ref.py`) | `KEEP_OUTFIT=1` | `output/final_bodyinfer_seed42/` | weakest: the generated kimono replaces the video's |
  | 3. Full body photo | `full_body_cutout.png` (from `full_body.jpg`) | `KEEP_OUTFIT=1` | `output/final_fullbody_seed42/` | good whole-person swap; close-ups become upright with eyes open |
  | **4. Character sheet** (case default) | `sheet/view_{front,front_left,left,back}.png` (Qwen-Image-2.1 from `full_body.jpg` + `outfit_ref_nohead.jpg` + `face_cutout.png`) | `CHAR_REF` + `CHAR_REF_EXTRA`, `KEEP_OUTFIT=1` | `output/final_sheet_seed42/` | **best whole-person swap**: kimono closest to the video's, back views stay back views, close-up tilt and closed eyes kept |

  First attempt with the face selfie as a whole-body reference: every shot became a frontal portrait.
- Dark segments 3 (walking away) and 13 (spin) were rendered 2-5x brighter than the original in every
  whole-person variant; `TONE_PERSON` brings them back into the shadow.
- Reproduce the best result: `CASE=cases/tiktok_trend bash run_swap.sh` (defaults in `case.env`: character
  sheet, `PRESENCE=head`, `KEEP_OUTFIT=1`, Realism 0.8, ColorTransfer 1.0, `TONE_PERSON` 0.8).

## Licences
MiniMax H3, the BFS LoRA and the Realism People LoRA follow the MiniMax H3 Community Licence. Read
its territorial terms before you use them. The BFS card also asks for appropriate rights and consent
for the reference images and guide videos.
