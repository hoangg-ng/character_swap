# Character swap: MiniMax H3 + BFS + Realism

This pipeline replaces **one chosen person** in a video with a new character. Everyone else and the
background stay as in the source. Each video is a **case** folder with its own inputs, settings and outputs:

```
cases/<name>/input/     video, character image(s), face crop of the person to replace
cases/<name>/case.env   settings (paths relative to the case folder)
cases/<name>/output/    <JOB>/shotNN/...  (per-shot generations)   <OUT>/swapped.mp4  (result)
```
Run a case with `CASE=cases/<name> bash scripts/run_swap.sh`. Per-case results and findings are in `log.md`.

## Pipeline

| # | Step | Component | What it does |
|---|---|---|---|
| 0 | **Target check** | `bfs_full.py` (insightface `buffalo_l`, YOLO11) | Decides which shots to generate. `PRESENCE=face`: a shot is generated only if the **person to replace** is visible (ArcFace match against `TARGET_FACE`, cosine ≥ 0.35 in ≥ 10% of frames), because BFS always swaps the most visible person and would otherwise turn someone else into the character. `PRESENCE=head` (single-person videos): shots are split into segments with and without the person; segments showing only body parts or a silhouette also stay source. Saved as `output/<job>/target_check.json`. |
| 1 | **Shot split** | `bfs_full.py` | Cuts the clip (`CUTS`, or `auto` with PySceneDetect), converts to 24 fps and resizes to the generation size: short side 480 (736 with `RES=720`), long side from the source aspect, 32-aligned. Adds 9 sacrificial lead frames (H3 leaks the reference pose into the first latent group) and pads to H3's 17n+5 frame grid. Shots run in parallel, one ComfyUI server per GPU. |
| 2 | **Colour-match the references** | ComfyUI `ColorTransfer` | Recolours each character reference toward the shot's frames (`reinhard_lab`, stats pooled over the shot, strength 1.0), so the character enters the model with the scene's tint and lighting instead of a studio look. |
| 3 | **Lock the others** (`PROTECT`) | SAM 3.1 (`SAM3_Detect` → `SAM3_VideoTrack`) | Tracks `PROTECT` (threshold 0.7), grows the mask 4 px, inverts it and applies it as the latent edit mask (`H3RelayApplyEditMask`): those people cannot change. The target has **no mask**, so her body, hair and clothes are generated freely and blend with the scene. |
| 4 | **Conditioning** | MiniMax H3 `MiniMaxH3ReferenceToVideo` + `MiniMaxH3AddGuide` | `<Picture 1>`…`<Picture N>` are the (recoloured) character references, e.g. the views of a character sheet. The prompt is `r34l1sm, ` (Realism trigger) + the BFS `body_swap:` template with the character's head-to-toe description. The original shot is the guide video. |
| 5 | **Model stack** | MiniMax H3 + 3 LoRAs | H3 ref2va (int8), shift 12/3, pytorch attention, then DMD ref2va 8-step turbo, **BFS v1.1** (swaps the subject, follows the guide's motion, camera and lighting) and the **Realism People LoRA** 0.8 (skin, hair strands, hands, expressions). |
| 6 | **Sampling** | `SamplerCustomAdvanced` | euler, simple scheduler, 8 steps, CFG 1. The whole frame is generated, guided by the original shot; locked people keep their latent. |
| 7 | **Assembly** | `assemble_bfs.py --raw --letterbox --tone-person` | Drops the lead frames, trims to the shot's length and scales back to the source size. Uses the **pure generated frames** (pasting locked people back from the source only adds a visible edge). Re-applies the source's letterbox bars, and in shots where the generated person is much brighter than the original (dark, low-key shots) matches her tone to the original person. |
| 8 | **Final cut** | `scripts/run_swap.sh` | Generated and source shots are concatenated with the source audio, plus a `side_by_side.mp4`. Optional `SPLICE="N=clip.mp4"` replaces a shot with a prepared clip. |

### Character sheet (recommended reference)
`utils/make_char_sheet.py` renders the character from several angles with **Qwen-Image-2.1** + the
lightx2v Lightning LoRA (4 steps), and the views go to H3 as references (`CHAR_REF` + `CHAR_REF_EXTRA`).
A single photo pulls every shot toward its own pose and framing; a sheet gives BFS the identity and
outfit from front, side and back.

1. **Stage 1, front view:** the person photo (`--person`, a cut-out on white) and a face close-up (`--face`)
   become one clean front view with the arms down. With `--outfit` (a crop of the outfit, head cut off) she
   is dressed in that outfit, e.g. the clothes of the video.
2. **Stage 2, rotation:** only the stage-1 image is rotated to side and back views (a second input image
   makes the model draw a second person). Some angles do not rotate for some seeds; check the views.

```bash
.venv/bin/python utils/make_char_sheet.py --person cases/<name>/input/person_cutout.png \
    --face cases/<name>/input/face.jpg [--outfit cases/<name>/input/outfit_nohead.jpg] \
    --views front,left,back --out cases/<name>/input/sheet        # -> view_*.png + sheet.png
```

## Settings
`scripts/run_swap.sh` reads `cases/<name>/case.env`. Any setting can be overridden on the command line
(`CASE=cases/<name> SEED=7 bash scripts/run_swap.sh`).

| setting | meaning |
|---|---|
| `VIDEO` | the source video (any size and fps) |
| `CUTS` | shot boundaries in source frames, or `auto` (PySceneDetect) |
| `CHAR_REF` / `CHAR_REF_EXTRA` | the character reference, and extra views of the same character (character sheet), space-separated; each becomes another `<Picture N>` |
| `CHAR_DETAILS` | the character's head-to-toe description, starting with gender |
| `TARGET_FACE` | a face crop of the person to **replace**; shots are generated only where this identity is visible. Without it, `TARGET_SEX=F/M` counts any face of that gender |
| `TARGET_TEXT` | how the prompt names the person to replace |
| `PROTECT` / `PROTECT_TEXT` | SAM 3.1 text of the people to **keep** (latent lock), and how the prompt names them; empty = nobody |
| `PRESENCE` | `face` (default, several people): generate where the target's face is seen. `person` (one person): YOLO11 detects her body in every frame (backs, silhouettes, partial bodies) and splits shots into segments with and without her (gaps < 0.5 s and blips < 2 frames smoothed). `head`: like `person`, but a segment also stays source if her head (YOLO11-pose nose/eye/ear keypoints, also from behind) is seen in < 30% of it (only a hem or a sleeve: BFS would invent a whole person) or if she is a silhouette (median brightness in her body box < 6/255) |
| `KEEP_OUTFIT=1` | identity, face, hair and body come from the references; clothing stays as worn in the video |
| `FACE_ONLY=1` | swap only the face: SAM tracks "face" as the edit mask, the prompt keeps hair/clothing/pose, and assembly composites the face region onto the source |
| `TONE_PERSON` (0.8) | assembly post-process (`utils/tone_match.py`): where the generated person is > 1.25x brighter than the original, her LAB tone is matched to the original person inside her box, feathered. Normally lit shots are left alone so the new identity keeps its colours |
| `LIGHT_MATCH=1` | adds a lighting-match sentence to the prompt (off by default; did not help in tests) |
| `SOLO_SHOTS` | shots where the target is the only person (the prompt drops the "keep the others" note); automatic for all shots when `PROTECT` is empty |
| `REALISM` (0.8), `REF_COLOR` (`reinhard_lab:1.0`), `RES` (480), `SEED` (42) | quality settings, as in the pipeline table |
| `JOB` / `OUT` | folder names under `output/` (default `gen_seed<SEED>`, `final_seed<SEED>`) |
| `ONLY="2,4"` | generate only these shots (the others reuse existing results in `JOB`) |
| `SPLICE="N=clip.mp4"` | replace shot N with a prepared clip |
| `CHECK_ONLY=1` | print the per-shot decision and stop (no GPU needed) |
| `ASSEMBLE_ONLY=1` | re-assemble `JOB` without generating |

A face crop for `TARGET_FACE` can be cut from any frame where the person faces the camera.

## 1. Setup (once)
Accept the MiniMax H3 licence on Hugging Face first, then:
```bash
HF_TOKEN=hf_xxx bash scripts/setup.sh              # ComfyUI, h3-relay nodes, Python deps, all model weights
HF_TOKEN=hf_xxx bash scripts/download_models.sh    # weights only (skips files already present)
```
Models (downloaded by `scripts/download_models.sh`):
- MiniMax H3: `minimax_h3_ref2va_pruned_int8_convrot`, `qwen3vl_32b_minimax_h3_nvfp4_awq`, video/audio VAEs
- SAM 3.1: `sam3.1_multiplex_fp16`
- Qwen-Image-2.1 (character sheet): `qwen_image_2.1_int8_convrot`, `qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16`
- LoRAs: BFS v1.1 `minimax_h3_character_swap_v1.1_ref2va_fro`, DMD turbo `minimax_h3_dmd_ref2va_8step_turbo_pruned`,
  Realism `h3-realism-people-t2v-i2v-r2v`, `qwen_image_2.1_lightning_4steps_lightx2v`
- YOLO11 `yolo11m.pt`, `yolo11m-pose.pt`; insightface `buffalo_l` is fetched on first use

**Where the models are.** Base models live on the checkpoint storage
**`/common/users/hn315/checkpoints/char_swap_models/`** (not counted against the `/research/cbim/vast`
quota; override with `MODELS_BASE=...`). Only the LoRAs stay in the project (`models/loras/`).
`ComfyUI/extra_model_paths.yaml` registers both.

**Two pitfalls.** Keep only `onnxruntime-gpu` installed: the CPU `onnxruntime` package hides it and
insightface falls back to the CPU. The driver is CUDA 12.8, so every graph uses `pytorch attention`.

## 2. Start ComfyUI (one server per GPU)
```bash
GPU=1 PORT=8188 bash scripts/start_comfy.sh &   # nvidia-smi numbering, CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU=2 PORT=8189 bash scripts/start_comfy.sh &
GPU=3 PORT=8190 bash scripts/start_comfy.sh &
```

## 3. Run
```bash
CASE=cases/<name> CHECK_ONLY=1 bash scripts/run_swap.sh        # which shots will be generated
CASE=cases/<name> bash scripts/run_swap.sh                     # -> output/<JOB>/, output/<OUT>/swapped.mp4
CASE=cases/<name> SEED=7 bash scripts/run_swap.sh              # another seed
CASE=cases/<name> RES=720 JOB=gen720 OUT=final720 bash scripts/run_swap.sh
CASE=cases/<name> ASSEMBLE_ONLY=1 bash scripts/run_swap.sh     # re-assemble without generating
```
Each run writes to `cases/<name>/output/`:
- `<OUT>/swapped.mp4` (with the source audio), `<OUT>/side_by_side.mp4` (source | result)
- `<OUT>/s*.mp4` per shot, `<OUT>/target_check.json` (the per-shot decision)
- `<JOB>/shotNN/`: the generation (workflow, prompt, recoloured references, lock masks)

## 4. Files
```
char_swap/
├── run_bfs.py, bfs_full.py, assemble_bfs.py   main pipeline (one shot / all shots / assembly)
├── scripts/   run_swap.sh (entry point), setup.sh, download_models.sh, start_comfy.sh
├── utils/     comfy_client.py, tone_match.py, make_char_sheet.py, make_fullbody_ref.py, face_refine.py
├── cases/     <name>/input, case.env, output
└── log.md     per-case results and findings
```

| file | role |
|---|---|
| `scripts/run_swap.sh` | **entry point** (`CASE=cases/<name>`): target check → BFS + Realism + ColorTransfer with the SAM lock → assembly |
| `bfs_full.py` | target check (`--presence`, `--target-face`, `--check-only`), shot split (`--cuts auto`), padding, dispatches `run_bfs.py` over the GPUs; writes to `$CHAR_SWAP_OUT/<job>` |
| `run_bfs.py` | builds and submits the BFS v1.1 graph for one shot (`--protect --protect-only`, `--realism`, `--ref-color`, `--ref-extra`, `--keep-outfit`, `--face-only`) |
| `assemble_bfs.py` | per-shot assembly: `--raw`, `--letterbox`, `--tone-person`, optional mask composite / colour match; source size and fps, concat, source audio |
| `utils/make_char_sheet.py` | character sheet with Qwen-Image-2.1 + Lightning (see above) |
| `utils/tone_match.py` | per-frame person tone matching (YOLO11 boxes, LAB Reinhard), used by `assemble_bfs.py --tone-person` |
| `utils/comfy_client.py` | minimal ComfyUI HTTP client (submit, wait, download) |
| `utils/make_fullbody_ref.py` | full-body reference generated by H3 from a face photo (weaker than a real photo or a sheet) |
| `utils/face_refine.py` | optional FaceRefine post-pass (needs `Carasibana/ComfyUI-H3-FaceRefine`; not in the main pipeline) |

## Licences
MiniMax H3, the BFS LoRA and the Realism People LoRA follow the MiniMax H3 Community Licence. Read
its territorial terms before you use them. The BFS card also asks for appropriate rights and consent
for the reference images and guide videos. Qwen-Image-2.1 follows the Qwen research licence.
