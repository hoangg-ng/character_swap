# Character swap: MiniMax H3 + BFS + Realism

This pipeline replaces **one chosen person** in a video with a new character. Everyone else and the
background stay as in the source. Each video is a **case** folder with its own inputs, settings and outputs:

```
cases/<name>/input/     video, character image(s), face crop of the person to replace
cases/<name>/case.env   settings (paths relative to the case folder)
cases/<name>/output/    <JOB>/shotNN/...  (per-shot generations)   <OUT>/swapped.mp4  (result)
```
Run a case with `CASE=cases/<name> bash scripts/run_swap.sh`. Per-case results and findings are in `log.md`.

## Standard pipeline (use this for every case)
Everything below is the default of `scripts/run_swap.sh`; a case only needs its inputs in `case.env`.

```
 0. character sheet      SHEET_PERSON + SHEET_FACE -> Qwen-Image-2.1 sheet, in her own clothes or (KEEP_OUTFIT=1)
                         in the video's outfit; panels sorted into main / extra / back / face references
 1. scene cuts           CUTS=auto: sensitive PySceneDetect + colour check + 1-frame flash detection
 2. target check         which shots/segments contain the person to replace (PRESENCE=face|head)
 3. references per shot  recoloured to the scene's light (scene only, people masked out);
                         face refs only where a face is visible, back view first where none is
 4. soft-face guide      SOFT_FACE=1: the original person's face is softened in the guide video (pose and coarse
                         expression kept, identity detail gone), so the model cannot copy the original face
 5. generation 480p      MiniMax H3 + DMD turbo + BFS v1.1 (+ SAM lock on PROTECT people)
 6. upscale to 720p      H3 latent upscaler x1.5 + refine pass with the Realism People LoRA 0.8
 7. identity check       shots whose face looks more like the original person than the character are reported;
                         AUTO_RETRY=N re-renders them with the next seeds and keeps the best attempt (off by default)
 8. assembly             raw generated frames, letterbox, tone match for dark shots (or RELIGHT=0-1: the source's
                         broad light re-applied to the generated shots), crop bars padded back
 9. FaceRefine + guard   face re-render, any frame whose identity drops is reverted; always last
```
```bash
CASE=cases/<name> bash scripts/run_swap.sh                 # standard: 720p, ~45-60 min for a 20 s clip on 3 GPUs
CASE=cases/<name> UPSCALE=0 bash scripts/run_swap.sh       # quick test at 480p (Realism in the generation), ~25 min
CASE=cases/<name> CHECK_ONLY=1 bash scripts/run_swap.sh    # cuts + per-shot decisions only (no GPU)
```

**A new case** (`cases/<name>/case.env`, paths relative to the case folder):
```bash
VIDEO=${VIDEO:-input/video.mp4}
CROP=${CROP-auto}                                     # remove black bars if any
SHEET_PERSON=${SHEET_PERSON:-input/<char>_cutout.png}  # full-body cut-out of the new character (on white)
SHEET_FACE=${SHEET_FACE:-input/<char>_face.png}        # her face photo (large, sharp face works best)
CHAR_DETAILS=${CHAR_DETAILS:-"Female, ..., hair ..., wearing ..."}   # head-to-toe description
KEEP_OUTFIT=${KEEP_OUTFIT:-0}                          # 1 = she wears the video's clothes (+ OUTFIT_TEXT)
TARGET_TEXT=${TARGET_TEXT:-"the woman in the red dress"}
PRESENCE=${PRESENCE:-head}                             # one person in the video; face + TARGET_FACE if several
PROTECT=${PROTECT-}                                    # people to keep unchanged, e.g. "man"
```

**Why each choice (measured on the cases in `log.md`):**
| choice | evidence |
|---|---|
| character sheet over one photo | a single photo pulls every shot toward its pose; the sheet keeps back views as back views |
| sheet in the video's outfit when keeping clothes | with her own clothes in the references the original face stayed (identity 0.19 → 0.27 with the outfit sheet; on tiktok 0.43) |
| face photo as face reference, only where a face is seen | close-up identity 0.16 → 0.56; in back views it caused frontal portraits |
| back view leads in faceless shots | a small figure walking away was drawn as a large frontal portrait (2 seeds); fixed with the back view first |
| auto cuts with verification and flash detection | missed cuts caused an invented person over a hand close-up and the original face in 1-frame flashes |
| scene-only reference colour | the replaced person's red dress turned the new skin red (+31 → +14); identity unchanged |
| Realism in the upscale pass | identity 0.366 vs 0.350 and sharper faces than Realism in the 480p generation |
| 720p (x1.5) as standard | real detail gain; 1080p costs ~3x the 480p time for little visible gain at normal size |
| FaceRefine + guard, last | the guard removed all collapsed faces at shot ends (24 frames → 1); FaceRefine before the upscale was no better |

## Pipeline steps in detail

| # | Step | Component | What it does |
|---|---|---|---|
| 0 | **Target check** | `bfs_full.py` (insightface `buffalo_l`, YOLO11) | Decides which shots to generate. `PRESENCE=face`: a shot is generated only if the **person to replace** is visible (ArcFace match against `TARGET_FACE`, cosine ≥ 0.35 in ≥ 10% of frames), because BFS always swaps the most visible person and would otherwise turn someone else into the character. `PRESENCE=head` (single-person videos): shots are split into segments with and without the person; segments showing only body parts or a silhouette also stay source. Saved as `output/<job>/target_check.json`. |
| 1 | **Shot split** | `bfs_full.py` | Cuts the clip (`CUTS`, or `auto`: see *Scene cuts* below), converts to 24 fps and resizes to the generation size: short side 480 (736 with `RES=720`), long side from the source aspect, 32-aligned. Adds 9 sacrificial lead frames (H3 leaks the reference pose into the first latent group) and pads to H3's 17n+5 frame grid. Shots run in parallel, one ComfyUI server per GPU. |
| 2 | **Colour-match the references** | `utils/scene_color.py` | Recolours each character reference toward the shot's light (`reinhard_lab`, strength 1.0) with statistics from the **scene without people** (YOLO11-seg), so the original person's clothes cannot tint the character. `REF_COLOR_SCENE=0` uses the whole frames (ComfyUI `ColorTransfer`, the old behaviour). |
| 3 | **Lock the others** (`PROTECT`) | SAM 3.1 (`SAM3_Detect` → `SAM3_VideoTrack`) | Tracks `PROTECT` (threshold 0.7), grows the mask 4 px, inverts it and applies it as the latent edit mask (`H3RelayApplyEditMask`): those people cannot change. The target has **no mask**, so her body, hair and clothes are generated freely and blend with the scene. |
| 4 | **Conditioning** | MiniMax H3 `MiniMaxH3ReferenceToVideo` + `MiniMaxH3AddGuide` | `<Picture 1>`…`<Picture N>` are the (recoloured) character references, e.g. the views of a character sheet. The prompt is `r34l1sm, ` (Realism trigger) + the BFS `body_swap:` template with the character's head-to-toe description. The original shot is the guide video. |
| 5 | **Model stack** | MiniMax H3 + 3 LoRAs | H3 ref2va (int8), shift 12/3, pytorch attention, then DMD ref2va 8-step turbo, **BFS v1.1** (swaps the subject, follows the guide's motion, camera and lighting) and the **Realism People LoRA** 0.8 (skin, hair strands, hands, expressions). |
| 6 | **Sampling** | `SamplerCustomAdvanced` | euler, simple scheduler, 8 steps, CFG 1. The whole frame is generated, guided by the original shot; locked people keep their latent. |
| 6b | **Upscale** (`UPSCALE`, optional) | H3 latent upscaler (`LBH-123-AI`, 2D v1) + refine pass | The stage-1 latent is upscaled (1.5 = 480p → 720p) and refined at the new size with 3 steps (sigmas 0.90 → 0.63 → 0.32), same references and guide, long shots in 73-frame chunks. The Realism LoRA goes in this pass (`REALISM_STAGE=2`, default), where its skin and hair detail has the pixels to show. The result keeps the higher resolution. |
| 7 | **Assembly** | `assemble_bfs.py --raw --letterbox --tone-person` | Drops the lead frames, trims to the shot's length and scales back to the source size. Uses the **pure generated frames** (pasting locked people back from the source only adds a visible edge). Re-applies the source's letterbox bars, and in shots where the generated person is much brighter than the original (dark, low-key shots) matches her tone to the original person. |
| 8 | **Final cut** | `scripts/run_swap.sh` | Generated and source shots are concatenated with the source audio. Optional `SPLICE="N=clip.mp4"` replaces a shot with a prepared clip. Cropped bars (`CROP`) are padded back. |
| 9 | **FaceRefine + identity guard** (default on) | `utils/face_refine.py` (`Carasibana/ComfyUI-H3-FaceRefine`) | Per generated shot: tracks her face (identity from `FACE_REFINE_REF`), re-renders the face crop with low denoise and pastes it back. Then the **identity guard** compares every refined frame with the frame before refining (ArcFace similarity to `FACE_REFINE_REF`): if the refined face is worse by more than 0.08, or lost, that frame cross-fades back to the unrefined one. Shots shorter than 24 frames are not refined. Runs last, because any later generation pass would re-render the face. |

### Scene cuts (`CUTS=auto`)
A missed or misplaced cut is the most damaging error: the shot is generated as one clip, so the model draws the
person into a scene without her (e.g. over a close-up of a hand) and her face drifts toward the original across
the change. `auto_cuts` in `bfs_full.py` detects sensitively and then checks every candidate:
1. **Candidates:** PySceneDetect `AdaptiveDetector` at 1.5 (default 3.0), shots ≥ 8 frames, so dissolves and
   cuts between similar-looking shots are found.
2. **Colour check:** Bhattacharyya distance between the HSV histograms of the 6 frames before and after ≥ 0.38.
   Fast motion (hair, arms) triggers step 1 but keeps the colours.
3. **Snap:** the cut moves to the sharpest frame change within ±3 frames. The detector can land 1–2 frames late;
   2 frames of the next shot (her head) made a sleeve-only segment count as "person visible", and a face was
   invented there.
4. **Same-scene check:** ORB features matched across the cut. A real cut shares almost none (≤ 5% on all
   labelled cuts); an occluder crossing the same shot (a puff of smoke) keeps the background (27%) and is dropped
   (threshold 15%).
5. **Flash frames:** 1–2 frames of another shot cut into a scene (e.g. the original person flashing up in fog)
   are too short for step 1. A frame that differs from both neighbours (mean grey difference ≥ 2.5/255) while the
   neighbours are close to each other (≤ half of that) gets its own cuts, so it is generated, not kept.

On three hand-labelled clips (33 cuts including 2 flash frames) every cut is found, all frame-exact except one
fade frame (728 instead of 727), with no extra cut. Each decision is printed. Hand-set `CUTS` in `case.env` win.

### Character sheet (recommended reference)
**Built automatically as the first step** when `case.env` sets `SHEET_PERSON` (a cut-out of the character)
and `SHEET_FACE` (her face photo). The sheet wears the clothes the result should show:
- `KEEP_OUTFIT=0`: her **own** clothes from `SHEET_PERSON` (`input/sheet_<name>_own/`, or `SHEET_OWN_DIR`).
- `KEEP_OUTFIT=1`: the **video's** outfit. It is cut from the video (`utils/make_outfit_ref.py`: the frame where
  the person is largest and most complete, cut out on white, head removed), or taken from `OUTFIT_REF`, and
  described by `OUTFIT_TEXT` (`input/sheet_<name>_outfit/`, or `SHEET_OUTFIT_DIR`). Without this, references in
  her own clothes contradicted "keep the clothing" and the model kept the original person, face included.

An existing sheet is reused. `utils/sheet_refs.py` assigns the panels by content (face close-ups → face
references, the most complete frontal body view → `CHAR_REF`, the rest → `CHAR_REF_EXTRA`), and `SHEET_FACE`
is added as a face reference and becomes the FaceRefine identity. A body panel without a face is the back view
(`CHAR_REF_BACK`): in shots where no face is visible it leads the references as `<Picture 1>`. With the front view
leading, a small dark figure walking away was drawn as a large frontal portrait (same result with two seeds);
with the back view leading it stayed a small figure from behind. Without `SHEET_PERSON`, the `CHAR_REF*`
settings are used as given. The manual steps below do the same by hand.

`utils/make_char_sheet.py` renders the character from several angles with **Qwen-Image-2.1** + the
lightx2v Lightning LoRA (4 steps), and the views go to H3 as references (`CHAR_REF` + `CHAR_REF_EXTRA`).
A single photo pulls every shot toward its own pose and framing; a sheet gives BFS the identity and
outfit from front, side and back.

1. **Stage 1, front view:** the person photo (`--person`, a cut-out on white) and a face close-up (`--face`)
   become one clean front view with the arms down (`front_reference.png`). With `--outfit` (a crop of the
   outfit, head cut off) she is dressed in that outfit, e.g. the clothes of the video. Stage 1 renders at
   `--front-resolution 1024`; higher resolutions tend to draw the person twice.
2. **Stage 2, the sheet (default `--mode grid`):** one generation from the front view with the
   character-sheet prompt (`GRID_PROMPT`: face close-up, full-body front / back / side, three-quarter front /
   back, plain light grey background), at `--grid-resolution 2048` (~2496x1664). The model may choose its own
   layout (a 3x2 grid, or one row of 4 panels); the panels are found automatically (background gaps between
   figures, thin border lines ignored) and saved as `view_panel1.png`, `view_panel2.png`, ... (left to right,
   top to bottom), plus the whole sheet as `sheet_grid.png` and a preview `sheet.jpg`.
   `--mode views` instead makes one generation per camera angle (`--views front,left,back,face`).
3. **Add her face photo (and the face panel) as face references** (`CHAR_REF_FACE`). Full-body views carry
   little face detail; in an extreme close-up BFS then keeps the original face (similarity to the new face
   0.16 → 0.56 with the face photo). They are only used where a face is visible: in a back view they turned
   the shot into a frontal face portrait.

```bash
.venv/bin/python utils/make_char_sheet.py --person cases/<name>/input/person_cutout.png \
    --face cases/<name>/input/face.jpg [--outfit cases/<name>/input/outfit_nohead.jpg] \
    --out cases/<name>/input/sheet              # -> front_reference.png, sheet_grid.png, view_panel*.png
# --reuse-front: keep an existing front_reference.png / view_front.png and only redo the sheet
```
```bash
# case.env: body panels as references, face panel + real face photo as face references
CHAR_REF=input/sheet/view_panel2.png                                     # front
CHAR_REF_EXTRA="input/sheet/view_panel3.png input/sheet/view_panel4.png"  # side, back
CHAR_REF_FACE="input/sheet/view_panel1.png input/face.png"               # face close-up, face photo
```

### Reference colour (`REF_COLOR_SCENE`, `REF_COLOR_MODE`)
Whole-frame statistics count the replaced person's clothes as light: in a close-up dominated by a red dress the
new character's skin came out flushed (redness +31 vs +25 in the source). Four variants, same seed, 480p (jennie):

| variant | identity | skin redness (red close-up) | vs source skin | vs her own photo |
|---|---|---|---|---|
| `scene` (default): scene without people | 0.345 | +21 (+17) | 4.2 | 6.0 |
| `scene_lum`: same, colour only at 30% | 0.323 | +20 (+17) | 4.5 | 5.0 |
| none (`REF_COLOR=`) | 0.359 | +19 (+17) | 5.8 | 3.7 |
| `skin`: light on the original's face skin | 0.345 | +24 (+25) | 1.9 | 9.1 |

Once the outfit is out of the statistics the recolouring matters little: the model takes most of its light from
the guide video. `scene` stays the default (it should help where coloured light changes skin, e.g. a night scene);
no recolouring is an equal alternative. `skin` copies the original person's skin tone onto the character (pinker,
9.1 from her own photo) and is not recommended. At 720p `scene` vs whole frames: identity 0.401 vs 0.408, red
close-up +14 vs +31. Not implemented: an automatic redness check after generation.

## Settings
`scripts/run_swap.sh` reads `cases/<name>/case.env`. Any setting can be overridden on the command line
(`CASE=cases/<name> SEED=7 bash scripts/run_swap.sh`).

| setting | meaning |
|---|---|
| `VIDEO` | the source video (any size and fps) |
| `CUTS` | shot boundaries in source frames, or `auto` (sensitive PySceneDetect + colour verification, see *Scene cuts*) |
| `CHAR_REF` / `CHAR_REF_EXTRA` / `CHAR_REF_BACK` | set automatically from the sheet with `SHEET_PERSON`; by hand otherwise: the main reference, extra views (each another `<Picture N>`), and the back view that leads in shots without a face |
| `CHAR_REF_FACE` | face references (the real face photo, a face close-up panel), space-separated; added **only to shots where a face is visible** in the source (insightface, ≥ 20% of frames). In back views they would turn the shot into a face portrait |
| `CHAR_DETAILS` | the character's head-to-toe description, starting with gender |
| `TARGET_FACE` | a face crop of the person to **replace**; shots are generated only where this identity is visible. Without it, `TARGET_SEX=F/M` counts any face of that gender |
| `TARGET_TEXT` | how the prompt names the person to replace |
| `PROTECT` / `PROTECT_TEXT` | SAM 3.1 text of the people to **keep** (latent lock), and how the prompt names them; empty = nobody |
| `PRESENCE` | `face` (default, several people): generate where the target's face is seen. `person` (one person): YOLO11 detects her body in every frame (backs, silhouettes, partial bodies) and splits shots into segments with and without her (gaps < 0.5 s and blips < 2 frames smoothed). `head`: like `person`, but a segment also stays source if her head (YOLO11-pose nose/eye/ear keypoints, also from behind) is seen in < 30% of it (only a hem or a sleeve: BFS would invent a whole person) or if she is a silhouette (median brightness in her body box < 6/255) |
| `SHEET_PERSON` / `SHEET_FACE` | the character's full-body cut-out and face photo: the character sheet is built first and replaces `CHAR_REF*` (see *Character sheet*). `SHEET_OWN_DIR` / `SHEET_OUTFIT_DIR` choose where the sheets go |
| `KEEP_OUTFIT=1` | identity, face, hair and body come from the references; clothing stays as worn in the video. The sheet is then built in the video's outfit (cut from the video, or `OUTFIT_REF`; `OUTFIT_TEXT` = short outfit description, Chinese works best). With references in her own clothes the model kept the original person, face included |
| `REF_COLOR_SCENE` (1) / `REF_COLOR_MODE` (`scene`) | reference colour statistics from the scene only (`0` = whole frames); mode `scene`, `scene_lum` or `skin` (see *Reference colour*) |
| `FACE_ONLY=1` | swap only the face: SAM tracks "face" as the edit mask, the prompt keeps hair/clothing/pose, and assembly composites the face region onto the source |
| `TONE_PERSON` (0.8) | assembly post-process (`utils/tone_match.py`): where the generated person is > 1.25x brighter than the original, her LAB tone is matched to the original person inside her box, feathered. Normally lit shots are left alone so the new identity keeps its colours |
| `LIGHT_MATCH=1` | adds a lighting-match sentence to the prompt (off by default; did not help in tests) |
| `SOFT_FACE` (1) / `SOFT_FACE_SCALE` (0.15) | soften the original person's face in the guide video (identity removed, pose and coarse expression kept), so the model cannot copy the original face in close-ups. Per shot, `shot_edits.json` can set a stronger `soft_face_scale` / `soft_face_px` (web UI "Identity strength") |
| `AUTO_RETRY` (0) | after generation `utils/identity_check.py` always checks every shot and warns when its face matches the original person more than the character; `AUTO_RETRY=N` re-renders those shots with the next N seeds and keeps the best attempt |
| `RELIGHT` (0) | 0-1: re-light generated shots with the source's broad light (`utils/relight.py`) instead of `TONE_PERSON`; silhouettes are then generated too. Per shot via `shot_edits.json` (`relight`) |
| `UPSCALE` (1.5) / `REALISM_STAGE` (2) | two-stage generation (step 6b): `UPSCALE=1.5` gives 720p from 480p (standard), `0` = 480p only for quick tests; 2.25 gives 1080p but takes ~3x the time of the 480p generation. `REALISM_STAGE=1/2/both` puts the Realism LoRA in stage 1, stage 2 or both; 2 tested best (identity 0.366 vs 0.350, sharper faces) |
| `FACE_REFINE` (1) / `FACE_REFINE_REF` | step 9, FaceRefine + identity guard (default on; `0` = off). Identity from `FACE_REFINE_REF` (default the last `CHAR_REF_FACE` entry); shots that were not generated are skipped. `face_refine.py --guard-margin` (0.08, < 0 = guard off) and `--min-shot` (24) tune the guard. `FACE_REFINE=before` (with `UPSCALE`) refines the 480p shots before the upscale instead; it tested no better |
| `CROP` | `auto` or `w:h:x:y`: crop constant black bars (pillarbox / letterbox) off the video for generation, pad the result back afterwards |
| `SOLO_SHOTS` | shots where the target is the only person (the prompt drops the "keep the others" note); automatic for all shots when `PROTECT` is empty |
| `REALISM` (0.8), `REF_COLOR` (`reinhard_lab:1.0`, empty = no recolouring), `RES` (480), `SEED` (42) | quality settings, as in the pipeline table |
| `JOB` / `OUT` | folder names under `output/` (default `gen_seed<SEED>`, `final_seed<SEED>`, or as set in `case.env`) |
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
- YOLO11 `yolo11m.pt`, `yolo11m-pose.pt`, `yolo11m-seg.pt`; insightface `buffalo_l` is fetched on first use
- latent upscaler (`UPSCALE`): `minimax_h3_latent_upscaler_2d_conv_v1_fp16` (`latent_upscale_models/`)
- custom nodes installed by `scripts/setup.sh`: `akatz-ai/h3-relay`, `Carasibana/ComfyUI-H3-FaceRefine`,
  `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`

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
CASE=cases/<name> UPSCALE=0 JOB=gen480 OUT=final480 bash scripts/run_swap.sh   # quick 480p test
CASE=cases/<name> FACE_REFINE=0 bash scripts/run_swap.sh       # skip FaceRefine (~10 min faster)
CASE=cases/<name> ASSEMBLE_ONLY=1 bash scripts/run_swap.sh     # re-assemble without generating
```
Each run writes to `cases/<name>/output/`:
- `<OUT>/swapped.mp4`: the final video with the source audio. It is the only file kept in `<OUT>`.
- `<JOB>/target_check.json` and `<JOB>/picks_<OUT>.json`: the per-shot decisions and what was assembled
- `<JOB>/shotNN/`: the generation (workflow, prompt, recoloured references, lock masks)

## 3b. Web UI: "Swap character" (from Safari on another device)
`webui.py` (Gradio, dark theme). One workflow: upload a source video, say who to replace, pick a character from
the library (`characters/<name>/`: cut-out, face photo, description; new ones are added from photos) and press
**Swap character**. Each run is a new project, Demo 1, Demo 2, ... (`cases/demoN`, outputs in `output/gen` and
`output/final`). The result is the final video plus its list of shots; select a swapped shot, type an instruction
("softer smile, she looks down") and **Re-render shot**: only that shot is generated again (the instruction is
appended to its prompt via `output/gen/shot_edits.json`, which also holds the shot's identity strength, light,
clothes and take), then the final video is rebuilt. **Plan edit** turns a free-form request in any language into those
settings (`utils/edit_planner.py`, Qwen3-VL on GPU 0).
Options: keep the video's clothes, 480p test / 720p final, re-light, people to keep unchanged.
The project logic is in `utils/swap_project.py`, shared with the web-app worker (3c): both UIs see the same
projects and characters.
```bash
bash scripts/start_ui.sh            # http://127.0.0.1:7860, login swap / password in .ui_password
ssh -N -J <user>@ilab2.cs.rutgers.edu -L 7860:127.0.0.1:7860 <user>@lerna.cs.rutgers.edu   # from a laptop abroad
```
Run it inside `tmux` to keep it alive after logging out. `UI_HOST=0.0.0.0` exposes it on the network instead.

## 3c. Web-app worker (GoldenAI web-app, "Character swap")
The GoldenAI web-app (Flask + React, separate repo) offers the same workflow as `webui.py`. It does not run the
pipeline itself: it enqueues jobs on Redis streams and this repo's worker runs them (`python -m worker`). The message
format is `contracts/` (1.2.0), vendored byte-identical in the web-app and Research repos: never edit it here alone;
change it in one repo, `python contracts/python/checksum.py --write`, then `scripts/sync-contracts.sh <other repo>`.

| role | stream / group | does | GPU |
|---|---|---|---|
| `render` | `jobs.character_swap` / `swap-workers` | a whole video (`shot_id` "all") or one shot ("3" = `ONLY=3`, with the edit merged into `shot_edits.json`) through `scripts/run_swap.sh`; progress from the log, Stop kills it and clears the ComfyUI queues; then `manifest.json` (final video, shots, clips, thumbnails) | the ComfyUI servers (1-3) |
| `util` | `jobs.character_swap_util` / `swap-util-workers` | `plan_edit` (edit planner) and `add_character` (cut-out + face crop of an uploaded photo), so they answer while a render runs | `PLANNER_GPU` (0) |

Run **exactly one render worker per cluster** (one job drives all three ComfyUI servers); the render role claims a
job from a dead worker only after 4 h (`WORKER_CLAIM_IDLE_MS`), longer than any render.

**Shared folders.** The web-app and the worker must see the same `SWAP_CASES_DIR` (projects `demoN/`) and
`SWAP_CHARS_DIR` (character library); by default the worker uses `cases/` and `characters/` of this repo, the
Gradio UI's folders. The web-app writes `project.json` and `input/video.*` (and a new character's photos in
`<name>/_upload/`); the worker writes everything else (`case.env`, `output/`, `manifest.json`).

**Settings** (`worker/.env`, see `worker/.env.example`, or the environment): `REDIS_URL` and `STREAM_PREFIX` (must
match the web-app's), `SWAP_CASES_DIR`, `SWAP_CHARS_DIR`, `PLANNER_GPU`, optional `WORKER_NAME`,
`WORKER_CLAIM_IDLE_MS`, `SWAP_RUN_CMD`. The worker needs Python >= 3.11 and `redis` (installed by `setup.sh`).

**Start order on the cluster** (one `tmux` window each):
```bash
redis-server --bind 127.0.0.1 --requirepass <password> --appendonly yes     # Redis 7 (e.g. from conda)
GPU=1 PORT=8188 bash scripts/start_comfy.sh                                  # x3, as in section 2
bash scripts/start_workers.sh                                                # render + util (ROLE=render|util for one)
# then the web-app (its README): SWAP_CASES_DIR / SWAP_CHARS_DIR / REDIS_URL set to the same values
```
Tests (no GPU, fake pipeline `tests/fake_run_swap.sh`, fake Redis):
`python -m venv .venv-test && .venv-test/bin/pip install pytest fakeredis redis jsonschema numpy opencv-python-headless && .venv-test/bin/python -m pytest tests`

## 4. Files
```
char_swap/
├── run_bfs.py, bfs_full.py, assemble_bfs.py   main pipeline (one shot / all shots / assembly)
├── webui.py   web UI (scripts/start_ui.sh)
├── worker/    web-app worker (python -m worker --role render|util, scripts/start_workers.sh)
├── contracts/ message format shared with the web-app (vendored, do not edit here alone)
├── scripts/   run_swap.sh (entry point), setup.sh, download_models.sh, start_comfy.sh, start_workers.sh
├── utils/     swap_project, comfy_client, tone_match, scene_color, make_char_sheet, make_outfit_ref, sheet_refs, ...
├── tests/     worker + swap_project tests (pytest, no GPU)
├── cases/     <name>/input, case.env, output
└── log.md     per-case results and findings
```

| file | role |
|---|---|
| `scripts/run_swap.sh` | **entry point** (`CASE=cases/<name>`): character sheet → cuts + target check → BFS with scene-coloured references and the SAM lock → upscale with Realism → assembly → FaceRefine + guard |
| `bfs_full.py` | scene cuts (`auto_cuts`), target check (`--presence`, `--target-face`, `--check-only`), shot split, padding, dispatches `run_bfs.py` over the GPUs; writes to `$CHAR_SWAP_OUT/<job>` |
| `run_bfs.py` | builds and submits the BFS v1.1 graph for one shot (`--protect --protect-only`, `--realism`, `--ref-color`, `--ref-extra`, `--keep-outfit`, `--face-only`) |
| `assemble_bfs.py` | per-shot assembly: `--raw`, `--letterbox`, `--tone-person`, optional mask composite / colour match; source size and fps, concat, source audio |
| `utils/make_char_sheet.py` | character sheet with Qwen-Image-2.1 + Lightning (see above) |
| `utils/tone_match.py` | per-frame person tone matching (YOLO11 boxes, LAB Reinhard), used by `assemble_bfs.py --tone-person` |
| `utils/comfy_client.py` | minimal ComfyUI HTTP client (submit, wait, download) |
| `utils/make_fullbody_ref.py` | full-body reference generated by H3 from a face photo (weaker than a real photo or a sheet) |
| `utils/scene_color.py` | reference recolouring from the scene without people (`REF_COLOR_MODE` scene / scene_lum / skin) |
| `utils/make_outfit_ref.py` | outfit crop from the video (largest complete frontal frame, cut out, head removed) for `KEEP_OUTFIT=1` sheets |
| `utils/sheet_refs.py` | sorts a sheet's panels into main / extra / back / face references by content |
| `utils/swap_project.py` | character-swap projects (`cases/demoN`) shared by `webui.py` and the worker: case preparation, pipeline environment, shot edits, progress from the log, manifest (shot clips + thumbnails), character cut-out |
| `worker/` | web-app worker: `loop.py` (Redis consumer), `render.py` (run_swap.sh jobs), `util.py` (plan_edit, add_character), `capabilities.py` (the options the web-app shows) |
| `utils/face_refine.py` | FaceRefine + identity guard, the last step (needs `Carasibana/ComfyUI-H3-FaceRefine`); `--shots-job` refines a job's generated shots, `--guard-only` re-applies the guard to refined shots |

## Licences
MiniMax H3, the BFS LoRA and the Realism People LoRA follow the MiniMax H3 Community Licence. Read
its territorial terms before you use them. The BFS card also asks for appropriate rights and consent
for the reference images and guide videos. Qwen-Image-2.1 follows the Qwen research licence.
