# Experiment log

Per-case notes, results and what was learned. The general pipeline is described in `README.md`.

| case | video | status |
|---|---|---|
| `cases/hug_scene` | K-drama hug scene, 1280x720, 6 shots, an actress and a man | done: `output/final_sheet_seed42/swapped.mp4` |
| `cases/tiktok_trend` | 948x576, 30 fps, 25.6 s, one woman in a red kimono | done: `output/final_sheet_seed42/swapped.mp4` (4 variants compared) |

---

## Case hug_scene

Replace the actress with the character "phuc" (`input/phuc*.jpg`), keep the man (`PROTECT=man`).
Cuts `33,92,138,187,256` (manual; auto puts the 4/5 cut at 180), generation 864x480, letterbox rows <52 and >667.

### What we learned
- **BFS v1.1 beats the akatz masked edit:** it generates the whole frame from the guide, so she gets new
  hair motion and lighting instead of a pasted-in look.
- **Lock the man, not her.** With two people in shot, BFS tends to swap the more prominent one (it
  swapped the man in shots 1 and 5). A SAM mask on her forced the right person but left a visible
  colour zone. Locking the man instead (edit mask = NOT man) fixes the target in shots 2, 4, 5 and 6
  with nothing visible on her.
- **No pixel paste of the man.** Copying his pixels back from the source after generation left a visible
  box edge (the generated man sits a few pixels off). The latent lock alone keeps him; fresh runs with
  seeds 42 and 7 have no box.
- **Target check.** Shot 1 is her back view and shot 3 has only the man. Generating them (seeds 42 and 7)
  turned the man into the character even with the lock, so shots where her face is not seen stay
  source. With the man's face as the target the check picks exactly shots 1 and 3, so it works both ways.
- **Realism People LoRA** was the most visible gain in a 4-way test (baseline / FaceRefine / Realism /
  both): hair strands, skin and hands.
- **ColorTransfer on the reference** (`reinhard_lab` 1.0): the photo is studio-lit, the scene is night with
  green bokeh. 1.0 gives warmer, less pink-white skin than 0.6; the hair drifts slightly toward copper.
  `mkl_lab` turned the red hair magenta or green.
- **FaceRefine** (`utils/face_refine.py`) is not in the main pipeline: with one face in frame its identity
  check never runs (it re-rendered the man's face in shot 1), and her face is already large
  (147 to 580 px), so the gain is small.
- Generating at 480p fixed a shot-6 failure; 720p is optional.
- The raised arm in the reference photo leaked into the first frames of shot 2; the character sheet
  (arms down) fixed it.

### Results
| output (in `cases/hug_scene/`) | what |
|---|---|
| `output/final_sheet_seed42/swapped.mp4` | **best (case default)**: character sheet `input/sheet/view_{front,left,back}.png` (Qwen-Image-2.1 from `phuc_cutout.png` + `phuc_face.jpg`, her own outfit, arms down). Shots 2, 4, 5, 6 swapped, 1 and 3 kept by the target check, man locked, no box |
| `output/compare_sheet_vs_previous.mp4` | original / previous best (single photo) / character sheet |
| `output/swapped_seed42.mp4`, `swapped_seed7.mp4` | previous best: single reference photo `phuc_cutout.png` |
| `input/` | `hug_scene.mp4`, `phuc*.jpg/png` (character photos), `target_face.jpg` (actress face crop), `sheet/` (character sheet), `shot1_backview_v5.mp4` (optional shot-1 splice from the old akatz pipeline) |

For this sheet the front-left view would not rotate with any seed, so it uses front, side and back.
Reproduce: `CASE=cases/hug_scene bash scripts/run_swap.sh`.

---

## Case tiktok_trend

One woman in a red kimono, no other people (`PROTECT` empty). Generation size 800x480.

- `input/face.jpg` (selfie) and `input/full_body.jpg`: the new character. `face_cutout.png` and
  `full_body_cutout.png` are SAM 3.1 cut-outs on white.
- `input/target_face.jpg`: the original woman's face (frame 405), for `PRESENCE=face`.
- `input/outfit_ref_nohead.jpg`: the video's kimono, cropped below the chin, for the character sheet.
- Shots: PySceneDetect cuts with the last cut corrected (728 → 727). `PRESENCE=head` splits shot 15
  into her 6-frame flash (727–732) and the foggy train, giving 16 segments:

  | generated (her head visible) | kept as source |
  |---|---|
  | 3 (walking away), 4 (walking toward camera), 6, 8 (close-ups), 11 (dance), 13 (spin), 15 (flash) | 1 (silhouette), 2 (kimono hem and feet), 7, 10 (sleeve under the newspaper), 5, 9, 12, 14, 16 (no one) |

  Segment 2 (≈1.2–2.5 s) broke every variant before the head rule: a face, a sideways head or a second
  whole woman was invented in a shot of only a hem and feet. The silhouette (segment 1, person median
  brightness 2/255) was rendered as a brightly lit profile.

### Results
Seed 42, Realism 0.8, ColorTransfer 1.0, clothing kept from the video. Compared in
`output/compare_4_variants.mp4` (original | 1 | 2 / 3 | 4 | the sheet):

| variant | reference | settings | output | verdict |
|---|---|---|---|---|
| 1. Face only | `face_cutout.png` | `FACE_ONLY=1` (SAM "face" edit mask, face composited on the source) | `output/final_face_seed42/` | most faithful to the video: only the face changes; light-brown bangs sit on the black hair |
| 2. Body inferred | `fullbody_ref.png` (H3-generated from `face.jpg`, `utils/make_fullbody_ref.py`) | `KEEP_OUTFIT=1` | `output/final_bodyinfer_seed42/` | weakest: the generated kimono replaces the video's |
| 3. Full body photo | `full_body_cutout.png` | `KEEP_OUTFIT=1` | `output/final_fullbody_seed42/` | good whole-person swap; close-ups become upright with eyes open |
| **4. Character sheet** (case default) | `sheet/view_{front,front_left,left,back}.png` (Qwen-Image-2.1 from `full_body.jpg` + `outfit_ref_nohead.jpg` + `face_cutout.png`) | `CHAR_REF` + `CHAR_REF_EXTRA`, `KEEP_OUTFIT=1` | `output/final_sheet_seed42/` | **best whole-person swap**: kimono closest to the video's, back views stay back views, close-up tilt and closed eyes kept |

- First attempt with the face selfie as a whole-body reference: every shot became a frontal portrait.
- Character sheet, first try: Qwen drew two people side by side (her in her own clothes next to a kimono
  wearer). Fixed by cropping the head off the outfit reference and working in two stages (dress the
  front view, then rotate only that image).
- Dark segments 3 and 13 were rendered 2-5x brighter than the original in every whole-person variant;
  `TONE_PERSON` brings them back into the shadow. The `LIGHT_MATCH` prompt was tried on segment 13 and
  made it brighter, with her turning to the camera.

Reproduce: `CASE=cases/tiktok_trend bash scripts/run_swap.sh` (defaults in `case.env`: character sheet,
`PRESENCE=head`, `KEEP_OUTFIT=1`, Realism 0.8, ColorTransfer 1.0, `TONE_PERSON` 0.8).
