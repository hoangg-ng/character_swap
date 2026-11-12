#!/usr/bin/env bash
# Character swap for one case: replace ONE chosen person in a video with a character, keep everything else.
#
#   CASE=cases/<name> bash scripts/run_swap.sh            # settings from cases/<name>/case.env
#   CASE=cases/<name> SEED=7 RES=720 bash scripts/run_swap.sh   # any setting can be overridden on the command line
#
# A case folder:  cases/<name>/input/   video, character image(s), target face crop, ...
#                 cases/<name>/case.env settings (paths relative to the case folder)
#                 cases/<name>/output/  generations (<JOB>/shotNN/...) and final videos (<OUT>/swapped.mp4)
#
# Settings (case.env or environment)
#   VIDEO         source video (any size / fps; shots are converted to 24 fps for generation)
#   CUTS          shot boundaries in source frames, "a,b,c" or "auto" (PySceneDetect)            [auto]
#   CHAR_REF      the new character (best: full body cut-out on white)
#   CHAR_REF_EXTRA  more views of the same character (character sheet, utils/make_char_sheet.py), space-separated;
#                   each becomes another H3 reference <Picture 2..N>                                  [""]
#   CHAR_REF_FACE   face references (face photo, face close-up), space-separated; added only to shots where a
#                   face is visible in the source (in back views they turn the shot into a face portrait) [""]
#   CHAR_DETAILS  head-to-toe description of the character, starting with gender
#   TARGET_FACE   face crop of the person to REPLACE. Each shot is generated only where this identity is
#                 visible (insightface); the other shots stay exactly as the source.
#   TARGET_SEX    fallback when no TARGET_FACE: any F / M face counts as the target                [F]
#   PRESENCE      face   = shot generated only where the target's face is seen (multi-person videos)  [face]
#                 person = shot generated wherever anyone is visible (single-person videos: backs,
#                          silhouettes, partial bodies too); scenery-only shots stay source
#                 head   = like person, but segments showing only body parts (hem, sleeve: no head) also
#                          stay source - the model would invent a whole person there
#   TARGET_TEXT   how the prompt names the person to replace                                       [the woman]
#   PROTECT       SAM 3.1 text of people to keep unchanged (latent lock), "" = none               [""]
#   PROTECT_TEXT  how the prompt names them                                                         [""]
#   FACE_ONLY=1   swap ONLY the face: SAM 3.1 tracks "face" -> edit mask (everything else locked), prompt keeps
#                 hair/clothing/pose of the video, assembly composites the face region back onto the source
#   FACE_GROW     px the face mask is grown (room for the jaw line / hairline)                         [12]
#   TONE_PERSON   post-process: in shots where the generated person is >1.25x brighter than the original
#                 (e.g. a dark, low-key shot), match her tone to the original person (tone_match.py)     [0.8]
#   LIGHT_MATCH=1 add the lighting-match instruction (light only from the video's light sources, not from the
#                 reference photo) - for dark / low-key shots where the swapped person looks too bright      [0]
#   KEEP_OUTFIT=1 face-only character reference: swap face/head/hair, keep the clothing of the video     [0]
#   SHEET_PERSON / SHEET_FACE  cut-out and face photo of the character: a character sheet is built first and used
#                 as the references (replaces CHAR_REF*). KEEP_OUTFIT=1 -> sheet in the video's outfit, 0 -> in her
#                 own clothes. OUTFIT_TEXT: short outfit description for KEEP_OUTFIT=1 (Chinese works best).
#                 OUTFIT_REF: outfit crop (head cut off) to use instead of the automatic one; SHEET_OUTFIT_DIR:
#                 / SHEET_OWN_DIR: where the sheets go   [input/outfit_ref_nohead.png, input/sheet_<name>_outfit|_own]
#   CROP          "auto" or "w:h:x:y": crop constant black bars (pillarbox / letterbox) off the video before
#                 generation and pad the result back to the original size afterwards                    [""]
#   UPSCALE       two-stage: latent-upscale the stage-1 result by this factor (H3 latent upscaler) and refine
#                 at the higher resolution: 1.5 = 480p -> 720p (standard), 0 = off (quick 480p tests)    [1.5]
#   REALISM_STAGE with UPSCALE: Realism LoRA in stage 1, 2 or both                                       [2]
#   FACE_REFINE (1 = default) last step: FaceRefine (utils/face_refine.py, Carasibana/ComfyUI-H3-FaceRefine) on the assembled
#                 result - crops her face, re-renders it larger (identity from FACE_REFINE_REF, Realism if REALISM>0)
#                 and pastes it back; shots that were not generated are skipped. An identity guard reverts any
#                 frame whose face got less like the reference. Pipeline: generation + Realism -> (UPSCALE: upscale
#                 with Realism in the upscale pass) -> FaceRefine + guard. 0 = off                         [1]
#   FACE_REFINE=before (needs UPSCALE): 480p generation without Realism -> FaceRefine on each shot -> upscale with
#                 Realism in the upscale pass (FaceRefine works best while the face is small)
#   REF_COLOR_SCENE (1 = default)  take the reference colour statistics from the scene only (people masked out), so the
#                 replaced person's clothes do not tint the character; 0 = whole frames (old)       [1]
#   FACE_REFINE_REF  face photo for FaceRefine's identity reference                      [last CHAR_REF_FACE entry]
#   SOLO_SHOTS    shots where the target is the only person (no "keep the others" prompt note)    [""]
#   REALISM (0.8)  REF_COLOR (reinhard_lab:1.0)  RES (480|720)  SEED (42)  MIN_TARGET (0.1)
#   SPLICE="N=file.mp4 ..."   replace shot N by a prepared clip (output size, 24 fps)
#   ONLY="2,4"                generate only these shots (others reuse existing results in $JOB)
#   ASSEMBLE_ONLY=1           re-assemble $JOB without generating;  CHECK_ONLY=1  print the target check only
#   JOB (gen_seed$SEED), OUT (final_seed$SEED)   folder names under the case's output/
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"; PY="$R/.venv/bin/python"
CASE=${CASE:?set CASE=cases/<name>}; CASE=$(cd "$CASE" && pwd)
set -a; . "$CASE/case.env"; set +a                # case.env uses ${VAR:-default}, so the environment wins
abspath() { case "$1" in /*|"") echo "$1";; *) echo "$CASE/$1";; esac; }
VIDEO=$(abspath "${VIDEO:?VIDEO not set}")
[ -z "${SHEET_PERSON:-}" ] && : "${CHAR_REF:?set CHAR_REF, or SHEET_PERSON (+ SHEET_FACE) to build a character sheet}"
CHAR_REF=$(abspath "${CHAR_REF:-}")
TARGET_FACE=$(abspath "${TARGET_FACE:-}")
EXTRA=""; for x in ${CHAR_REF_EXTRA:-}; do EXTRA="$EXTRA${EXTRA:+,}$(abspath "$x")"; done
FACEREF=""; for x in ${CHAR_REF_FACE:-}; do FACEREF="$FACEREF${FACEREF:+,}$(abspath "$x")"; done
CUTS=${CUTS:-auto}; CHAR_DETAILS=${CHAR_DETAILS:?CHAR_DETAILS not set}
TARGET_SEX=${TARGET_SEX:-F}; PRESENCE=${PRESENCE:-face}; TARGET_TEXT=${TARGET_TEXT:-the woman}
PROTECT=${PROTECT:-}; PROTECT_TEXT=${PROTECT_TEXT:-}; SOLO_SHOTS=${SOLO_SHOTS:-}
REALISM=${REALISM:-0.8}; REF_COLOR=${REF_COLOR-reinhard_lab:1.0}; RES=${RES:-480}; SEED=${SEED:-42}
MIN_TARGET=${MIN_TARGET:-0.1}; UPSCALE=${UPSCALE:-1.5}; REALISM_STAGE=${REALISM_STAGE:-2}; KEEP_OUTFIT=${KEEP_OUTFIT:-0}; LIGHT_MATCH=${LIGHT_MATCH:-0}; TONE_PERSON=${TONE_PERSON:-0.8}; FACE_ONLY=${FACE_ONLY:-0}; FACE_GROW=${FACE_GROW:-12}; SPLICE=${SPLICE:-}; ONLY=${ONLY:-}
JOB=${JOB:-gen_seed$SEED}; OUT=${OUT:-final_seed$SEED}
export CHAR_SWAP_OUT="$CASE/output"; W_DIR="$CHAR_SWAP_OUT/$JOB"; O_DIR="$CHAR_SWAP_OUT/$OUT"

# optional: crop black bars off the source (pillarbox / letterbox); the result is padded back at the end
SRC_VIDEO="$VIDEO"; CROP=${CROP:-}; CROP_IN="$CROP"; PAD=""
if [ -n "$CROP" ]; then
  if [ "$CROP" = auto ]; then
    CROP=$($PY -c "
import cv2, numpy as np
c=cv2.VideoCapture('$VIDEO'); mc=mr=None; i=0
while True:
    ok,f=c.read()
    if not ok: break
    if i%6==0:
        a,b=f.mean(axis=(0,2)),f.mean(axis=(1,2)); mc=a if mc is None else np.maximum(mc,a); mr=b if mr is None else np.maximum(mr,b)
    i+=1
cc,rr=np.where(mc>12)[0],np.where(mr>12)[0]; x0,x1,y0,y1=cc[0],cc[-1]+1,rr[0],rr[-1]+1
x0+=x0%2; y0+=y0%2; w=(x1-x0)//2*2; h=(y1-y0)//2*2
print(f'{w}:{h}:{x0}:{y0}' if (w,h)!=(len(mc),len(mr)) else '')")
  fi
  if [ -n "$CROP" ]; then
    mkdir -p "$CHAR_SWAP_OUT"; CV="$CHAR_SWAP_OUT/_source_cropped.mp4"
    [ -s "$CV" ] || ffmpeg -y -loglevel error -i "$VIDEO" -vf "crop=$CROP" -c:v libx264 -crf 10 -pix_fmt yuv420p -c:a copy "$CV"
    read VW VH < <(ffprobe -v error -select_streams v -show_entries stream=width,height -of csv=p=0 "$VIDEO" | tr ',' ' ')
    IFS=: read cw ch cx cy <<< "$CROP"; VIDEO="$CV"
    echo "[crop] $CROP (bars removed for generation, padded back to ${VW}x${VH} at the end)"
  fi
fi

# generation size: short side 480 (or 736 for RES=720), long side from the source aspect, 32-aligned
read W H < <($PY -c "
import cv2; c=cv2.VideoCapture('$VIDEO'); w,h=c.get(3),c.get(4); s=736 if '$RES'=='720' else 480
L=int(round(s*max(w,h)/min(w,h)/32))*32; print(*( (L,s) if w>=h else (s,L) ))")
echo "[case] $CASE  video=$(basename "$VIDEO")  gen=${W}x${H}  job=$JOB  out=$OUT"
# output scale: with UPSCALE the result keeps the higher resolution (factor = upscaled height / source height)
OSCALE=$($PY -c "
import cv2; c=cv2.VideoCapture('$VIDEO'); h=c.get(4); u=float('$UPSCALE')
print(round(max(1.0, $H*u/h), 4) if u>1 else 1.0)")

# Character sheet first (SHEET_PERSON = cut-out of the character, SHEET_FACE = face photo): the references must
# wear the clothes the result should show.
#   KEEP_OUTFIT=1: the sheet wears the VIDEO's outfit (cut from the video by utils/make_outfit_ref.py, or OUTFIT_REF).
#                  With references in her own clothes the model keeps the original person, face included.
#   KEEP_OUTFIT=0: the sheet wears the character's OWN clothes from SHEET_PERSON.
# Rendered once (utils/make_char_sheet.py, grid mode) into SHEET_OUTFIT_DIR / SHEET_OWN_DIR (default
# input/sheet_<name>_outfit | _own) and reused; utils/sheet_refs.py assigns the panels to CHAR_REF /
# CHAR_REF_EXTRA / CHAR_REF_FACE, and SHEET_FACE becomes the FaceRefine identity.
if [ -n "${SHEET_PERSON:-}" ] && [ "${ASSEMBLE_ONLY:-0}" != 1 ]; then
  SP=$(abspath "$SHEET_PERSON"); SO_FACE=$(abspath "${SHEET_FACE:-}")
  NAME=$(basename "${SP%.*}"); NAME=${NAME%_cutout}; NAME=${NAME%_full_body}
  if [ "$KEEP_OUTFIT" = 1 ]; then
    SO_DIR=$(abspath "${SHEET_OUTFIT_DIR:-input/sheet_${NAME}_outfit}")
    SO_OUTFIT=$(abspath "${OUTFIT_REF:-input/outfit_ref_nohead.png}")   # a hand-made crop wins over the automatic one
  else
    SO_DIR=$(abspath "${SHEET_OWN_DIR:-input/sheet_${NAME}_own}")
  fi
  if ! ls "$SO_DIR"/view_panel*.png >/dev/null 2>&1; then
    SHEET_ARGS=(--person "$SP" --face "${SO_FACE:-$SP}" --out "$SO_DIR")
    if [ "$KEEP_OUTFIT" = 1 ]; then
      echo "[sheet] character sheet of $NAME in the video's outfit -> $SO_DIR"
      [ -s "$SO_OUTFIT" ] || $PY utils/make_outfit_ref.py --video "$VIDEO" --out "$SO_OUTFIT" || exit 1
      SHEET_ARGS+=(--outfit "$SO_OUTFIT" --outfit-text "${OUTFIT_TEXT:-与图2完全相同的服装和配饰}")
    else
      echo "[sheet] character sheet of $NAME in her own clothes -> $SO_DIR"
    fi
    $PY utils/make_char_sheet.py "${SHEET_ARGS[@]}" 2>&1 | grep -v -E "running\.\.\.|^\s*$"
    ls "$SO_DIR"/view_panel*.png >/dev/null 2>&1 || { echo "[sheet] FAILED"; exit 1; }
  fi
  eval "$($PY utils/sheet_refs.py --sheet "$SO_DIR" ${SO_FACE:+--face-photo "$SO_FACE"} 2>/dev/null | grep '^CHAR_REF')"
  EXTRA=""; for x in ${CHAR_REF_EXTRA:-}; do EXTRA="$EXTRA${EXTRA:+,}$x"; done
  FACEREF=""; for x in ${CHAR_REF_FACE:-}; do FACEREF="$FACEREF${FACEREF:+,}$x"; done
  [ -n "$SO_FACE" ] && FACE_REFINE_REF="$SO_FACE"
  echo "[sheet] references from $SO_DIR: main $(basename "$CHAR_REF"), $(echo $CHAR_REF_EXTRA | wc -w) extra views, $(echo $CHAR_REF_FACE | wc -w) face refs"
fi

# FACE_REFINE=before (with UPSCALE): 480p generation (no Realism) -> FaceRefine on each generated shot -> upscale
# with Realism in that pass (stage 2 from the refined clips). Three passes of this script; intermediate jobs
# <JOB>_480 and <JOB>_fr, the 480p preview in <OUT>_480.
if [ "${FACE_REFINE:-0}" = before ] && [ "$UPSCALE" != 0 ] && [ "${ASSEMBLE_ONLY:-0}" != 1 ] && [ -z "${CHAR_SWAP_FROM_JOB:-}" ]; then
  echo "[pipeline] 480p -> FaceRefine -> upscale x$UPSCALE + Realism"
  VIDEO="$SRC_VIDEO" CROP="$CROP_IN" JOB="${JOB}_480" OUT="${OUT}_480" UPSCALE=0 REALISM=0 FACE_REFINE=0 bash "$0" || exit 1
  FR_REF="${FACE_REFINE_REF:-}"; [ -z "$FR_REF" ] && for x in ${CHAR_REF_FACE:-}; do FR_REF="$x"; done
  [ -z "$FR_REF" ] && FR_REF="$CHAR_REF"; FR_REF=$(abspath "$FR_REF")
  FR_ONLY=$($PY -c "import json;d=json.load(open('$CHAR_SWAP_OUT/${JOB}_480/target_check.json'));print(','.join(k for k,v in d.items() if v['generate']))")
  echo "[facerefine] on the 480p shots $FR_ONLY (ref=$(basename "$FR_REF"))"
  $PY -u utils/face_refine.py --shots-job "$CHAR_SWAP_OUT/${JOB}_480" --only "$FR_ONLY" --face-ref "$FR_REF" \
      --details "$CHAR_DETAILS" --out "$CHAR_SWAP_OUT/${JOB}_fr" 2>&1 | grep -E --line-buffered "shot [0-9]+:|FINAL" || exit 1
  VIDEO="$SRC_VIDEO" CROP="$CROP_IN" CHAR_SWAP_FROM_JOB="${JOB}_fr" REALISM_STAGE=2 FACE_REFINE=0 bash "$0"
  exit $?
fi

if [ "${ASSEMBLE_ONLY:-0}" != 1 ]; then
  ARGS=(--video "$VIDEO" --cuts "$CUTS" --job "$JOB" --keep "" --min-target "$MIN_TARGET" --only "$ONLY"
        --lead 9 --width $W --height $H --comfy-urls http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190
        --realism "$REALISM" --ref-color "$REF_COLOR" --seed "$SEED" --ref "$CHAR_REF" --details "$CHAR_DETAILS"
        --target "$TARGET_TEXT" --keep-others "${PROTECT_TEXT:-every other person}" --target-sex "$TARGET_SEX" --presence "$PRESENCE"
        --solo-shots "$SOLO_SHOTS")
  [ -n "$TARGET_FACE" ] && ARGS+=(--target-face "$TARGET_FACE")
  [ -n "$PROTECT" ] && ARGS+=(--protect "$PROTECT" --protect-only)
  [ "$KEEP_OUTFIT" = 1 ] && [ "$FACE_ONLY" != 1 ] && ARGS+=(--keep-outfit)
  [ -n "$EXTRA" ] && ARGS+=(--ref-extra "$EXTRA")
  [ -n "$FACEREF" ] && ARGS+=(--ref-face "$FACEREF")
  [ -n "${CHAR_REF_BACK:-}" ] && ARGS+=(--ref-back "$(abspath "$CHAR_REF_BACK")")
  [ "$UPSCALE" != 0 ] && ARGS+=(--upscale "$UPSCALE" --realism-stage "$REALISM_STAGE")
  [ "$LIGHT_MATCH" = 1 ] && ARGS+=(--light-match)
  [ "${REF_COLOR_SCENE:-1}" = 1 ] && [ -n "$REF_COLOR" ] && ARGS+=(--ref-color-scene --ref-color-mode "${REF_COLOR_MODE:-scene}")
  [ "$FACE_ONLY" = 1 ] && [ -z "$PROTECT" ] && ARGS+=(--face-only --subject "face" --mask-grow "$FACE_GROW")
  [ "${CHECK_ONLY:-0}" = 1 ] && ARGS+=(--check-only)
  $PY -u bfs_full.py "${ARGS[@]}" 2>&1 | grep -v -E "running\.\.\.|^\s*$"
  [ "${CHECK_ONLY:-0}" = 1 ] && exit 0
  echo "[bfs] finished"
fi
# assemble: only the shots the target check approved; the rest come from the source
CUTS_USED=$(cat "$W_DIR/cuts.txt")
GEN=$($PY -c "import json;d=json.load(open('$W_DIR/target_check.json'));print(' '.join(k for k,v in d.items() if v['generate']))" 2>/dev/null || echo "")
SHOTS=""
for s in $GEN; do d="$W_DIR/shot$(printf %02d $s)"; ls "$d"/save_*.mp4 >/dev/null 2>&1 && SHOTS="$SHOTS --shot $s=$d" || echo "  shot $s: no result -> source"; done
if [ "$FACE_ONLY" = 1 ]; then ASM=(--color-match --feather 4); else ASM=(--raw --letterbox --tone-person "$TONE_PERSON"); fi
if [ "$OSCALE" != 1.0 ]; then
  ASM+=(--size "$($PY -c "
import cv2; c=cv2.VideoCapture('$VIDEO'); s=$OSCALE; print(f'{int(c.get(3)*s)//2*2}x{int(c.get(4)*s)//2*2}')")")
fi   # face: composite on source
$PY -u assemble_bfs.py --video "$VIDEO" --cuts "$CUTS_USED" --out "$O_DIR" "${ASM[@]}" $SHOTS
if [ -n "$SPLICE" ]; then
  for sp in $SPLICE; do n=${sp%%=*}; f=$(abspath "${sp#*=}"); cp "$f" "$O_DIR/s$n.mp4"; echo "  shot $n: spliced from $f"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$O_DIR/list.txt" -i "$VIDEO" -map 0:v -map 1:a -c:v libx264 -crf 16 \
    -pix_fmt yuv420p -c:a aac -shortest "$O_DIR/swapped.mp4"
  ffmpeg -y -loglevel error -i "$VIDEO" -i "$O_DIR/swapped.mp4" -filter_complex \
    "[0:v]scale=-2:360[a];[1:v]scale=-2:360[b];[a][b]hstack" -map 1:a -c:v libx264 -crf 18 -pix_fmt yuv420p "$O_DIR/side_by_side.mp4"
fi
if [ -n "$CROP" ] && [ -n "${VW:-}" ]; then   # restore the original frame (x OSCALE) with the black bars
  PAD=$($PY -c "s=$OSCALE; e=lambda v:int(v*s)//2*2; print(f'pad={e($VW)}:{e($VH)}:{e($cx)}:{e($cy)}:black')")
  ffmpeg -y -loglevel error -i "$O_DIR/swapped.mp4" -vf "$PAD" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a copy "$O_DIR/_padded.mp4" \
    && mv "$O_DIR/_padded.mp4" "$O_DIR/swapped.mp4"
  ffmpeg -y -loglevel error -i "$SRC_VIDEO" -i "$O_DIR/swapped.mp4" -filter_complex \
    "[0:v]scale=-2:360[a];[1:v]scale=-2:360[b];[a][b]hstack" -map 1:a -c:v libx264 -crf 18 -pix_fmt yuv420p "$O_DIR/side_by_side.mp4"
fi
cp "$W_DIR/target_check.json" "$O_DIR/" 2>/dev/null
echo "[assembled] $O_DIR/swapped.mp4"

# optional last step: FaceRefine on the assembled result
if [ "${FACE_REFINE:-1}" = 1 ]; then
  FR_REF="${FACE_REFINE_REF:-}"; [ -z "$FR_REF" ] && for x in ${CHAR_REF_FACE:-}; do FR_REF="$x"; done
  [ -z "$FR_REF" ] && FR_REF="$CHAR_REF"
  FR_REF=$(abspath "$FR_REF")
  # cuts and not-generated shots in 24 fps frames of the assembled clip
  read FR_CUTS FR_KEEP < <($PY -c "
import json, cv2
fps = cv2.VideoCapture('$SRC_VIDEO').get(cv2.CAP_PROP_FPS) or 24
cuts = [int(round(int(c) * 24 / fps)) for c in '$CUTS_USED'.split(',') if c]
try: d = json.load(open('$W_DIR/target_check.json'))
except Exception: d = {}
keep = [k for k, v in d.items() if not v.get('generate')]
print(','.join(map(str, cuts)), ','.join(keep) or '-')")
  [ "$FR_KEEP" = "-" ] && FR_KEEP=""
  FR_ARGS=(--clip "$O_DIR/swapped.mp4" --source "$SRC_VIDEO" --face-ref "$FR_REF" --cuts "$FR_CUTS" --keep "$FR_KEEP"
           --details "$CHAR_DETAILS" --out "$O_DIR/_facerefine")
  [ "$REALISM" != 0 ] && FR_ARGS+=(--realism "$REALISM")
  echo "[facerefine] ref=$(basename "$FR_REF") keep=${FR_KEEP:-none}"
  if $PY -u utils/face_refine.py "${FR_ARGS[@]}" 2>&1 | grep -E --line-buffered "SKIPPED|FAILED|FINAL"; then
    mv "$O_DIR/swapped.mp4" "$O_DIR/swapped_before_facerefine.mp4"
    cp "$O_DIR/_facerefine/swapped.mp4" "$O_DIR/swapped.mp4"
    cp "$O_DIR/_facerefine/before_after.mp4" "$O_DIR/facerefine_before_after.mp4" 2>/dev/null
    rm -rf "$O_DIR/_facerefine"
    echo "[facerefine] done -> $O_DIR/swapped.mp4 (before: swapped_before_facerefine.mp4)"
  fi
fi
