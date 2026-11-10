#!/usr/bin/env bash
# Character swap for one case: replace ONE chosen person in a video with a character, keep everything else.
#
#   CASE=cases/<name> bash run_swap.sh            # settings from cases/<name>/case.env
#   CASE=cases/<name> SEED=7 RES=720 bash run_swap.sh   # any setting can be overridden on the command line
#
# A case folder:  cases/<name>/input/   video, character image(s), target face crop, ...
#                 cases/<name>/case.env settings (paths relative to the case folder)
#                 cases/<name>/output/  generations (<JOB>/shotNN/...) and final videos (<OUT>/swapped.mp4)
#
# Settings (case.env or environment)
#   VIDEO         source video (any size / fps; shots are converted to 24 fps for generation)
#   CUTS          shot boundaries in source frames, "a,b,c" or "auto" (PySceneDetect)            [auto]
#   CHAR_REF      the new character (best: full body cut-out on white)
#   CHAR_REF_EXTRA  more views of the same character (character sheet, make_char_sheet.py), space-separated;
#                   each becomes another H3 reference <Picture 2..N>                                  [""]
#   CHAR_DETAILS  head-to-toe description of the character, starting with gender
#   TARGET_FACE   face crop of the person to REPLACE. Each shot is generated only where this identity is
#                 visible (insightface); the other shots stay exactly as the source.
#   TARGET_SEX    fallback when no TARGET_FACE: any F / M face counts as the target                [F]
#   PRESENCE      face   = shot generated only where the target's face is seen (multi-person videos)  [face]
#                 person = shot generated wherever anyone is visible (single-person videos: backs,
#                          silhouettes, partial bodies too); scenery-only shots stay source
#   TARGET_TEXT   how the prompt names the person to replace                                       [the woman]
#   PROTECT       SAM 3.1 text of people to keep unchanged (latent lock), "" = none               [""]
#   PROTECT_TEXT  how the prompt names them                                                         [""]
#   FACE_ONLY=1   swap ONLY the face: SAM 3.1 tracks "face" -> edit mask (everything else locked), prompt keeps
#                 hair/clothing/pose of the video, assembly composites the face region back onto the source
#   FACE_GROW     px the face mask is grown (room for the jaw line / hairline)                         [12]
#   KEEP_OUTFIT=1 face-only character reference: swap face/head/hair, keep the clothing of the video     [0]
#   SOLO_SHOTS    shots where the target is the only person (no "keep the others" prompt note)    [""]
#   REALISM (0.8)  REF_COLOR (reinhard_lab:1.0)  RES (480|720)  SEED (42)  MIN_TARGET (0.1)
#   SPLICE="N=file.mp4 ..."   replace shot N by a prepared clip (output size, 24 fps)
#   ONLY="2,4"                generate only these shots (others reuse existing results in $JOB)
#   ASSEMBLE_ONLY=1           re-assemble $JOB without generating;  CHECK_ONLY=1  print the target check only
#   JOB (gen_seed$SEED), OUT (final_seed$SEED)   folder names under the case's output/
set -u
R=/research/cbim/vast/hn315/char_swap; cd "$R"; PY="$R/.venv/bin/python"
CASE=${CASE:?set CASE=cases/<name>}; CASE=$(cd "$CASE" && pwd)
set -a; . "$CASE/case.env"; set +a                # case.env uses ${VAR:-default}, so the environment wins
abspath() { case "$1" in /*|"") echo "$1";; *) echo "$CASE/$1";; esac; }
VIDEO=$(abspath "${VIDEO:?VIDEO not set}"); CHAR_REF=$(abspath "${CHAR_REF:?CHAR_REF not set}")
TARGET_FACE=$(abspath "${TARGET_FACE:-}")
EXTRA=""; for x in ${CHAR_REF_EXTRA:-}; do EXTRA="$EXTRA${EXTRA:+,}$(abspath "$x")"; done
CUTS=${CUTS:-auto}; CHAR_DETAILS=${CHAR_DETAILS:?CHAR_DETAILS not set}
TARGET_SEX=${TARGET_SEX:-F}; PRESENCE=${PRESENCE:-face}; TARGET_TEXT=${TARGET_TEXT:-the woman}
PROTECT=${PROTECT:-}; PROTECT_TEXT=${PROTECT_TEXT:-}; SOLO_SHOTS=${SOLO_SHOTS:-}
REALISM=${REALISM:-0.8}; REF_COLOR=${REF_COLOR-reinhard_lab:1.0}; RES=${RES:-480}; SEED=${SEED:-42}
MIN_TARGET=${MIN_TARGET:-0.1}; KEEP_OUTFIT=${KEEP_OUTFIT:-0}; FACE_ONLY=${FACE_ONLY:-0}; FACE_GROW=${FACE_GROW:-12}; SPLICE=${SPLICE:-}; ONLY=${ONLY:-}
JOB=${JOB:-gen_seed$SEED}; OUT=${OUT:-final_seed$SEED}
export CHAR_SWAP_OUT="$CASE/output"; W_DIR="$CHAR_SWAP_OUT/$JOB"; O_DIR="$CHAR_SWAP_OUT/$OUT"

# generation size: short side 480 (or 736 for RES=720), long side from the source aspect, 32-aligned
read W H < <($PY -c "
import cv2; c=cv2.VideoCapture('$VIDEO'); w,h=c.get(3),c.get(4); s=736 if '$RES'=='720' else 480
L=int(round(s*max(w,h)/min(w,h)/32))*32; print(*( (L,s) if w>=h else (s,L) ))")
echo "[case] $CASE  video=$(basename "$VIDEO")  gen=${W}x${H}  job=$JOB  out=$OUT"

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
if [ "$FACE_ONLY" = 1 ]; then ASM=(--color-match --feather 4); else ASM=(--raw --letterbox); fi   # face: composite on source
$PY -u assemble_bfs.py --video "$VIDEO" --cuts "$CUTS_USED" --out "$O_DIR" "${ASM[@]}" $SHOTS
if [ -n "$SPLICE" ]; then
  for sp in $SPLICE; do n=${sp%%=*}; f=$(abspath "${sp#*=}"); cp "$f" "$O_DIR/s$n.mp4"; echo "  shot $n: spliced from $f"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$O_DIR/list.txt" -i "$VIDEO" -map 0:v -map 1:a -c:v libx264 -crf 16 \
    -pix_fmt yuv420p -c:a aac -shortest "$O_DIR/swapped.mp4"
  ffmpeg -y -loglevel error -i "$VIDEO" -i "$O_DIR/swapped.mp4" -filter_complex \
    "[0:v]scale=-2:360[a];[1:v]scale=-2:360[b];[a][b]hstack" -map 1:a -c:v libx264 -crf 18 -pix_fmt yuv420p "$O_DIR/side_by_side.mp4"
fi
cp "$W_DIR/target_check.json" "$O_DIR/" 2>/dev/null
echo "[assembled] $O_DIR/swapped.mp4"
