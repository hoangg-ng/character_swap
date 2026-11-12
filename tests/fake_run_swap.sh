#!/usr/bin/env bash
# Stand-in for scripts/run_swap.sh in the worker tests (SWAP_RUN_CMD): prints the same stage markers, writes the
# files the worker reads (cuts.txt, target_check.json, shotNN/, final/swapped.mp4) and honours ONLY.
#   FAKE_FAIL=1     fail after the sheet stage (exit 1)
#   FAKE_SLEEP=N    sleep N s while "generating" each shot (to test cancelling)
# Source video of the test projects: 72 frames at 24 fps; shots 1 and 3 show the person, shot 2 does not.
set -u
CASE_DIR=$(cygpath -u "$CASE" 2>/dev/null || echo "$CASE")
G="$CASE_DIR/output/gen"; F="$CASE_DIR/output/final"
mkdir -p "$G" "$F"
echo "[case] $CASE_DIR  job=${JOB:-}  out=${OUT:-}"
echo "[env] SEED=${SEED:-} ONLY=${ONLY:-} KEEP_OUTFIT=${KEEP_OUTFIT:-} UPSCALE=${UPSCALE:-} RELIGHT=${RELIGHT:-} PROTECT=${PROTECT:-} SHEET_PERSON=${SHEET_PERSON:-} SHEET_FACE=${SHEET_FACE:-}"
echo "[sheet] character sheet in her own clothes -> input/sheet"
if [ -n "${FAKE_FAIL:-}" ]; then echo "boom: the sheet model ran out of memory"; exit 1; fi
echo "auto cuts: 24,48"
echo "segments with/without the person: cuts 24,48"
echo "target check shot 1: target visible in 90% of frames -> generate"
echo "target check shot 2: target visible in 0% of frames -> KEEP SOURCE"
echo "target check shot 3: target visible in 80% of frames -> generate"
echo "24,48" > "$G/cuts.txt"
cat > "$G/target_check.json" <<'JSON'
{
 "1": {"target_frac": 0.9, "generate": true, "frames": [0, 24], "by": "head"},
 "2": {"target_frac": 0.0, "generate": false, "frames": [24, 48], "by": "head"},
 "3": {"target_frac": 0.8, "generate": true, "frames": [48, 72], "by": "head"}
}
JSON
SHOTS="${ONLY:-1,3}"
for s in ${SHOTS//,/ }; do
  echo "shot $s: generating on http://127.0.0.1:8188"
  [ -n "${FAKE_SLEEP:-}" ] && sleep "$FAKE_SLEEP"
  d="$G/shot$(printf %02d "$s")"; mkdir -p "$d"; echo "generated" > "$d/save_00001.txt"
  echo "  running... 12s"
  echo "shot $s: done"
done
echo "[bfs] finished"
ffmpeg -y -loglevel error -f lavfi -i testsrc=size=160x120:rate=24 -frames:v 72 -pix_fmt yuv420p "$F/swapped.mp4" || exit 1
echo "[assembled] $F/swapped.mp4"
echo "[facerefine] ref=face.png keep=2"
echo "[done] $F/swapped.mp4"
