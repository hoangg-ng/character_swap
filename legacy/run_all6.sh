#!/usr/bin/env bash
# Generate shots 1 and 3 too (normally spliced / kept as source) for an existing run, then assemble all 6 generated.
#   JOB=regen_s42 OUT=outputs/full6_seed42 SEED=42 bash run_all6.sh
set -u
R=/research/cbim/vast/hn315/char_swap; cd "$R"; PY="$R/.venv/bin/python"
DETAILS="Female, slim build, narrow shoulders, slender waist, oval face, fair pale skin, dark brown eyes, small nose, soft pink lips, long straight wine-red hair with blunt bangs, small stud earrings, black long-sleeve off-the-shoulder crop top, cream pleated midi skirt"
$PY -u bfs_full.py --job "$JOB" --out "outputs/_${JOB}_auto13" --only 1,3 --keep "" --lead 9 --width 864 --height 480 \
  --comfy-urls http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190 \
  --protect "man" --protect-only --realism 0.8 --ref-color reinhard_lab:1.0 --seed "$SEED" \
  --ref "$R/phuc_cutout.png" --details "$DETAILS"
SHOTS=""
for s in 1 2 3 4 5 6; do d=outputs/$JOB/shot0$s; ls $d/save_*.mp4 >/dev/null 2>&1 && SHOTS="$SHOTS --shot $s=$d" || echo "  shot $s: no result"; done
$PY -u assemble_bfs.py --out "$OUT" --raw --letterbox $SHOTS
rm -rf "outputs/_${JOB}_auto13"
echo "[assembled] $OUT"
