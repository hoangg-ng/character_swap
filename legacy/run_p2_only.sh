R=/research/cbim/vast/hn315/char_swap
cd "$R"
PY="$R/.venv/bin/python"
DETAILS="Female, slim build, narrow shoulders, slender waist, oval face, fair pale skin, dark brown eyes, small nose, soft pink lips, long straight wine-red hair with blunt bangs, small stud earrings, black long-sleeve off-the-shoulder crop top, cream pleated midi skirt"
$PY -u bfs_full.py --job bfs_p2_480 --out outputs/_bfs_p2_480_auto --only 1,2,4,5,6 --lead 9 --width 864 --height 480 \
  --comfy-urls http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190 \
  --target-ref-text "woman with long black hair in a white jacket" --ref "$R/phuc_cutout.png" --details "$DETAILS"
echo "[3] two-image batch finished (exit $?)"
SHOTS=""
for s in 1 2 4 5 6; do
  d=outputs/bfs_p2_480/shot0$s
  ls $d/save_*.mp4 >/dev/null 2>&1 && SHOTS="$SHOTS --shot $s=$d" || echo "  shot $s: no result"
done
$PY -u assemble_bfs.py --out outputs/final_bfs_p2_480 $SHOTS
rm -rf outputs/_bfs_p2_480_auto outputs/_bfs_sam480_auto
echo "[4] ALL DONE"
