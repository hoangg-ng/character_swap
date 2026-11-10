#!/usr/bin/env bash
# Regenerate every swapped shot of hug_scene with larryvrh Turbo LoRA at 4 steps.
# Lane A (GPU1/8188): waits for the running shot-4 job, then shot 2.
# Lane B (GPU2/8189): shots 5, 6, 1.
R=/research/cbim/vast/hn315/char_swap
cd "$R"
PY="$R/.venv/bin/python -u $R/swap_shots.py --video $R/hug_scene.mp4 --resolution 720 --turbo larryvrh --steps 4 --guide-gpu 0"
MID_DESC="a young Asian woman with long straight wine-red hair and blunt bangs, wearing a black long-sleeve off-the-shoulder crop top and a cream pleated midi skirt"
FACE_DESC="a young Asian woman with long straight wine-red hair and blunt bangs"
EMO="crying, tearful glistening eyes, trembling lips, raised inner eyebrows"

laneA() {
  while pgrep -f "[h]ug_phuc_720_turbo4_s4" >/dev/null; do sleep 20; done
  echo "[A] shot 2"
  $PY --character $R/phuc_crop.jpg --keep 1,3,4,5,6 --job hug_t4_s2 --comfy-urls http://127.0.0.1:8188 \
      --guide depth-mesh-cn --cn-strength 1.0 --ref-video composite --composite-ref $R/phuc.jpg \
      --composite-detail 1.0 --composite-relight 0.9 --composite-skin 1.0 --occluder "man" \
      --subject "woman with long black hair in a white jacket" --character-desc "$MID_DESC" --emotion "$EMO"
  echo "[A] shot 2 exit=$?"
}

laneB() {
  echo "[B] shot 5"
  $PY --character $R/phuc_crop.jpg --keep 1,2,3,4,6 --job hug_t4_s5 --comfy-urls http://127.0.0.1:8189 \
      --occluder "man" --subject "woman with long black hair in a white jacket" --character-desc "$MID_DESC"
  echo "[B] shot 5 exit=$?"
  echo "[B] shot 6"
  $PY --character $R/phuc_face.jpg --keep 1,2,3,4,5 --job hug_t4_s6 --comfy-urls http://127.0.0.1:8189 \
      --mask-expand 24 --subject "woman with long black hair" --character-desc "$FACE_DESC"
  echo "[B] shot 6 exit=$?"
  echo "[B] shot 1"
  $PY --character $R/phuc_crop.jpg --keep 2,3,4,5,6 --job hug_t4_s1 --comfy-urls http://127.0.0.1:8189 \
      --subject "woman with long black hair in a white jacket" --character-desc "$MID_DESC"
  echo "[B] shot 1 exit=$?"
}

laneA > "$R/run_t4_laneA.log" 2>&1 &
laneB > "$R/run_t4_laneB.log" 2>&1 &
wait
echo "ALL DONE"
