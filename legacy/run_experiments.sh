#!/usr/bin/env bash
# Experiments on the full clip with the two new add-ons (baseline = outputs/final_hybrid_manlock):
#  E1 baseline + FaceRefine                           -> outputs/exp_E1_facerefine
#  E2 baseline + FaceRefine with Realism LoRA 0.8     -> outputs/exp_E2_facerefine_realism
#  E3 BFS man-locked with Realism LoRA 0.8 (regen)    -> outputs/exp_E3_bfs_realism
#  E4 E3 + FaceRefine with Realism LoRA 0.8           -> outputs/exp_E4_bfs_realism_facerefine
R=/research/cbim/vast/hn315/char_swap; cd "$R"; PY="$R/.venv/bin/python"
BASE=outputs/final_hybrid_manlock/hug_scene_swapped_720p.mp4
$PY -u face_refine.py --clip $BASE --out outputs/exp_E1_facerefine;                    echo "[E1] done"
$PY -u face_refine.py --clip $BASE --out outputs/exp_E2_facerefine_realism --realism 0.8; echo "[E2] done"
JOB=bfs_manlock480_realism OUT=outputs/exp_E3_bfs_realism REALISM=0.8 bash run_manlock.sh; echo "[E3] done"
$PY -u face_refine.py --clip outputs/exp_E3_bfs_realism/hug_scene_swapped_720p.mp4 \
  --out outputs/exp_E4_bfs_realism_facerefine --realism 0.8;                            echo "[E4] done"
echo "[ALL] DONE"
