#!/usr/bin/env bash
# 2x2 comparison: baseline (man-locked BFS) | FaceRefine / Realism LoRA | Realism + FaceRefine, labelled, source audio
R=/research/cbim/vast/hn315/char_swap; cd "$R"; O=outputs/compare_4way; mkdir -p $O
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
L(){ echo "scale=960:540,drawtext=fontfile=$F:text='$1':x=16:y=16:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8"; }
ffmpeg -y -loglevel error \
  -i outputs/final_hybrid_manlock/hug_scene_swapped_720p.mp4 \
  -i outputs/exp_E1_facerefine/hug_scene_swapped_720p.mp4 \
  -i outputs/exp_E3_bfs_realism/hug_scene_swapped_720p.mp4 \
  -i outputs/exp_E4_bfs_realism_facerefine/hug_scene_swapped_720p.mp4 \
  -filter_complex "[0:v]$(L 'Original (man-locked BFS)')[a];[1:v]$(L 'FaceRefine')[b];[2:v]$(L 'Realism LoRA')[c];[3:v]$(L 'Realism + FaceRefine')[d];[a][b]hstack[t];[c][d]hstack[u];[t][u]vstack" \
  -map 0:a -c:v libx264 -crf 17 -pix_fmt yuv420p -c:a aac -shortest $O/compare_4way.mp4
cp outputs/final_hybrid_manlock/hug_scene_swapped_720p.mp4 $O/1_original.mp4
cp outputs/exp_E1_facerefine/hug_scene_swapped_720p.mp4 $O/2_facerefine.mp4
cp outputs/exp_E3_bfs_realism/hug_scene_swapped_720p.mp4 $O/3_realism.mp4
cp outputs/exp_E4_bfs_realism_facerefine/hug_scene_swapped_720p.mp4 $O/4_realism_facerefine.mp4
echo "[grid] $O/compare_4way.mp4"
