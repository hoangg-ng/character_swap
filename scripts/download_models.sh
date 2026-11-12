#!/usr/bin/env bash
# Download every model weight the pipeline uses (resumable curl; files already present are skipped).
# Base models go to MODELS_BASE (checkpoint storage on /common, outside the /research quota);
# LoRAs go to char_swap/models/loras. ComfyUI/extra_model_paths.yaml (written by setup.sh) points to both.
# Usage: HF_TOKEN=hf_xxx bash scripts/download_models.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$ROOT/models"                                                                 # LoRAs only
MODELS_BASE="${MODELS_BASE:-/common/users/hn315/checkpoints/char_swap_models}"   # base models
HF=https://huggingface.co
AUTH=(-H "Authorization: Bearer ${HF_TOKEN:?set HF_TOKEN}")

get() {  # repo  repo_path  local_rel_path   (loras/* -> project, everything else -> MODELS_BASE)
  local dst; case "$3" in loras/*) dst="$M/$3";; *) dst="$MODELS_BASE/$3";; esac; mkdir -p "$(dirname "$dst")"
  if [ -s "$dst" ]; then echo "[skip] $3"; return 0; fi
  echo "[$(date +%T)] $1/$2 -> $3"
  for i in 1 2 3 4 5; do
    curl -fL --retry 5 -C - "${AUTH[@]}" -o "$dst.part" "$HF/$1/resolve/main/$2" && mv "$dst.part" "$dst" && return 0
    echo "  retry $i"; sleep 10
  done
  echo "FAILED $1/$2"; return 1
}

rc=0
# --- MiniMax H3 (video generation): model, text encoder, VAEs
for f in diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
         text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
         vae/minimax_h3_video_vae_fp16.safetensors \
         vae/minimax_h3_audio_vae_fp32.safetensors; do
  get Comfy-Org/MiniMax-H3 "$f" "$f" || rc=1
done
# --- SAM 3.1 (locks protected people, face mask for FACE_ONLY, cut-outs)
get Comfy-Org/sam3.1 checkpoints/sam3.1_multiplex_fp16.safetensors checkpoints/sam3.1_multiplex_fp16.safetensors || rc=1
# --- Qwen-Image-2.1 (character sheet, make_char_sheet.py)
for f in diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
         text_encoders/qwen3vl_8b_int8_convrot.safetensors \
         vae/qwen_image_2.1_vae_bf16.safetensors; do
  get Comfy-Org/Qwen-Image-2.1 "$f" "$f" || rc=1
done

# --- LoRAs
get Alissonerdx/Minimax-H3-ComfyUI loras/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors \
    loras/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors || rc=1                       # BFS v1.1 character swap
get drbaph/MiniMax-H3-Turbo-Lora-ComfyUI experimental/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors \
    loras/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors || rc=1                        # DMD ref2va 8-step turbo
get fal/MiniMax-H3-Realism-People-LoRA h3-realism-people-t2v-i2v-r2v.safetensors \
    loras/h3-realism-people-t2v-i2v-r2v.safetensors || rc=1                                   # Realism People (r34l1sm)
get lightx2v/Qwen-Image-2.1-Lightning pytorch_lora_weights.safetensors \
    loras/qwen_image_2.1_lightning_4steps_lightx2v.safetensors || rc=1                        # Qwen-Image-2.1 4-step

# --- YOLO11 person / pose (presence check, tone matching); ultralytics fetches them from its GitHub releases
YD="$ROOT/ComfyUI/models/ultralytics/person"; mkdir -p "$YD"
( cd "$YD" && "$ROOT/.venv/bin/python" -c "from ultralytics import YOLO; YOLO('yolo11m.pt'); YOLO('yolo11m-pose.pt')" ) || rc=1
get LBH-123-AI/Minimax_h3_latent_Upscaler minimax_h3_latent_upscaler_2d_conv_v1/minimax_h3_latent_upscaler_2d_conv_v1_fp16.safetensors \
    latent_upscale_models/minimax_h3_latent_upscaler_2d_conv_v1_fp16.safetensors || rc=1          # UPSCALE (2D v1)
# FaceRefine's face detector (face_yolov8m.pt) is fetched by the node on first use
# insightface buffalo_l (target identity check) is downloaded automatically on first use
# into ComfyUI/models/insightface/models/buffalo_l.

du -sh "$M" "$MODELS_BASE"
echo "exit=$rc"
exit $rc
