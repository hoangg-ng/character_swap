#!/usr/bin/env bash
# Download only the weights (no code) with curl (resumable).
# Base models go to MODELS_BASE (checkpoint storage on /common, outside the /research quota);
# LoRAs go to char_swap/models/loras. ComfyUI/extra_model_paths.yaml points to both.
# Usage: HF_TOKEN=hf_xxx bash download_models.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
M="$ROOT/models"                                                                 # LoRAs only
MODELS_BASE="${MODELS_BASE:-/common/users/hn315/checkpoints/char_swap_models}"   # base models
HF=https://huggingface.co
AUTH=(-H "Authorization: Bearer ${HF_TOKEN:?set HF_TOKEN}")

get() {  # repo  repo_path  local_rel_path
  local dst; case "$3" in loras/*) dst="$M/$3";; *) dst="$MODELS_BASE/$3";; esac; mkdir -p "$(dirname "$dst")"
  echo "[$(date +%T)] $1/$2 -> $3"
  for i in 1 2 3 4 5; do
    curl -fL --retry 5 -C - "${AUTH[@]}" -o "$dst" "$HF/$1/resolve/main/$2" && return 0
    echo "  retry $i"; sleep 10
  done
  echo "FAILED $1/$2"; return 1
}

rc=0
get akatz-ai/MiniMax-H3-Character-Swap-LoRA h3_character_swap_v1.5.safetensors loras/h3_character_swap_v1.5.safetensors || rc=1
get lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors loras/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors || rc=1
get Comfy-Org/sam3.1 checkpoints/sam3.1_multiplex_fp16.safetensors checkpoints/sam3.1_multiplex_fp16.safetensors || rc=1
for f in vae/minimax_h3_audio_vae_fp32.safetensors vae/minimax_h3_video_vae_fp16.safetensors \
         text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
         diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors; do
  get Comfy-Org/MiniMax-H3 "$f" "$f" || rc=1
done

echo "8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f  $M/loras/h3_character_swap_v1.5.safetensors" | sha256sum -c - || rc=1
du -sh "$M" "$MODELS_BASE"
echo "exit=$rc"
exit $rc
