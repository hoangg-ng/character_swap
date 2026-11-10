#!/usr/bin/env bash
# One-time setup: ComfyUI (pinned to the qualified H3 Relay commit) + H3 Relay nodes
# + all model weights needed by akatz-ai/MiniMax-H3-Character-Swap-LoRA.
#
# Usage:   HF_TOKEN=hf_xxx bash setup.sh            # everything
#          SKIP_MODELS=1 bash setup.sh              # code/env only
# NOTE: MiniMax H3 weights are under the "MiniMax H3 Community License". Accept it on
#       https://huggingface.co/MiniMaxAI/MiniMax-H3 with the account owning HF_TOKEN.
# Disk: ~47 GB of weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COMFY="$ROOT/ComfyUI"
VENV="$ROOT/.venv"
COMFY_COMMIT=79be670e2d9be63e238785af307369d2b9039ed1   # qualified by H3 Relay README
LORA_SHA=8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f

# ---------- 1. Python env ----------
if [ ! -d "$VENV" ]; then
  uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---------- 2. ComfyUI @ pinned commit ----------
if [ ! -d "$COMFY/.git" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
fi
git -C "$COMFY" fetch --quiet origin
git -C "$COMFY" checkout --quiet "$COMFY_COMMIT"

uv pip install "torch==2.11.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r "$COMFY/requirements.txt"
uv pip install -U "huggingface_hub[cli]" hf_transfer requests

# ---------- 3. H3 Relay custom nodes ----------
if [ ! -d "$COMFY/custom_nodes/h3-relay/.git" ]; then
  git clone https://github.com/akatz-ai/h3-relay.git "$COMFY/custom_nodes/h3-relay"
fi
if [ -f "$COMFY/custom_nodes/h3-relay/requirements.txt" ]; then
  uv pip install -r "$COMFY/custom_nodes/h3-relay/requirements.txt"
fi

# ---------- 4. Model weights ----------
if [ "${SKIP_MODELS:-0}" = "1" ]; then echo "SKIP_MODELS=1 -> done (no weights)."; exit 0; fi
export HF_HUB_ENABLE_HF_TRANSFER=1
# base models live on /common (checkpoint storage, outside the /research quota); LoRAs stay in char_swap/models
M="$ROOT/models"
B="${MODELS_BASE:-/common/users/hn315/checkpoints/char_swap_models}"
mkdir -p "$M/loras" "$B"/{diffusion_models,text_encoders,vae,checkpoints,model_patches}
cat > "$COMFY/extra_model_paths.yaml" <<EOF
char_swap_base:
  base_path: $B
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  checkpoints: checkpoints
  model_patches: model_patches
char_swap_loras:
  base_path: $M
  loras: loras
EOF

# hf download keeps the repo sub-path, so download into models/ directly.
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir "$B"

hf download Comfy-Org/sam3.1 checkpoints/sam3.1_multiplex_fp16.safetensors --local-dir "$B"

hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors --local-dir "$M/loras"

hf download akatz-ai/MiniMax-H3-Character-Swap-LoRA \
  h3_character_swap_v1.5.safetensors --local-dir "$M/loras"

echo "$LORA_SHA  $M/loras/h3_character_swap_v1.5.safetensors" | sha256sum -c -

echo
echo "Setup complete. Start the server with:  bash start_comfy.sh"
