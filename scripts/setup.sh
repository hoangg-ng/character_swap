#!/usr/bin/env bash
# One-time setup for the character-swap pipeline (run_swap.sh):
#   ComfyUI (pinned commit) + H3 Relay nodes + Python deps + every model weight the pipeline uses.
#
# Usage:   HF_TOKEN=hf_xxx bash setup.sh            # everything
#          SKIP_MODELS=1 bash setup.sh              # code/env only (weights: bash scripts/download_models.sh)
# NOTE: MiniMax H3 weights are under the "MiniMax H3 Community License". Accept it on
#       https://huggingface.co/MiniMaxAI/MiniMax-H3 with the account owning HF_TOKEN.
# Disk: ~60 GB of base models in MODELS_BASE (default /common/... checkpoint storage) + ~2.5 GB of LoRAs in models/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY="$ROOT/ComfyUI"
VENV="$ROOT/.venv"
COMFY_COMMIT=79be670e2d9be63e238785af307369d2b9039ed1   # H3 nodes + Qwen-Image-2.1 nodes

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

# ---------- 3. H3 Relay custom nodes (H3RelayApplyEditMask: the SAM lock on protected people) ----------
if [ ! -d "$COMFY/custom_nodes/h3-relay/.git" ]; then
  git clone https://github.com/akatz-ai/h3-relay.git "$COMFY/custom_nodes/h3-relay"
fi
if [ -f "$COMFY/custom_nodes/h3-relay/requirements.txt" ]; then
  uv pip install -r "$COMFY/custom_nodes/h3-relay/requirements.txt"
fi

# ---------- 4. Pipeline Python deps ----------
# ultralytics: YOLO11 person/head presence + tone matching; insightface: target identity check;
# scenedetect: CUTS=auto. Keep ONLY onnxruntime-gpu: the CPU package would shadow it.
uv pip install ultralytics insightface "scenedetect>=0.7"
uv pip uninstall onnxruntime onnxruntime-gpu 2>/dev/null || true
uv pip install onnxruntime-gpu

# ---------- 5. Model weights ----------
if [ "${SKIP_MODELS:-0}" = "1" ]; then echo "SKIP_MODELS=1 -> done (no weights)."; exit 0; fi
export HF_HUB_ENABLE_HF_TRANSFER=1
# base models live on /common (checkpoint storage, outside the /research quota); LoRAs stay in char_swap/models
M="$ROOT/models"
B="${MODELS_BASE:-/common/users/hn315/checkpoints/char_swap_models}"
mkdir -p "$M/loras" "$B"/{diffusion_models,text_encoders,vae,checkpoints}
cat > "$COMFY/extra_model_paths.yaml" <<EOF
char_swap_base:
  base_path: $B
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  checkpoints: checkpoints
char_swap_loras:
  base_path: $M
  loras: loras
EOF

MODELS_BASE="$B" bash "$ROOT/scripts/download_models.sh"

echo
echo "Setup complete. Start one ComfyUI server per GPU:  GPU=1 PORT=8188 bash scripts/start_comfy.sh  (and 8189, 8190)"
