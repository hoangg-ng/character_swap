#!/usr/bin/env bash
# Launch the ComfyUI backend used by character_swap.py.
# Usage: GPU=1,2 PORT=8188 bash start_comfy.sh   (nvidia-smi numbering)
# Pick a 48 GB A6000 (the 20 GB A4500 is too small even with offloading).
set -euo pipefail
mkdir -p "$(cd "$(dirname "$0")" && pwd)/.comfy_tmp"
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # GPU ids == nvidia-smi indices
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd "$ROOT/ComfyUI"
PORT="${PORT:-8188}"
# separate db/temp per instance so several servers (one per GPU) can share this ComfyUI checkout
exec python main.py --listen 127.0.0.1 --port "$PORT" \
  --output-directory "$ROOT/outputs/comfy" \
  --temp-directory "$ROOT/.comfy_tmp/$PORT" \
  --database-url "sqlite:///$ROOT/.comfy_tmp/comfyui_$PORT.db" ${EXTRA_ARGS:-}
