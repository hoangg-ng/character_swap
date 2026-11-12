#!/usr/bin/env bash
# Web UI for the pipeline (webui.py, Gradio). Needs the ComfyUI servers running (scripts/start_comfy.sh).
#   UI_HOST      127.0.0.1 (default, reach it through an SSH tunnel) or 0.0.0.0 (reachable on the network)
#   UI_PORT      7860
#   UI_USER / UI_PASSWORD   login; without UI_PASSWORD a random one is created once in .ui_password
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
mkdir -p .comfy_tmp/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${PLANNER_GPU:-0}   # edit planner on the free GPU (nvidia-smi 0)
exec "$R/.venv/bin/python" -u webui.py
