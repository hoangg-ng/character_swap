#!/usr/bin/env bash
# Workers for the GoldenAI web-app (python -m worker). Needs Redis and the ComfyUI servers (scripts/start_comfy.sh).
#
#   bash scripts/start_workers.sh            # both roles: render in the background, util in the foreground
#   ROLE=render bash scripts/start_workers.sh    # one role only (e.g. one tmux window per role)
#   ROLE=util   bash scripts/start_workers.sh
#
# render  jobs.character_swap: runs scripts/run_swap.sh. Run exactly ONE per cluster: one job drives all three
#         ComfyUI servers.
# util    jobs.character_swap_util: edit planner (Qwen3-VL) and character preparation, on PLANNER_GPU (default 0,
#         nvidia-smi numbering), so they answer while a render is running.
# Settings: worker/.env (see worker/.env.example) or the environment: REDIS_URL, STREAM_PREFIX, SWAP_CASES_DIR,
# SWAP_CHARS_DIR, PLANNER_GPU. Logs: .comfy_tmp/logs/worker_<role>.log when both roles run from one shell.
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
PY="${WORKER_PYTHON:-$R/.venv/bin/python}"     # the pipeline venv: the util role imports the planner and insightface
mkdir -p .comfy_tmp/logs

run_util() {
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${PLANNER_GPU:-0}" exec "$PY" -u -m worker --role util
}

case "${ROLE:-both}" in
  render) exec "$PY" -u -m worker --role render ;;
  util)   run_util ;;
  both)
    "$PY" -u -m worker --role render >> .comfy_tmp/logs/worker_render.log 2>&1 &
    RENDER_PID=$!
    trap 'kill $RENDER_PID 2>/dev/null || true' EXIT INT TERM
    echo "[workers] render pid $RENDER_PID (log .comfy_tmp/logs/worker_render.log); util in the foreground"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${PLANNER_GPU:-0}" "$PY" -u -m worker --role util
    ;;
  *) echo "ROLE must be render, util or both" >&2; exit 64 ;;
esac
