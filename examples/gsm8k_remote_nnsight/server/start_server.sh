#!/usr/bin/env bash
# Robust server launcher for NNsight example. Runs in remote workspace under tmux.
set -euo pipefail

PORT=8000
HOST="0.0.0.0"
MODEL="willcb/Qwen3-0.6B"
DEVICE_MAP="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --device-map) DEVICE_MAP="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

echo "[start_server] Using PORT=$PORT HOST=$HOST MODEL=$MODEL DEVICE_MAP=$DEVICE_MAP"

LOG=~/nnsight_server.log
echo "[start_server] Logs → $LOG"

cd ~/.bifrost/workspace

# Ensure uv and deps
pip install -U uv >/dev/null 2>&1 || pip install -U uv
uv --version || true
uv sync --extra examples_gsm8k_remote_nnsight

# Stop any existing server
pkill -f examples.gsm8k_remote_nnsight.server || true
sleep 1

echo "[start_server] Launching server..."
set -x
uv run python -m examples.gsm8k_remote_nnsight.server.server \
  --host "$HOST" --port "$PORT" \
  --model "$MODEL" --device-map "$DEVICE_MAP" 2>&1 | tee -a "$LOG"
