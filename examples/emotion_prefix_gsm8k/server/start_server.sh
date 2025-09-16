#!/usr/bin/env bash
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

pip install -U uv >/dev/null 2>&1 || pip install -U uv
# Try to install via project extras; if not defined on remote, fall back to direct installs
uv sync --extra examples_emotion_prefix_gsm8k || uv sync --extra examples-emotion-prefix-gsm8k || true

# Fallback direct installs (best-effort)
python - <<'PY'
import subprocess, sys
pkgs = [
  'fastapi>=0.110.0', 'uvicorn>=0.23.0', 'pydantic>=2.0.0',
  'nnsight>=0.4', 'transformers>=4.40.0',
  'torch>=2.4.0,<=2.7.1',
  'accelerate>=0.20.0', 'datasets>=4.0.0', 'huggingface-hub>=0.34.4', 'numpy<2.0.0', 'requests>=2.28.0'
]
for p in pkgs:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])
        print('[deps] installed', p)
    except subprocess.CalledProcessError as e:
        print('[deps] failed', p, e)
PY

set +e
pkill -f 'python -m examples.emotion_prefix_gsm8k.server.server' >/dev/null 2>&1 || true
sleep 1
set -e

set -x
uv run python -m examples.emotion_prefix_gsm8k.server.server \
  --host "$HOST" --port "$PORT" \
  --model "$MODEL" --device-map "$DEVICE_MAP" 2>&1 | tee -a "$LOG"
