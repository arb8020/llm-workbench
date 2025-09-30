GSM8K Remote NNsight — OpenAI Chat wrapper over NNsight

## Status
✅ **Fully functional** - End-to-end tested on 2025-09-29

## Overview
- Minimal OpenAI-compatible chat server backed by NNsight `LanguageModel`.
- Supports configuring "interventions" and writing captured tensors to disk.
- Demo driver deploys server, runs a few GSM8K samples, and syncs activations + trajectories locally.

## Quick Start

### Local Server
```bash
python -m examples.gsm8k_remote_nnsight.server.server \
  --model willcb/Qwen3-0.6B \
  --device-map auto \
  --host 0.0.0.0 \
  --port 8000
```

### Remote Deploy + Data Collection
```bash
python examples/gsm8k_remote_nnsight/deploy_and_collect.py --samples 3
```

This script:
1. Provisions GPU instance (>=12GB VRAM, <=$0.40/hr)
2. Deploys code and installs dependencies
3. Starts NNsight server in tmux
4. Configures interventions for activation capture
5. Runs GSM8K samples
6. Downloads activations + trajectories to `examples/gsm8k_remote_nnsight/results/`

## API Endpoints
- `GET /health` — Server readiness check
- `GET /v1/models` — OpenAI-compatible models list
- `POST /v1/chat/completions` — Chat completions with optional activation capture
- `POST /v1/interventions` — Configure activation capture (layers, hook points, storage)
- `POST /v1/model` — Hot-reload model and device map

## Activation Capture

Captured tensors per request (when enabled):
- Format: PyTorch `.pt` files
- Location: Configurable save directory with per-request subdirectories
- Metadata: JSON file with tensor shapes, dtypes, file paths
- Hook points: `input_layernorm.output`, `post_attention_layernorm.output`

## Troubleshooting
- See `examples/gsm8k_remote_nnsight/HANDOFF_DEBUG.md` for debug playbook
- Check server logs: `tail -f ~/nnsight_server.log` on remote
- Verify tmux session: `tmux attach -t nnsight-server`

## Known Issues
- **Double GPU allocation**: In `mode="trace"` (default), server loads both NNsight-wrapped model and separate HF model, using ~2x expected VRAM (~5.5GB for 0.6B model instead of ~2.5GB). See `server/server.py:479-498`.
- **Streaming**: `stream: true` not fully implemented (single-chunk SSE only)
- **Tool/function calls**: Not supported in this demo

## Verified Features
- ✅ GPU provisioning and deployment
- ✅ Server startup and health checks
- ✅ Intervention configuration
- ✅ Chat completions
- ✅ Multi-layer activation capture (layers 8, 12, 16 tested)
- ✅ Hook point filtering (input_layernorm, post_attention_layernorm)
- ✅ Local download of activations and trajectories
- ✅ Metadata generation
