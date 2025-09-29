GSM8K Remote NNsight — OpenAI Chat wrapper over NNsight

Overview
- Minimal OpenAI-compatible chat server backed by NNsight `LanguageModel`.
- Supports configuring “interventions” and writing captured tensors to disk.
- Demo driver deploys server, runs a few GSM8K samples, and syncs activations + trajectories locally.

Quick Run (local)
- `python -m examples.gsm8k_remote_nnsight.server.server --model willcb/Qwen3-0.6B --device-map auto --host 0.0.0.0 --port 8000`

Endpoints
- `GET /health` — Simple readiness info.
- `GET /v1/models` — OpenAI-compatible models list.
- `POST /v1/chat/completions` — OpenAI-compatible chat (non-streaming).
- `POST /v1/interventions` — Configure activation capture (layers, hook points, storage).
- `POST /v1/model` — Hot-reload model and device map.

Demo Driver (remote deploy + collect)
- Provision/reuse GPU and deploy as a detached job:
  - `python examples/gsm8k_remote_nnsight/deploy_and_collect.py --samples 3`
  - The script configures interventions, runs a few GSM8K problems, and downloads each request’s activation folder to `examples/gsm8k_remote_nnsight/results/...` alongside trajectories.

Troubleshooting
- See `examples/gsm8k_remote_nnsight/HANDOFF_DEBUG.md` for a focused debug playbook (tmux, logs, proxy mapping, module path checks).

Known Caveats
- Streaming (`stream: true`) is not implemented; the demo uses non-streaming calls.
- Tool/function calls are not supported in this demo.
