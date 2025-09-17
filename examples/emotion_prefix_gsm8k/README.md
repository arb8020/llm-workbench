Emotion Prefix GSM8K — Rollouts + NNsight

This example evaluates GSM8K with emotional prompt prefixes against the copied NNsight OpenAI-compatible server. It captures hidden activations remotely while keeping a reproducible config for each run.

How To Run
- Configure
  - Edit `examples/emotion_prefix_gsm8k/config.yaml` to set instance name, model, layer capture, samples, and output directory (optional `out`).
  - Recommended for all-layer capture: `max_tokens: 128`, `mode: generate`.
- Launch (detached, config-driven)
  - `python examples/emotion_prefix_gsm8k/emotion_rollouts.py --detach`
  - The script reads the YAML, provisions or reuses the GPU, deploys the NNsight server in tmux, uploads a remote run config, and starts generation in another tmux session.
  - Once running it prints the tmux session name, remote log path, and local results directory (`examples/emotion_prefix_gsm8k/results/<exp_name>`).
- Monitor (logs + progress)
  - `python examples/emotion_prefix_gsm8k/monitor.py --ssh root@<ip>:<port> --exp <exp_name> --tail 60`
  - Shows tmux sessions, completed versus expected sample.json counts, and remote log tails.
- Analyze (summarize accuracy per variant)
  - `python examples/emotion_prefix_gsm8k/analyze.py examples/emotion_prefix_gsm8k/results/<exp_name>`
- Cleanup (optional)
  - Terminate instances when done: `broker instances terminate --yes <instance_id>`

Config YAML
- `name`, `port`, `min_vram`, `max_price`: broker provisioning parameters.
- `model`, `device_map`: vLLM server launch arguments. Defaults to `willcb/Qwen3-0.6B` and `auto`.
- `samples`, `seed`: GSM8K selection (test split).
- `variants`: emotional prefixes applied to the first user turn.
- `max_tokens`, `temperature`: rollout generation parameters.
- `layers`: `all` or a list such as `[8,12,16]` for targeted capture.
- `mode`: NNsight capture mode (`generate` for full forward pass, `trace` for faster metadata-only capture).
- `out`: optional explicit local results directory; otherwise a timestamped folder is created.

Run Pipeline (detached)
1. `deploy`: provision GPU, push repo (uv extra `examples_gsm8k_remote_nnsight`), launch server in tmux.
2. `configure`: query `/v1/model/structure`, enable interventions (layers, hook points), and record experiment metadata.
3. `generate`: run rollouts across selected GSM8K samples × variants; each request writes activations to `~/.bifrost/workspace/examples/emotion_prefix_gsm8k/results/<exp>/activations/...` using `ndif` overrides.
4. `analyze`: summarize accuracy/format/efficiency metrics and store a report.
5. `download-activations` (optional): fetch remote activation trees later.
6. `monitor`: convenience log tail via `monitor.py`.

Results Layout
- `examples/emotion_prefix_gsm8k/results/<exp>/`
  - `experiment.yaml` (local run metadata) and `experiment_config.json` (remote mirror).
  - `report.json` (overall metrics).
  - `<variant>/<sample_id>/{trajectory.jsonl, agent_state.json, sample.json}` per request.
  - `activations_manifest.json` mapping `<variant>/<sample_id>` → `{save_dir, request_id}` for remote tensors.

Additional Notes
- The non-detached path in `emotion_rollouts.py` is deprecated; prefer the config + `--detach` flow.
- `emotion_collect.py` remains for the older foreground HTTP demo but does not use rollouts.
- The server lives under `examples/gsm8k_remote_nnsight/server`; see `HANDOFF.md` for operational details, troubleshooting tips, and roadmap items.
- `deploy_and_evaluate.py` accepts `--config <path>` so you can keep multiple YAML presets (CLI flags still override the loaded values).

Recommended Settings
- Capturing all layers is heavy: keep `max_tokens` conservative (≤128) or switch to selective layers (e.g., `[8,12,16]`).
- Consider setting `sample_hidden` in the config/remote interventions if prompts become large.

Server Highlights
- `/v1/model/structure` returns `num_layers` and available hook points.
- `/v1/interventions` supports `layers: "all"` and records capture settings.
- `/v1/chat/completions` accepts `ndif` overrides and supports `stream: true` for SSE (single chunk + `[DONE]`).
