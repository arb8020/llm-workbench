Emotion Prefix GSM8K — Rollouts + NNsight

Overview
- Evaluates GSM8K with emotional prompt prefixes using rollouts, backed by a copied NNsight OpenAI-compatible server.
- Captures hidden activations on the remote GPU per request; does not download by default.
- Staged, resumable pipeline with a YAML config saved in results for reproducibility.

Core Assumptions
- One GPU/server at a time during development (see deploy.md and prefer reusing the same instance).
- Non-streaming chat completions; rollouts uses `provider="vllm"` talking to the server’s `/v1/chat/completions`.
- Emotional variants prefix the first user message (copied from mats_neel’s basic_prompt_variation_gsm8k).
- Per-request activation overrides via the `ndif` field in the chat payload; the server allows unknown fields and records an `ndif` breadcrumb.

Quick Start
- Deploy and run generate stage (detached tmux on remote):
  - `python examples/emotion_prefix_gsm8k/deploy_and_evaluate.py --name nano-emopfx --samples 8 --variants control,frustration,impatience,anxiety,collaborative,patience,calm --stages deploy configure generate`
- Monitor progress (tail remote logs):
  - `python examples/emotion_prefix_gsm8k/deploy_and_evaluate.py --stages monitor`
- Analyze results:
  - `python examples/emotion_prefix_gsm8k/deploy_and_evaluate.py --stages analyze`

Stages (resumable)
- `deploy`: provision GPU, push code (uv extra: examples_emotion_prefix_gsm8k), start server in tmux.
- `configure`: query `/v1/model/structure`, enable capture (layers=all, two hook points), and record config.
- `generate`: run rollouts over selected GSM8K samples × variants. For each request, set `ndif` overrides per-sample to write activations to `~/.bifrost/workspace/examples/emotion_prefix_gsm8k/results/<exp>/activations/<variant>/<sample_id>/<request_id>/`.
- `analyze`: compute accuracy/format/efficiency and summarize.
- `download-activations` (optional): pull remote activation dirs locally. Off by default.
- `monitor`: convenience tail of remote `nnsight_server.log` and job logs.

Results Layout
- `examples/emotion_prefix_gsm8k/results/<exp>/`
  - `experiment.yaml` and `experiment_config.json`
  - `report.json`
  - `<variant>/<sample_id>/{trajectory.jsonl, agent_state.json, sample.json}`
  - `activations_manifest.json` (maps `<variant>/<sample_id>` → `{save_dir, request_id}`)

Config Hygiene
- Records: run info, model, server (including `num_layers` and access info), dataset, generation params, variants, interventions (layers=all, hook points), and pipeline stages.

TODOs
- Server: add SSE streaming to `/v1/chat/completions`.
- Storage: helpers to push/pull activation trees to S3/blob storage; support URI manifests.
- Analysis pipeline: consume activation dirs (local or URI) and run linear probes/SAE/other analyses; emit artifacts next to results.
- Progress UI: add a proper progress monitor (tqdm or equivalent) that reads from logs/metadata, keeping core stages detached.

Hardest Parts (notes)
- Ensuring rollouts can pass per-request overrides cleanly: we updated `Endpoint` to accept `extra_params` and the NNsight server to allow unknown fields.
- Mapping request → activation path without extra round-trips: we rely on the server’s `completion.id = chatcmpl-{request_id}` and a deterministic `save_dir` override.
- Keeping remote-only activation storage hygienic across runs: the pipeline records breadcrumbs and a manifest to enable later batch downloads or cloud sync.

