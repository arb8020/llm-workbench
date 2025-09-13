Nano NDIF — OpenAI Chat wrapper over NNsight

Overview
- Minimal OpenAI-compatible chat server backed by NNsight `LanguageModel`.
- Supports configuring “interventions” via POST to persistently enable activation capture.
- Writes captured tensors to disk and returns a small `ndif` breadcrumb in responses.

Quick Run (local)
- `python -m nano_ndif.server --model willcb/Qwen3-0.6B --device-map auto --host 0.0.0.0 --port 8002`
- Env fallbacks still supported: `NANO_NDIF_MODEL`, `NANO_NDIF_DEVICE_MAP`, `NANO_NDIF_HOST`, `NANO_NDIF_PORT`

Endpoints
- `GET /health` — Simple readiness info.
- `GET /v1/models` — OpenAI-compatible models list.
- `POST /v1/chat/completions` — OpenAI-compatible chat (non-streaming).
- `POST /v1/interventions` — Configure activation capture (layers, hook points, storage).
- `POST /v1/model` — Hot-reload model and device map `{ "model": "...", "device_map": "auto" }`.

Interventions API
- POST `/v1/interventions` body fields:
  - `enabled` (bool)
  - `layers` (list[int]) — e.g., `[8, 12, 16]`
  - `hook_points` (list[str]) — supported: `input_layernorm.output`, `post_attention_layernorm.output`
  - `mode` (`trace` | `generate`) — capture on forward over prompt, or during generation
  - `save_dir` (str) — base directory to write captures (default `./activations`)
  - `per_request_subdir` (bool) — create a run subdir per request
  - `sample_hidden` (int|null) — if set, slice last hidden dim to first N features
  - `save_format` (`pt` | `npy`)

Behavior Notes
- When `enabled`, requests cause tensors to be captured and written under `save_dir`.
- Responses include `ndif` with `request_id`, `artifacts` (file paths, shapes), and `index` (metadata file).
- Prompts use HF chat templates where available; falls back to a simple role-tagged format.

Remote Deploy (RunPod via broker + bifrost)
- Provision/reuse an instance and deploy as a detached job:
  - Ensure `.env` contains `RUNPOD_API_KEY`.
  - `python scripts/deploy_nano_ndif.py --name nano-ndif-qwen3 --gpu-type A10 --manufacturer nvidia --port 8002 --model willcb/Qwen3-0.6B --device-map auto --skip-bootstrap`
  - Or pass `--runpod-api-key ...` to avoid editing `.env`.
  - Output prints the proxy URL (e.g., `https://<pod>-8002.proxy.runpod.net`).

Smoke Test
- Use `scripts/smoke_nano_ndif.py` to exercise endpoints:
  - `python scripts/smoke_nano_ndif.py --base-url https://<pod>-8002.proxy.runpod.net`
  - Steps: health, models, chat (baseline), configure interventions, chat (captures), and asserts on `ndif`.

Example Requests
- Configure capture:
  - POST `/v1/interventions`
    - {"enabled": true, "layers": [8,12,16], "hook_points": ["input_layernorm.output","post_attention_layernorm.output"], "mode": "trace", "save_dir": "./activations", "sample_hidden": 64}
- Chat:
  - POST `/v1/chat/completions`
    - {"model": "willcb/Qwen3-0.6B", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 16, "temperature": 0.1}

Known Caveats
- Some NNsight versions do not expose a `tracer.generator`; the server falls back to HF `AutoModelForCausalLM.generate` for robustness.
- Health may report device as `"meta"` depending on NNsight internals; generation still functions.

Roadmap / TODOs
- Streaming (`stream: true`) compatible with OpenAI SSE.
- Tool/function call support and output parsing.
- Expanded hook-points and model-specific capture presets.
- Optional artifact serving endpoint to fetch activation files over HTTP.
