Nano NDIF — OpenAI chat wrapper over NNsight

Run
- `python -m nano_ndif.server`
- Env overrides: `NANO_NDIF_MODEL`, `NANO_NDIF_DEVICE_MAP`, `NANO_NDIF_HOST`, `NANO_NDIF_PORT`

Endpoints
- `GET /v1/models` — OpenAI-compatible models list
- `POST /v1/chat/completions` — OpenAI chat-completions request (non-streaming)
- `POST /v1/interventions` — Configure activation capture (layers, hook points, storage)

Interventions
- Payload fields:
  - `enabled` (bool)
  - `layers` (list[int])
  - `hook_points` (list[str]) — supported: `input_layernorm.output`, `post_attention_layernorm.output`
  - `mode` (`trace` or `generate`)
  - `save_dir` (str path)
  - `per_request_subdir` (bool)
  - `sample_hidden` (int|null) — slice last dim if large
  - `save_format` (`pt`|`npy`)

Behavior
- When enabled, the server captures tensors per request and writes them under `save_dir`.
- The chat response includes an `ndif` breadcrumb indicating where activations were saved.
- Default model is `willcb/Qwen3-0.6B` and chat prompts use HF chat templates when available.

