# Emotion Prefix GSM8K — Handoff Notes (September 2025)

## TL;DR
- Pipeline launches fine, but generation OOMs on the 16 GB RTX A4000 when capturing all layers with `max_tokens=2048`.
- Server still loads *two* copies of Qwen3-0.6B (NNsight + raw HF model). Refactor to reuse the existing model; we built a prototype (`server_llm_generate.py`) but it still needs integration.
- New logging is in place: look for `[memory] generate: before=… after=… peak=…` in `~/nnsight_server.log`. 500 responses now expose the underlying CUDA error message.
- Next steps: reduce memory settings or finish the single-model refactor; consider capturing fewer hook points or enabling `sample_hidden`.

## Current State
- Repo: `main` @ **1c1ec74** (`Define server logger for error logging`).
- Recent commits:
  - `43d3ea0` – Wrap format/efficiency rewards as factories.
  - `74ebc95` – Log CUDA memory around model load/generate.
  - `1f7cd64` – Log failures from `generate` with CUDA stats.
  - `2c5d07c` – Added experimental `server_llm_generate.py` (single-model attempt).
- Remote instance: `157.157.221.29` (RunPod RTX A4000, 16 GB).
- Server tmux session: `nnsight-server` (port 8000). Experimental server tested in session `llmgen-test` (port 8100).
- `deploy_and_evaluate.py --stages deploy configure generate` loops because every `/v1/chat/completions` call after the first hits CUDA OOM.

## Key Issues & Diagnostics
1. **Duplicate model load**
   - `server/server.py` loads NNsight (`llm = LanguageModel`) *and* a separate `AutoModelForCausalLM` for generation. That doubles memory (≈1.2 GB per copy) and doubles KV caches.
   - Experimental `server_llm_generate.py` tries to remove the duplicate, but `llm.model` is an NNsight Envoy (`Qwen3Model` without `.generate`). We need to call the inner HF module instead (e.g., `llm.model._module.generate(...)`).
2. **Peak memory summary** (from logs):
   - `[memory] generate failed: allocated=15096.5MiB reserved=15790.0MiB peak=15158.3MiB max_new_tokens=2048`
   - Peaks >15.7 GB ⇒ the 16 GB card has no headroom.
3. **OOM root causes**
   - `max_tokens=2048` + 28 layers × 2 hook points ⇒ huge activation footprint.
   - Duplicate model weights & KV caches.
   - All-layer capture writes tens of MB per request before freeing.
4. **Retries**
   - Deploy script uses exponential backoff (4s → 8s → …) but fails every time; job never completes.

## Changes Made This Week
- Reward factories updated to avoid passing sample dicts into trajectory-based rewards.
- Added detailed CUDA logging and surfaced server-side errors back to the client (so 500s now include the OOM message).
- Captured logging prints to help spot memory spikes.
- Added experimental server variant to test running with one model copy (not fully working yet).
- `deploy_and_evaluate.py` commits include CLI config support, memory logging, reward fixes.

## What Works
- Deployment (`deploy` stage) via tmux + `bifrost` is reliable.
- Health check, `/v1/model/structure`, `/v1/interventions` respond normally.
- First request often succeeds; long sequences + capture eventually OOM.
- Logging is informative; investigating is easier now.

## What’s Broken
- Continuous generation fails on 16 GB due to double load + heavy capture.
- Experimental server’s generation path currently raises `AttributeError: Qwen3Model… has no attribute generate` when we attempt `llm.model.generate`.
- `deploy_and_evaluate.py --stages generate` expects `deploy_info['ready_url']` (only set during deploy); can’t hit the scratch server without manual override.

## Immediate Next Steps
1. **Single-model refactor**
   - In `server_llm_generate.py`: call the inner HF module from `llm` (likely `llm.model._module` or `llm.lm_head`). Confirm with tests; handle text decoding manually.
   - Replace current server or update `start_server.sh` once verified.
2. **Reduce memory usage for testing**
   - Temporarily lower `max_tokens` to ≤512, or capture fewer layers (e.g., `[8,12,16]`), or set `mode: trace` to avoid activation capture during debugging.
   - Enable `sample_hidden` to reduce hook tensor size if needed.
3. **Re-run deploy/eval** after adjustments; inspect `~/nnsight_server.log` for new peak values.
4. **Optional**: choose a larger GPU (24 GB+) if you must capture all layers with 2048 tokens.

## Nice-to-Have Improvements
- Add CLI flag to `deploy_and_evaluate.py` to point at custom server module/port.
- Automate GPU cleanup/reset before redeploy (e.g., `nvidia-smi --gpu-reset` only when server tmux is stopped).
- Persist scratch server logs (`tmux capture-pane > ~/tmux-logs/…`) for easier postmortem.
- Document the two-server configuration vs. single-server variant.

## Known Files & Paths
- Server code: `examples/emotion_prefix_gsm8k/server/server.py`
- Experimental single-model server: `examples/emotion_prefix_gsm8k/server/server_llm_generate.py`
- Logging: `~/nnsight_server.log` on the GPU (`bifrost exec … tail -f ~/nnsight_server.log`).
- Results: `examples/emotion_prefix_gsm8k/results/<exp>` both locally and under `~/.bifrost/workspace/...`.

## Contacts / Context
- Current run uses `willcb/Qwen3-0.6B` (0.6 B parameters). Context length in tests ~3 k tokens; there’s no context window enforcement yet.
- Expectation: once duplicate weights are removed and capture scope is tuned, the 16 GB card should handle ~512–1024 token runs comfortably.

## Checklist for Handoff
- [ ] Decide whether to lower `max_tokens` or move to a bigger GPU.
- [ ] Finish integrating `server_llm_generate` (ensure it calls the main HF model correctly).
- [ ] Update `start_server.sh` to launch the new module once verified.
- [ ] Rerun `deploy_and_evaluate.py` end-to-end; confirm accuracy metrics make sense.
- [ ] Document final config in `config.yaml` (e.g., recommended `max_tokens`, `layers`).
- [ ] Terminate unused GPUs if you’re done testing (`broker instances terminate …`).

Feel free to reach out if you need help navigating the NNsight wrapper or want to pair on the single-model refactor.
