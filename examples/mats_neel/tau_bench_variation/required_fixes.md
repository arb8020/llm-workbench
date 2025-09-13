**Purpose**
- Document the minimal, required fixes to make Tau-Bench emotional user variants run reliably with our integration, and to explain the observed failures in recent logs.

**Current Symptoms**
- In `output2.txt`, tasks execute but conversations devolve into repeated “<think> …” monologue text on both assistant and user turns.
- No tool calls are made; reward is `0.0` for the task.
- Earlier runs showed a checkpoint write failure due to missing parent directories (now guarded) and, before that, context-window errors (now avoided by larger max context).

**What We Changed (Delta From Stock Tau-Bench)**
- Emotional user variant via monkey‑patch:
  - Patched `tau_bench.envs.user.load_user` to return subclasses of `LLMUserSimulationEnv` that only override `build_system_prompt` to append an emotional context suffix.
  - To pass Tau‑Bench’s enum validation, we always set `config.user_strategy = "llm"` and select the emotion via a process‑local `ContextVar` (no env vars).
- Endpoint routing:
  - Route both the agent and the LLM user simulation through a vLLM OpenAI‑compatible server.
  - Server was launched with `--enable-auto-tool-choice` and `--tool-call-parser hermes`.
- Observability and reliability hardening (wrapper only):
  - Write `variant_summary.json` with `status` and explicit `exit_reason`.
  - Preflight `/v1/models` readiness check and environment logging.
  - Guard checkpoint writes by auto‑creating parent directories.
  - Increase default `--max-model-len` to `32768` to avoid earlier token-limit failures.
  - Idempotent deploy (`--reuse`, `--gpu-id`, clean tmux restarts, optional `--reuse-running-server`).
  - Optional sanitizer: strip `<think>…</think>` leakage from model outputs at the client layer (non-invasive).

**Why It’s Failing Now (Root Cause Hypothesis)**
- Both the agent and the LLM user simulation are hitting the same endpoint that is globally configured with the `hermes` tool‑call parser. With a small model (`Qwen3-0.6B`), this can bias outputs toward “reasoning markup” (e.g., `<think>`) and away from clean tool-call JSON or plain replies.
- The user simulation’s responses then contain `<think>` monologue instead of normal user text, and those responses are fed back into the dialogue. The trajectory shows `<think>` text on both roles, consistent with conversation content being polluted at generation time, not a role swap in our code.
- Net effect: the agent never emits valid `tool_calls`; the dialogue loops on meta‑reasoning text and tasks fail.

**Required Fixes (Minimal, Robust)**
- Limit tool‑calling template influence to the agent only:
  - Option A (simplest): Remove `--tool-call-parser hermes` from the vLLM server command; keep `--enable-auto-tool-choice` enabled.
    - Command change in `launch_experiment.py` vLLM start:
      - Remove `--tool-call-parser hermes`.
  - Option B (isolation): Run two endpoints:
    - Agent endpoint: tool‑calling enabled (may keep hermes off for small models).
    - User endpoint: plain chat (no tool‑calling template). Point `LLMUserSimulationEnv` at this endpoint while keeping the agent on the tool‑calling endpoint.
    - Note: vLLM config is process‑wide; different parsing requires a second server/session.
- Keep the user strategy enum valid and emotional behavior minimal:
  - Always set `RunConfig.user_strategy = "llm"`.
  - Select the emotion inside our patched `load_user` (already implemented via `ContextVar`).
- Retain hardening:
  - Keep the checkpoint write guard (prevents `FileNotFoundError`).
  - Keep max context at `32768` unless memory is constrained.
  - Keep `variant_summary.json` for explicit exit reasons.
- Optional but recommended: Keep the output sanitizer to strip `<think>` if models still leak chain‑of‑thought; this reduces downstream parsing issues.

**Validation Plan**
- Run a small test (`--tasks 2`, variants: `control,frustration`) with server restarted (do not pass `--reuse-running-server` immediately after code changes).
- Expect to see:
  - Assistant turns emitting proper `tool_calls` entries.
  - No `<think>` content in either role (or sanitized away).
  - `variant_summary.json` with `status: succeeded` or, at minimum, no `no_tasks_executed` exit reason.
- If tool calls still don’t appear, try removing any tool‑call parser entirely and/or using a model known to follow OpenAI tool‑calling reliably.

**Operational Notes**
- Reuse/resume:
  - `--reuse` reuses RUNNING instances named `{experiment}-{worker}`; `--gpu-id` reuses a specific instance.
  - Workers and vLLM tmux sessions are restarted cleanly by default; add `--reuse-running-server` only when code hasn’t changed and `/v1/models` is healthy.
- Memory considerations:
  - `--max-model-len 32768` increases VRAM needs. If OOM occurs, lower it or reduce `--gpu-memory-utilization`.

**Why This Is Still a Minimal Diff**
- We do not modify Tau‑Bench core logic; we only:
  - Select an emotional variant by altering the user’s system prompt (via subclass of `LLMUserSimulationEnv`).
  - Route traffic to a selectable endpoint.
  - Add wrapper-side guards and observability (no API changes to Tau‑Bench).
- The critical behavior change came from global server‑side parsing settings (hermes) applied to both roles; scoping or removing that restores the original assumptions of Tau‑Bench’s design.

**Action Checklist**
- [ ] Remove `--tool-call-parser hermes` (or split endpoints) and restart server.
- [ ] Re-run with 2 tasks; inspect first few trajectory turns for roles/content and presence of `tool_calls`.
- [ ] Confirm `variant_summary.json` shows clear success/failure reasons.
- [ ] If needed, keep sanitizer enabled to strip `<think>` leakage.
