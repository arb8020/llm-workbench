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

**Why It's Failing Now (Root Cause Analysis)**

**Primary Issue: Response Truncation During Reasoning**
- Analysis of `output2.txt` reveals the actual pattern:
  - User simulation generates `<think>` reasoning + actual response content (complete)
  - Agent generates `<think>` reasoning but NO actual response or tool calls (truncated)
  - Pattern: `"tool_calls": null, "function_call": null` with incomplete `<think>` blocks missing `</think>` closing tags
- **Root cause**: Agent responses are being truncated mid-generation, likely due to `max_tokens` limits or response length restrictions
- The agent starts reasoning (`<think>`) but gets cut off before completing the thought AND generating actual tool calls
- This creates a feedback loop where the agent never actually responds, user gets confused, generates more reasoning

**Secondary Issue: Tool-Calling Parser Impact**  
- Both agent and user simulation hit the same endpoint with `--tool-call-parser hermes`
- Small model (`Qwen3-0.6B`) may struggle with hermes format, defaulting to reasoning markup
- User simulation should never see tool-calling templates (should generate natural user text only)

**Required Fixes (Priority Order)**

**1. Fix Response Truncation (Critical)**
- **Investigate `max_tokens` configuration**:
  - Check vLLM server startup flags: Look for `--max-tokens` or similar response length limits
  - Check Tau-Bench completion requests: Verify if `max_tokens` is being passed too low
  - Test hypothesis: Set `max_tokens` to higher value (e.g., 1000+ tokens) to allow completion of reasoning + tool calls
- **Alternative**: Configure stop sequences properly to ensure `</think>` tags close correctly

**2. Remove Tool-Calling Parser for User Simulation (Important)**  
- Option A (simplest): Remove `--tool-call-parser hermes` from vLLM server entirely
  - Keep `--enable-auto-tool-choice` for basic tool-calling support
  - This removes reasoning markup bias for both agent and user
- Option B (isolation): Split endpoints:
  - Agent endpoint: Tool-calling enabled
  - User endpoint: Plain chat only (no tool templates)

**3. Keep Working Components (Low Risk)**
- Always set `RunConfig.user_strategy = "llm"` (enum-valid)
- Keep emotional context injection via `ContextVar` (works independently)  
- Keep checkpoint write guard and `variant_summary.json`
- Keep output sanitizer as fallback cleanup

**Validation Plan**
- **Step 1**: Check current vLLM server configuration for token limits
- **Step 2**: Run small test with increased `max_tokens` (`--tasks 1`, variant: `control`)
- **Step 3**: Examine conversation logs for:
  - Complete assistant responses (not truncated mid-`<think>`)
  - Proper `tool_calls` entries with actual function calls
  - Complete conversation flow without reasoning loops
- **Step 4**: If truncation fixed but still no tool calls, remove `--tool-call-parser hermes`

**Debug Commands**
```bash
# Check current vLLM server logs for token limits
tmux capture-pane -t vllm-session -p | grep -i "max.*token"

# Test with higher token limits in completion request
# (Modify Tau-Bench or add request logging to verify max_tokens parameter)
```

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
