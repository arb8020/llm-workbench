# Type Checker Issues TODO

Generated from `ty check` output on 2025-09-23. Total: 22 diagnostics found.

## Priority 1: Core Type System Issues (High Impact)

### 1. Environment Interface Issues
**Files:** Multiple example files
**Problem:** Type `Environment` missing `inner_env` and `current_value` attributes
**Impact:** Core functionality broken in examples

- [ ] **examples/controlled_search_demo.py:76** - Fix `final_state.environment.inner_env.current_value`
- [ ] **examples/search_agent.py:715** - Fix `final_state.environment.inner_env.current_value`
- [ ] **examples/search_calculator_demo.py:78** - Fix `final_state.environment.inner_env.current_value`
- [ ] **examples/simple_calculator.py:76** - Fix `final_state.environment.current_value`
- [ ] **examples/simple_search_demo.py:75** - Fix `final_state.environment.inner_env.current_value`
- [ ] **examples/tmux_calculator_demo.py:156** - Fix `final_state.environment.current_value`

**Solution:** Define proper Environment interface with these attributes or fix access patterns.

### 2. Missing asyncio.to_thread (Python Version Issue)
**File:** rollouts/dtypes.py:586
**Problem:** `asyncio.to_thread` not available in older Python versions
**Impact:** Runtime errors on Python < 3.9

- [ ] **rollouts/dtypes.py:586** - Replace `asyncio.to_thread(input, prompt)` with compatible alternative

**Solution:** Use `asyncio.get_event_loop().run_in_executor(None, input, prompt)` for older Python compatibility.

### 3. Invalid Type Annotations
**Files:** tests/regression/ files
**Problem:** Using `any` instead of `Any` in type hints
**Impact:** Type checking failures

- [ ] **tests/regression/debug_stream_parsing.py:40** - Fix `dict[str, any]` → `dict[str, Any]`
- [ ] **tests/regression/proposed_fix.py:25** - Fix `dict[str, any]` → `dict[str, Any]`

**Solution:** Import `Any` from `typing` and use it instead of lowercase `any`.

## Priority 2: Evaluation System Issues (Medium Impact)

### 4. EvalSample Constructor Issues
**File:** rollouts/evaluation.py:51
**Problem:** Missing required parameters in EvalSample construction
**Impact:** Evaluation system broken

- [ ] **rollouts/evaluation.py:51** - Fix `EvalSample(**data)` missing required args `sample_id`, `input_data`, `trajectory`, `agent_states`, `metrics`

### 5. Type Mismatches in Evaluation
**Files:** rollouts/evaluation.py
**Problem:** Function expects `list[EvalSample]` but gets `list[Unknown] | tuple[Unknown]`
**Impact:** Evaluation pipeline broken

- [ ] **rollouts/evaluation.py:330** - Fix type mismatch in `compute_summary_metrics(results, reward_functions)`
- [ ] **rollouts/evaluation.py:341** - Fix type mismatch in `sample_results=results`

## Priority 3: Syntax Compatibility Issues (Low Impact)

### 6. Python 3.8 f-string Syntax
**File:** scripts/refactor_providers.py
**Problem:** Escape sequences in f-strings not supported in Python 3.8
**Impact:** Script fails on Python 3.8

- [ ] **scripts/refactor_providers.py:168** - Fix `f"...{len(agents_body.split('\n'))}..."`
- [ ] **scripts/refactor_providers.py:169** - Fix `f"...{len(providers_body.split('\n'))}..."`

**Solution:** Extract to variables: `lines = agents_body.split('\n'); f"...{len(lines)}..."`

### 7. Function Shadowing Warnings
**Files:** tests/regression/ files
**Problem:** Implicit shadowing of `aggregate_stream` function
**Impact:** Debugging/maintenance issues

- [ ] **tests/regression/debug_stream_parsing.py:194** - Add explicit annotation for monkey patching
- [ ] **tests/regression/proposed_fix.py:150** - Add explicit annotation for monkey patching

**Solution:** Add `# type: ignore[assignment]` or proper function signature annotations.

## Priority 4: Missing Dependencies (Optional)

### 8. Optional Dependencies
**Files:** Various
**Problem:** Missing optional dependencies cause type checker warnings
**Impact:** Examples/features unavailable when deps not installed

- [ ] **examples/oracle_analysis.py:18** - `dotenv` import (optional)
- [ ] **examples/tmux_calculator_demo.py:25** - `libtmux` import (optional)
- [ ] **rollouts/cli/agent_cli.py:30** - `libtmux` import (optional)
- [ ] **rollouts/inference/jax/tok_weights.py:147,157** - `safetensors.numpy` import (optional)
- [ ] **rollouts/inference/jax/tok_weights.py:166** - `torch` import (optional)

**Solution:** These are correctly wrapped in try/except blocks. Consider adding type stubs or `# type: ignore` comments.

## Quick Wins (30min total)

1. **Fix type annotations** (5min): Replace `any` with `Any` in tests
2. **Fix f-string syntax** (5min): Extract variables in refactor script
3. **Fix asyncio compatibility** (10min): Replace `to_thread` with `run_in_executor`
4. **Add type ignores** (10min): Silence optional dependency warnings

## Major Fixes (2-4 hours)

1. **Environment interface** (2h): Define proper Environment ABC with required attributes
2. **Evaluation system** (2h): Fix EvalSample construction and type flow

## Test After Fixes

```bash
ty check
# Should show 0 diagnostics after all fixes
```

## Notes

- Type checker assumes Python 3.8 due to `pyproject.toml` setting
- Most optional dependency warnings are expected and can be ignored
- Focus on Priority 1-2 issues for core functionality
- Priority 3-4 are polish/compatibility improvements