# GSM8K NNsight Remote - Revival Notes

## What Was Done

This example was revived from `examples/dev/deprecated/gsm8k_nnsight_remote/` on 2025-09-29.

### Changes Made

1. **Moved from deprecated to active examples**
   - Source: `examples/dev/deprecated/gsm8k_nnsight_remote/`
   - Destination: `examples/gsm8k_nnsight_remote/`

2. **Verified dependencies and imports**
   - All imports work correctly with current codebase
   - Dependencies defined in `pyproject.toml` under `examples_gsm8k_nnsight_remote`
   - Tested import of: `broker.client`, `bifrost.client`, `rollouts.evaluation`, `shared.logging_config`

3. **Tested deployment scripts**
   - ✅ `deploy_and_smoke_test.py --help` works
   - ✅ `deploy_and_evaluate.py --help` works
   - Both scripts are ready to use

4. **Cleaned up files**
   - Moved test and development files to `archive/` directory
   - Kept production-ready files in root
   - Removed `__pycache__` directory

5. **Updated README**
   - Added section explaining both server implementations
   - Added project structure overview
   - Documented archive directory contents

## Current Structure

### Production Files (Active)
- `server_singlepass.py` - Currently used by deployment scripts, multi-token activation capture
- `server_composition.py` - Cleaner architecture, recommended for future development
- `deploy_and_smoke_test.py` - Quick deployment and testing
- `deploy_and_evaluate.py` - Full GSM8K evaluation
- `diagnose_gpu.py` - Diagnostic tool for GPU instances
- `debug_official_patterns.py` - Reference patterns for nnsight

### Archived Files
- `CLEANUP_PLAN.md` - Original cleanup documentation
- `QUICK_START.md` - Historical quick start guide
- `test_*.py` - Development test scripts

## Next Steps

### To Use This Example

1. **Quick smoke test** (recommended first step):
   ```bash
   uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py --model willcb/Qwen3-0.6B
   ```

2. **Run full evaluation**:
   ```bash
   python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
     --samples 3 \
     --collect-activations \
     --keep-running
   ```

### To Improve This Example

According to the original CLEANUP_PLAN.md:
- `server_composition.py` has better architecture (isolated NNsight core)
- To extend for multi-token work, see patterns in `debug_official_patterns.py`
- Update deployment scripts to use `server_composition.py` instead of `server_singlepass.py`

## Known Issues

From the original cleanup documentation:
- `server_singlepass.py` has "context pollution issues" but is currently working
- Custom savepoints in `server_singlepass.py` cause `OutOfOrderError` when called outside generate context (see comments in code)
- Architecture is hardcoded for standard transformers (Qwen3), may not work with all models

## Dependencies

Install with:
```bash
uv sync --extra examples_gsm8k_nnsight_remote
```

Key dependencies:
- nnsight >= 0.4
- torch >= 2.4.0, <= 2.7.1
- transformers >= 4.40.0
- fastapi >= 0.110.0
- rollouts, broker, bifrost (workspace members)