# Getting Started with GSM8K NNsight Remote

This is a quick 5-minute guide to get you running your first nnsight activation capture.

## Prerequisites

- Access to GPU via broker/bifrost
- `uv` installed and workspace synced

## Option 1: Quick Smoke Test (Recommended First)

This deploys an nnsight server and verifies activation capture works:

```bash
uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py \
  --model willcb/Qwen3-0.6B
```

**What happens:**
1. Provisions a GPU (or reuses if you pass `--gpu-id`)
2. Deploys nnsight server code
3. Loads the model
4. Runs a simple chat completion with activation capture
5. Verifies `.pt` activation files are saved
6. Leaves GPU running for inspection

**Useful flags:**
- `--gpu-id <id>` - Reuse existing GPU
- `--reuse` - Find and reuse GPU by name
- `--skip-sync` - Fast reuse (skip dependency install)
- `--port 8011` - Use different port

## Option 2: Full GSM8K Evaluation

This runs actual math problems and collects activations:

```bash
python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
  --samples 3 \
  --collect-activations \
  --keep-running
```

**What you get:**
- GSM8K accuracy metrics
- Full conversation trajectories
- Neural activation tensors for each problem
- Results saved locally + activations on GPU

## What to Do After Deployment

### Check Server Health
```bash
# The deployment script will print something like:
# 🌐 nnsight server available at: http://194.68.245.163:8001

curl http://194.68.245.163:8001/health
```

### SSH to GPU
```bash
bifrost ssh <gpu_id>

# Once on GPU:
ls -la /tmp/nnsight_activations/  # See activation files
tmux attach -t nnsight-singlepass  # Attach to server session
tail -f ~/nnsight_singlepass.log   # View server logs
```

### Load Activations
```python
import torch
import glob

# Find activation files
files = glob.glob('/tmp/nnsight_activations/activations_*.pt')
print(f"Found {len(files)} activation files")

# Load one
tensor = torch.load(files[0], map_location='cpu')
print(f"Shape: {tensor.shape}")
print(f"Dtype: {tensor.dtype}")
```

## Troubleshooting

### Server won't start
```bash
# Diagnose the instance
uv run python examples/gsm8k_nnsight_remote/diagnose_gpu.py \
  --gpu-id <id> --port 8001

# Try fresh start (wipes venv and restarts clean)
uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py \
  --model willcb/Qwen3-0.6B --gpu-id <id> --fresh
```

### Port conflicts
```bash
# Use different port
uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py \
  --model willcb/Qwen3-0.6B --port 8011
```

### Import errors on remote
```bash
# Force dependency reinstall
uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py \
  --model willcb/Qwen3-0.6B --gpu-id <id> --fresh
```

## Next Steps

1. **Understand the servers**: Read about `server_singlepass.py` vs `server_composition.py` in README.md
2. **Scale up evaluation**: Try `--samples 50` for more robust results
3. **Compare modes**: Run with `--mode no-tools` vs `--mode with-tools`
4. **Analyze activations**: Write custom analysis scripts using the captured tensors
5. **Train probes**: Use activations for linear probe training

## Quick Reference

```bash
# Fast reuse with existing GPU
deploy_and_smoke_test.py --gpu-id <id> --skip-sync

# Full evaluation with tools
deploy_and_evaluate.py --samples 10 --mode with-tools

# Diagnose issues
diagnose_gpu.py --gpu-id <id> --port 8001

# Clean restart
deploy_and_smoke_test.py --gpu-id <id> --fresh
```

For complete documentation, see [README.md](README.md).