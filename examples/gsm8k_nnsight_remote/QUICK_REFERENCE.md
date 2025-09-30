# Quick Reference - GSM8K NNsight Remote

One-page reference for common operations.

## 🚀 Quick Start

```bash
# Deploy and test in one command
uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py \
  --model willcb/Qwen3-0.6B

# Run full evaluation (3 problems)
python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
  --samples 3 --collect-activations --keep-running
```

## 🧪 Testing

```bash
# Integration tests
python test_integration.py --host localhost --port 8001

# Check for model duplication
python check_gpu_memory.py --gpu-id gpu_12345

# Full end-to-end test (deploy + test)
python test_integration.py --deploy --model willcb/Qwen3-0.6B
```

## 🔧 Common Operations

### Reuse Existing GPU
```bash
# By ID
deploy_and_smoke_test.py --gpu-id gpu_12345 --skip-sync

# By name
deploy_and_smoke_test.py --reuse --name my-server
```

### Fresh Start
```bash
# Clean everything and restart
deploy_and_smoke_test.py --gpu-id gpu_12345 --fresh
```

### Diagnose Issues
```bash
# Full diagnostic report
diagnose_gpu.py --gpu-id gpu_12345 --port 8001
```

### SSH to GPU
```bash
# Get SSH command
bifrost ssh gpu_12345

# View server logs
tail -f ~/nnsight_singlepass.log

# Check activations
ls -la /tmp/nnsight_activations/

# Attach to tmux
tmux attach -t nnsight-singlepass
```

## 📊 Check Results

### Local Results
```bash
ls examples/gsm8k_nnsight_remote/results/
cat examples/gsm8k_nnsight_remote/results/*/summary.json | jq
```

### Remote Activations
```bash
# On remote GPU
python3 << EOF
import torch
import glob

files = glob.glob('/tmp/nnsight_activations/*.pt')
print(f"Found {len(files)} activation files")

for f in files[:3]:
    t = torch.load(f, map_location='cpu')
    print(f"{f}: shape={t.shape}, size={t.numel()*t.element_size()/(1024**2):.1f}MB")
EOF
```

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| Server won't start | `deploy_and_smoke_test.py --fresh` |
| Port conflict | `deploy_and_smoke_test.py --port 8011` |
| Import errors | `bifrost exec <ssh> 'cd ~/.bifrost/workspace && uv sync --extra examples_gsm8k_nnsight_remote'` |
| Out of memory | Check `nvidia-smi`, reduce batch size, or use larger GPU |
| No activations | Verify `"store_activations": true` in request |

## 📁 File Locations

| File | Purpose |
|------|---------|
| `server_singlepass.py` | Production server (multi-token) |
| `server_composition.py` | Cleaner server (for development) |
| `deploy_and_smoke_test.py` | Quick deployment + test |
| `deploy_and_evaluate.py` | Full GSM8K evaluation |
| `test_integration.py` | Comprehensive test suite |
| `check_gpu_memory.py` | Memory duplication check |
| `diagnose_gpu.py` | Full diagnostic tool |

## 🔗 Quick Links

- Full docs: [README.md](README.md)
- Testing guide: [TESTING.md](TESTING.md)
- Getting started: [GETTING_STARTED.md](GETTING_STARTED.md)
- Revival notes: [REVIVAL_NOTES.md](REVIVAL_NOTES.md)

## ⚡ Copy-Paste Commands

```bash
# Complete workflow: deploy, test, analyze
uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py --model willcb/Qwen3-0.6B
python examples/gsm8k_nnsight_remote/test_integration.py --host <HOST> --port 8001
python examples/gsm8k_nnsight_remote/check_gpu_memory.py --gpu-id <GPU_ID>
bifrost ssh <GPU_ID>
ls -la /tmp/nnsight_activations/
```