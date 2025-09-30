# Testing GSM8K NNsight Remote

This document explains how to test the nnsight remote activation capture system.

## Quick Test (Recommended First)

```bash
# Deploy and run smoke test
uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py \
  --model willcb/Qwen3-0.6B
```

This will provision a GPU, deploy the server, and verify basic activation capture works.

## Integration Tests

### Test Suite Overview

We have a comprehensive integration test that checks:

1. **Server Health** - Server is accessible
2. **Model Loading** - Model loads successfully
3. **No Duplication** - Multiple requests use same model instance
4. **Activation Capture** - Activations are captured correctly
5. **Remote Storage** - Activations are stored on GPU
6. **Intervention Pattern** - Server supports custom savepoints

### Running Integration Tests

#### Option 1: Test Already-Running Server

```bash
# Test local server
python examples/gsm8k_nnsight_remote/test_integration.py \
  --host localhost --port 8001

# Test remote server
python examples/gsm8k_nnsight_remote/test_integration.py \
  --host 194.68.245.163 --port 8001
```

#### Option 2: Deploy and Test (Full End-to-End)

```bash
python examples/gsm8k_nnsight_remote/test_integration.py \
  --deploy --model willcb/Qwen3-0.6B
```

This will:
1. Provision a GPU via broker
2. Deploy nnsight server
3. Run all integration tests
4. Report results

### Test Output Example

```
ℹ️  GSM8K NNsight Remote - Integration Test Suite
==================================================================
ℹ️  Test 1: Server Health Check
✅ Server is healthy: {'status': 'ok', 'loaded_models': []}
ℹ️  Test 2: Model Loading
✅ Model loaded successfully: willcb/Qwen3-0.6B
ℹ️  Test 3: Check for Model Duplication
✅ Multiple requests succeeded with same model (no duplication detected)
ℹ️  Test 4: Activation Capture
✅ Activations captured successfully:
  - _logits: shape=[1, 10, 151936], dtype=torch.float32, size=5.82MB
    🎉 Multi-token capture detected (10 tokens)!
✅ Activation files saved: 1 files
  - _logits: /tmp/nnsight_activations/activations_test_session_001_abc123__logits.pt
==================================================================
ℹ️  TEST SUMMARY
==================================================================
ℹ️  Tests Passed: 6/6
✅ Multi-token activation capture: WORKING ✅
✅ Memory efficiency: Model not duplicated ✅
```

## GPU Memory Testing

### Check for Model Duplication

To verify the model is loaded only once (not duplicated in memory):

```bash
# Check by GPU ID
python examples/gsm8k_nnsight_remote/check_gpu_memory.py \
  --gpu-id gpu_12345

# Check by SSH
python examples/gsm8k_nnsight_remote/check_gpu_memory.py \
  --ssh root@194.68.245.163:22
```

### Expected Memory Usage

For Qwen3-0.6B (~600M parameters):
- **FP16**: ~1.2 GB
- **FP32**: ~2.4 GB

If you see memory usage significantly higher (e.g., >2.5GB for FP16 or >5GB for FP32), the model might be duplicated.

### Memory Check Output

```
🔍 Checking GPU Memory Usage
============================================================

1. GPU Memory (nvidia-smi):
   Used: 1534 MB
   Total: 24576 MB
   Free: 23042 MB
   Usage: 6.2%
   ✅ Memory usage suggests single model instance (~1200MB expected)

2. Python Processes:
   Running Python processes: 1
   ✅ Single Python process (good - model not duplicated across processes)

3. Detailed Process Info:
   Server processes:
   PID 1234: RSS=1843.2MB, MEM=7.5%, CMD=python server_singlepass.py

4. PyTorch Memory Stats:
   Allocated: 1234.5 MB
   Reserved: 1536.0 MB
   Max Allocated: 1534.8 MB
```

## Testing Interventions

### What Are Interventions?

Interventions allow you to modify activations during model forward pass. Example use cases:
- Zero out specific neurons
- Add steering vectors
- Modify attention patterns

### How to Test Interventions

The integration test checks that the server supports custom savepoints (required for interventions):

```python
# The server should support loading models with custom savepoints
payload = {
    "model_id": "willcb/Qwen3-0.6B",
    "savepoints": [
        {"name": "layer_0", "selector": "model.layers[0].input_layernorm.output"},
        {"name": "layer_12", "selector": "model.layers[12].post_attention_layernorm.output"}
    ]
}
```

### Manual Intervention Test

To manually test interventions, you'll need to:

1. **SSH to the GPU**
   ```bash
   bifrost ssh gpu_12345
   ```

2. **Create an intervention script**
   ```python
   from nnsight import LanguageModel
   import torch

   lm = LanguageModel("willcb/Qwen3-0.6B", device_map="auto")

   # Intervention example: zero out layer 0 output
   with lm.generate("Hello, world!", max_new_tokens=5) as tracer:
       with tracer.invoke():
           # Capture original activation
           original = lm.model.layers[0].input_layernorm.output.save()

           # Intervene: zero out the activation
           lm.model.layers[0].input_layernorm.output = torch.zeros_like(
               lm.model.layers[0].input_layernorm.output
           )

           # Capture modified activation
           modified = lm.model.layers[0].input_layernorm.output.save()

   print(f"Original shape: {original.shape}")
   print(f"Modified shape: {modified.shape}")
   print(f"Modified is zeros: {torch.allclose(modified, torch.zeros_like(modified))}")
   ```

3. **Run the script**
   ```bash
   python test_intervention.py
   ```

## Troubleshooting Test Failures

### Test 1: Server Health - FAILED
```
❌ Cannot connect to server
```
**Solution**:
- Check server is running: `curl http://<host>:<port>/health`
- Check firewall/port exposure
- Try different port: `--port 8011`

### Test 2: Model Loading - FAILED
```
❌ Model loading failed: 500 Internal Server Error
```
**Solution**:
- Check GPU has enough memory: `nvidia-smi` on remote
- Check disk space: `df -h`
- Look at server logs: `tail -f ~/nnsight_singlepass.log`

### Test 3: No Duplication - FAILED
```
❌ Request 2 failed: 500
```
**Solution**:
- Model might have crashed after first request
- Check server logs for errors
- Restart server with `--fresh` flag

### Test 4: Activation Capture - FAILED
```
❌ No activations captured
```
**Solution**:
- Check `server_singlepass.py` is being used (not old version)
- Verify nnsight version: `pip show nnsight` (need >=0.4)
- Check activation capture is enabled: `"store_activations": true`

### Test 5: Remote Storage - FAILED
```
⚠️  No activation files to verify
```
**Solution**:
- SSH to GPU and check: `ls /tmp/nnsight_activations/`
- Check disk space: `df -h /tmp`
- Check permissions: `ls -la /tmp/nnsight_activations/`

## Continuous Integration

### GitHub Actions (Future)

We can add a CI workflow to test this automatically:

```yaml
name: Test NNsight Remote
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: uv sync --extra examples_gsm8k_nnsight_remote
      - name: Run integration tests
        run: |
          python examples/gsm8k_nnsight_remote/test_integration.py \
            --deploy --model willcb/Qwen3-0.6B
        env:
          BROKER_API_KEY: ${{ secrets.BROKER_API_KEY }}
```

## Performance Benchmarks

### Expected Performance

- **Model loading**: ~30-60 seconds
- **Single completion (5 tokens)**: ~1-2 seconds
- **Activation capture overhead**: ~10-20% slower than without capture
- **Memory usage (Qwen3-0.6B FP16)**: ~1.2GB

### Benchmark Test

```bash
python -c "
import time
import requests

base_url = 'http://194.68.245.163:8001'

# Warmup
requests.post(f'{base_url}/v1/chat/completions', json={
    'model': 'willcb/Qwen3-0.6B',
    'messages': [{'role': 'user', 'content': 'Hello'}],
    'max_tokens': 5
}, timeout=30)

# Benchmark
start = time.time()
for i in range(10):
    requests.post(f'{base_url}/v1/chat/completions', json={
        'model': 'willcb/Qwen3-0.6B',
        'messages': [{'role': 'user', 'content': f'Test {i}'}],
        'max_tokens': 5,
        'store_activations': True
    }, timeout=30)
end = time.time()

print(f'Average time per request: {(end-start)/10:.2f}s')
"
```

## Summary

Use these testing tools to ensure:
- ✅ Model loads successfully
- ✅ No model duplication in memory
- ✅ Activations captured correctly
- ✅ Multi-token capture works
- ✅ Remote storage functions properly
- ✅ Performance is reasonable

For questions or issues, see the main [README.md](README.md) or check server logs.