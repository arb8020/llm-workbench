# Inference Server Deployment Options

This directory contains deployment scripts for different types of inference servers, all using the same broker + bifrost infrastructure.

## Available Server Types

### 🚀 simple_vllm
**Production-ready vLLM server** - High-performance inference with vLLM optimizations.

- **Use case**: Production inference, high throughput
- **Features**: PagedAttention, dynamic batching, optimized memory usage
- **VRAM**: 8GB minimum
- **API**: Standard OpenAI-compatible endpoints

```bash
python examples/deploy_inference_server/simple_vllm/deploy.py --min-vram 8 --max-price 0.40
```

### 📊 simple_sglang  
**Alternative production server** - SGLang inference with RadixAttention.

- **Use case**: Production inference with prefix caching
- **Features**: RadixAttention, prefix caching, structured generation
- **VRAM**: 8GB minimum  
- **API**: OpenAI-compatible + SGLang extensions

```bash
python examples/deploy_inference_server/simple_sglang/deploy.py --min-vram 8 --max-price 0.40
```

### 🧠 simple_vllm_nnsight
**Interpretability research server** - FastAPI + nnsight.VLLM integration.

- **Use case**: Interpretability research, activation analysis
- **Features**: Activation collection, intervention capabilities, research tools
- **VRAM**: 12GB minimum (interpretability overhead)
- **API**: OpenAI-compatible + interpretability extensions
- **⚠️ Version constraints**: Uses vLLM 0.6.4.post1 + Triton 3.1.0 (nnsight requirements)

```bash
python examples/deploy_inference_server/simple_vllm_nnsight/deploy.py --min-vram 12 --max-price 0.60
```

## Usage Patterns

### Deploy and Test

#### Quick Test with Universal Client
```bash
# Deploy any server type
python examples/deploy_inference_server/simple_vllm/deploy.py --json > server_info.json
SERVER_URL=$(jq -r '.url' server_info.json)

# Test with universal client (no external dependencies needed)
python examples/deploy_inference_server/test_client.py --url $SERVER_URL --prompt "The capital of France is"

# Test streaming
python examples/deploy_inference_server/test_client.py --url $SERVER_URL --prompt "Write a haiku about coding" --stream

# Test with activation collection (interpretability servers only)
python examples/deploy_inference_server/test_client.py --url $SERVER_URL \
  --prompt "The answer to 2+2 is" --collect-activations --activation-layers 6 12 --verbose
```

#### Manual cURL Testing  
```bash
# Test standard completion
curl -X POST $SERVER_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-community/gpt2","messages":[{"role":"user","content":"Hello!"}],"max_tokens":20}'
```

### With Interpretability Features (nnsight server only)
```bash
# Test activation collection
curl -X POST $SERVER_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"openai-community/gpt2",
    "messages":[{"role":"user","content":"The capital of France is"}],
    "max_tokens":10,
    "collect_activations":{
      "layers":[6,12],
      "hook_points":["output"],
      "positions":[-1]
    }
  }'

# Check interpretability capabilities
curl $SERVER_URL/v1/capabilities
```

### Integration with Rollouts

All servers expose the same OpenAI-compatible API, so they work seamlessly with existing rollouts code:

```python
# Standard inference (any server)
endpoint = Endpoint(
    provider="vllm",  # or "sglang" 
    model="openai-community/gpt2",
    api_base="http://server-url:8000"
)

# Interpretability-enabled inference
endpoint = Endpoint(
    provider="nnsight",  # Routes to rollout_nnsight()
    model="openai-community/gpt2", 
    api_base="http://interp-server-url:8000",
    collect_activations={
        "layers": [6, 12],
        "hook_points": ["output"]
    }
)
```

## Server Management

### Monitoring
```bash
# Check server logs
bifrost exec <ssh-connection> 'cat ~/server.log'

# View tmux session
bifrost exec <ssh-connection> 'tmux attach -t <session-name>'

# Check server health
curl <server-url>/health
```

### Cleanup
```bash
# Stop server
bifrost exec <ssh-connection> 'tmux kill-session -t <session-name>'

# Terminate GPU instance
broker terminate <instance-id>
```

## Performance Characteristics

| Server Type | VRAM (GB) | Latency | Throughput | Use Case |
|-------------|-----------|---------|------------|----------|
| simple_vllm | 8+ | Low | High | Production |
| simple_sglang | 8+ | Low | High | Production + prefix caching |
| simple_vllm_nnsight | 12+ | Higher | Lower | Research + interpretability |

## Architecture Notes

- All servers use the same **broker + bifrost** deployment pattern
- All expose **port 8000** for consistency
- All provide **OpenAI-compatible APIs** for seamless integration
- Interpretability server adds **activation collection extensions**
- Future: server startup scripts will migrate to `engine/` module

## Version Compatibility

**⚠️ Important**: Different server types may use different dependency versions:

| Server Type | vLLM Version | Triton Version | Notes |
|-------------|--------------|----------------|-------|
| simple_vllm | Latest | Latest | Uses current project dependencies |
| simple_sglang | N/A | Latest | Uses SGLang instead of vLLM |
| simple_vllm_nnsight | 0.6.4.post1 | 3.1.0 | **Fixed versions required by nnsight** |

**Isolation**: Each deployment creates a fresh GPU instance, so version conflicts are avoided. The interpretability server explicitly removes conflicting vLLM versions before installing nnsight-compatible versions.

## Testing Guide

### Universal Test Client

The `test_client.py` script works with all server types and provides comprehensive testing:

```bash
# Basic usage
python examples/deploy_inference_server/test_client.py --url http://server:8000 --prompt "Hello world"

# All options
python examples/deploy_inference_server/test_client.py \
  --url http://server:8000 \
  --prompt "Explain quantum computing" \
  --system "You are a helpful physics tutor" \
  --max-tokens 100 \
  --temperature 0.7 \
  --stream \
  --collect-activations \
  --activation-layers 6 12 18 \
  --activation-hooks output input \
  --json \
  --verbose
```

### Testing Different Server Types

```bash
# Test vLLM server
python test_client.py --url http://vllm-server:8000 --prompt "2+2 equals"

# Test SGLang server  
python test_client.py --url http://sglang-server:8000 --prompt "2+2 equals"

# Test interpretability server with activation collection
python test_client.py --url http://interp-server:8000 --prompt "2+2 equals" \
  --collect-activations --activation-layers 6 12 --verbose
```

### Performance Comparison

```bash
# Benchmark different servers
for server in vllm-url sglang-url interp-url; do
  echo "Testing $server:"
  time python test_client.py --url $server --prompt "Write a short story about AI" --max-tokens 100
done
```

## Coming Soon

- **simple_sglang** implementation
- **Engine module integration** for server startup scripts  
- **Batch processing** support for interpretability features
- **Performance benchmarks** comparing all server types