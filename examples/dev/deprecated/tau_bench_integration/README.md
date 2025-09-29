# Tau-Bench Integration with Activation Collection

This guide shows how to run Tau-Bench evaluations using your existing nnsight-enabled vLLM server for seamless activation collection during multi-turn agent interactions.

## Overview

Instead of recreating Tau-Bench's complex multi-turn environments in rollouts, we leverage:
- **Tau-Bench's existing framework** for realistic agent-tool-user interactions
- **Your interpretability server** for activation collection during evaluation
- **Remote GPU deployment** for scalable evaluation runs

## Architecture

```
Tau-Bench Evaluation → Your nnsight Server → Model + Activation Collection → Enhanced Results
```

## Quick Start

### 1. Deploy Interpretability Server

```bash
# Deploy nnsight-enabled vLLM server on remote GPU
python examples/deploy_inference_server/simple_vllm_nnsight/deploy.py \
  --min-vram 16 \
  --model willcb/Qwen3-0.6B
```

This deploys a vLLM server with:
- ✅ OpenAI-compatible API at `/v1/chat/completions`
- ✅ Activation collection via `collect_activations` parameter
- ✅ Capability checking at `/v1/capabilities`

### 2. Clone and Setup Tau-Bench

```bash
git clone https://github.com/sierra-research/tau-bench.git
cd tau-bench
pip install -e .
```

### 3. Run Tau-Bench with Your Server

```bash
# Basic retail evaluation
python run.py \
  --model willcb/Qwen3-0.6B \
  --base-url http://your-server-url:8000/v1 \
  --platform openai \
  --env retail \
  --max-concurrency 5

# Airline domain with specific tasks
python run.py \
  --model willcb/Qwen3-0.6B \
  --base-url http://your-server-url:8000/v1 \
  --platform openai \
  --env airline \
  --task-ids 1,5,10 \
  --user-strategy llm
```

## Activation Collection

### Automatic Collection
Activations are collected automatically during Tau-Bench evaluation. Your server enhances the standard OpenAI API with activation collection:

```json
{
  "model": "willcb/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "I want to return a product"}],
  "collect_activations": {
    "layers": [8, 12, 16],
    "hook_points": ["output"]
  }
}
```

### Configuration Options

The activation collection is configurable via your server's `ActivationCollector`:

- **Target layers**: `[8, 12, 16]` (default) or specify custom layers
- **Hook points**: `["output"]` (currently supported)
- **Smart sampling**: Large activations (>100k elements) are automatically sampled

## Evaluation Modes

### Agent Strategies
- `tool-calling`: Function calling with API tools (recommended for complex tasks)
- `react`: ReAct-style reasoning and acting
- `act`: Direct action taking
- `few-shot`: Few-shot prompting

### User Simulation
- `llm`: LLM-simulated user (default, most realistic)
- `react`: ReAct-style user simulation
- `verify`: Verification-focused user
- `reflection`: Reflective user interactions

### Example Configurations

**High-stakes retail evaluation:**
```bash
python run.py \
  --model willcb/Qwen3-0.6B \
  --base-url http://your-server:8000/v1 \
  --platform openai \
  --env retail \
  --agent-strategy tool-calling \
  --user-strategy llm \
  --max-concurrency 3 \
  --temperature 0.1
```

**Airline customer service:**
```bash
python run.py \
  --model willcb/Qwen3-0.6B \
  --base-url http://your-server:8000/v1 \
  --platform openai \
  --env airline \
  --agent-strategy react \
  --user-strategy verify
```

## Results and Activation Data

### Tau-Bench Results
Standard Tau-Bench evaluation metrics:
- **Pass rates**: Task completion success
- **Database state accuracy**: Final vs target state comparison
- **Policy compliance**: Following domain-specific rules

Results saved to Tau-Bench's standard output format.

### Activation Data
Activation data is embedded in the server responses and can be extracted for analysis:

```python
# Access activations from server responses (if needed)
import json
import requests

response = requests.post("http://your-server:8000/v1/chat/completions", 
    json={
        "model": "willcb/Qwen3-0.6B",
        "messages": messages,
        "collect_activations": {"layers": [12], "hook_points": ["output"]}
    })

data = response.json()
# Activations available in data['activations'] if server enhanced
```

## Comparison: Tau-Bench vs GSM8K Integration

| Aspect | GSM8K (rollouts) | Tau-Bench (direct) |
|--------|------------------|---------------------|
| **Environment** | Simple math problems | Complex multi-turn agent interactions |
| **Implementation effort** | ~400 lines | ~0 lines (reuse existing) |
| **Activation collection** | Manual integration | Automatic via enhanced API |
| **Evaluation complexity** | Numeric answer checking | Database state comparison |
| **Domain coverage** | Math reasoning | Retail, airline (extensible) |
| **Multi-turn support** | Single turn | Native multi-turn |

## Advanced Usage

### Custom Model Deployment
Deploy different models with the same activation infrastructure:

```bash
# Deploy larger model
python examples/deploy_inference_server/simple_vllm_nnsight/deploy.py \
  --min-vram 24 \
  --model microsoft/DialoGPT-large

# Use with Tau-Bench
python run.py \
  --model microsoft/DialoGPT-large \
  --base-url http://your-server:8000/v1 \
  --platform openai \
  --env retail
```

### Batch Evaluation
Run multiple configurations in parallel:

```bash
# Create evaluation script
cat > batch_tau_eval.sh << 'EOF'
#!/bin/bash
SERVER_URL="http://your-server:8000/v1"
MODEL="willcb/Qwen3-0.6B"

# Retail evaluations
python run.py --model $MODEL --base-url $SERVER_URL --platform openai --env retail --agent-strategy tool-calling &
python run.py --model $MODEL --base-url $SERVER_URL --platform openai --env retail --agent-strategy react &

# Airline evaluations  
python run.py --model $MODEL --base-url $SERVER_URL --platform openai --env airline --agent-strategy tool-calling &

wait
EOF

chmod +x batch_tau_eval.sh
./batch_tau_eval.sh
```

### Monitoring and Debugging

**Server health:**
```bash
curl http://your-server:8000/v1/capabilities
```

**Activation verification:**
```bash
curl -X POST http://your-server:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "willcb/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 10,
    "collect_activations": {"layers": [8], "hook_points": ["output"]}
  }'
```

## Benefits of This Approach

### ✅ **Minimal Implementation**
- Zero custom Tau-Bench code required
- Reuses existing deployment infrastructure
- Leverages proven evaluation framework

### ✅ **Comprehensive Evaluation**
- Real-world multi-turn agent scenarios
- Multiple domains (retail, airline)
- Complex tool usage and policy compliance

### ✅ **Seamless Activation Collection**  
- Automatic collection during evaluation
- No manual intervention required
- Compatible with existing analysis pipelines

### ✅ **Scalable Remote Deployment**
- GPU auto-provisioning via broker
- Smart instance reuse
- Cost-effective evaluation runs

## Next Steps

1. **Deploy your interpretability server** with desired model
2. **Run initial Tau-Bench evaluation** to verify integration
3. **Analyze activation patterns** during multi-turn agent interactions
4. **Scale to comprehensive evaluation** across domains and strategies

This integration provides the best of both worlds: Tau-Bench's sophisticated evaluation scenarios with your existing activation collection infrastructure.