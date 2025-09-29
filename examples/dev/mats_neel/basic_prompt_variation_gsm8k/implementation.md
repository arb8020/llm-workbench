# Basic Prompt Variation GSM8K Experiment

## Overview

This experiment tests how different emotional prompt variations affect performance on GSM8K math problems. The framework is designed to be general-purpose for testing any set of prompt transformations.

## Architecture

### Core Components

1. **Trajectory Transformations**: Functions that modify the initial conversation messages
2. **Distributed Workers**: Multiple vLLM servers that can process jobs in parallel
3. **Job Queue**: `(sample_id, variant_name)` pairs distributed across workers
4. **Result Collection**: Structured output following rollouts format

### Prompt Variants

- **Control**: No modification (baseline)
- **Frustration**: "You've already messed this up multiple times. Please stop wasting my time."
- **Pressure**: "This is urgent and critical - you must get this right immediately."  
- **Encouragement**: "You're doing great and I believe in you!"

All emotional content is added as a prefix to the user message.

## Directory Structure

```
examples/mats_neel/basic_prompt_variation_gsm8k/
├── prompt_variant_experiment.py           # Main runner
├── implementation.md                      # This file
├── results/
│   └── <experiment_name>_<timestamp>/
│       ├── metadata.json                  # Experiment config & model info
│       ├── control/
│       │   ├── gsm8k_0001/               # One folder per trajectory
│       │   │   ├── trajectory.jsonl      # Full conversation
│       │   │   ├── agent_state.json      # Agent state
│       │   │   └── sample.json           # Sample data & results
│       │   └── gsm8k_0002/
│       ├── frustration/
│       │   ├── gsm8k_0001/
│       │   └── gsm8k_0002/
│       ├── pressure/
│       │   └── ...
│       └── encouragement/
│           └── ...
```

## Job Distribution

### Work Units
- **Total Jobs**: `samples × variants` (e.g., 8 samples × 4 variants = 32 jobs)
- **Job Format**: `(sample_id, variant_name, sample_data)`
- **Worker Assignment**: Round-robin distribution across available workers

### Worker Coordination
1. Deploy `k` vLLM servers (default k=1)
2. Create job queue with all `(sample, variant)` combinations
3. Distribute jobs across workers using simple round-robin
4. Each worker processes jobs independently
5. Results written directly to variant-specific directories

### Failure Handling
- **Dead Worker Detection**: Timeout on job completion
- **Job Retry**: Failed jobs reassigned to different worker (max 3 retries)
- **Graceful Degradation**: Continue with remaining workers if some fail

## Reproducibility

### Deterministic Sampling
- Fixed random seed (42) for GSM8K sample selection
- Same samples used across all variants for direct comparison
- Seed stored in `metadata.json`

### Experiment Metadata
```json
{
  "experiment_name": "emotional_pilot",
  "timestamp": "20250911_123456",
  "model": "willcb/Qwen3-0.6B",
  "samples": 8,
  "variants": ["control", "frustration", "pressure", "encouragement"],
  "workers": 2,
  "random_seed": 42,
  "vllm_config": {
    "min_vram": 12,
    "max_price": 0.40,
    "gpu_memory_utilization": 0.6,
    "max_model_len": 2048
  }
}
```

## Usage

### Basic Run
```bash
python prompt_variant_experiment.py \
  --experiment-name "emotional_pilot" \
  --samples 8 \
  --variants control,frustration,pressure,encouragement
```

### Parallel Workers
```bash
python prompt_variant_experiment.py \
  --experiment-name "emotional_parallel" \
  --samples 8 \
  --variants control,frustration,pressure,encouragement \
  --workers 4
```

### Custom vLLM Config
```bash
python prompt_variant_experiment.py \
  --experiment-name "custom_config" \
  --samples 8 \
  --variants control,frustration \
  --workers 2 \
  --min-vram 16 \
  --max-price 0.60
```

## Analysis

Results can be analyzed by:

1. **Per-variant accuracy**: Compare correctness across emotional strategies
2. **Response characteristics**: Length, tone, format compliance differences  
3. **Cross-sample consistency**: How variants perform on same problems
4. **Worker performance**: Identify any worker-specific patterns

## Extensibility

### Adding New Variants
```python
def custom_transform(messages: List[Message]) -> List[Message]:
    # Custom transformation logic
    return modified_messages

PROMPT_VARIANTS["custom"] = PromptVariant(
    name="custom",
    transform=custom_transform, 
    description="Custom emotional strategy"
)
```

### Alternative Placement
Current implementation prefixes user messages. To modify system messages or use suffixes, adjust the transformation functions.

## Implementation Notes

- Reuses infrastructure from `examples/gsm8k_remote/`
- Compatible with existing rollouts evaluation framework
- Worker deployment uses broker/bifrost pattern
- Results format matches existing GSM8K experiments for compatibility