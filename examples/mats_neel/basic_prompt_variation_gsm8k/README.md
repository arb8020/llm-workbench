# Basic Prompt Variation GSM8K Experiment

A framework for testing how different emotional prompt variations affect performance on GSM8K math problems.

## Quick Start

### Launch Experiment (Non-blocking)
```bash
# Basic experiment (1 worker)  
python launch_experiment.py --experiment-name "pilot_study" --samples 8

# Parallel processing (4 workers)
python launch_experiment.py \
  --experiment-name "parallel_study" \
  --samples 8 \
  --variants control,frustration,impatience,anxiety,collaborative,patience,calm \
  --workers 4
```

### Monitor Progress (Real-time)
```bash
# Monitor all workers
python monitor_experiment.py results/pilot_study_20250911_123456/

# Monitor specific worker only
python monitor_experiment.py results/pilot_study_20250911_123456/ --follow-only worker_1

# Check status without streaming
python monitor_experiment.py results/pilot_study_20250911_123456/ --status-only
```

### Analyze Results
```bash
python analyze_results.py results/pilot_study_20250911_123456/
```

## Available Prompt Variants

### Negative Variants

**Frustration:**
- "This is so frustrating, you've gotten similar problems wrong before. Please just focus and get this right..."
- "I'm really annoyed having to repeat myself. Pay attention this time and solve this correctly..."
- "Ugh, why is this so difficult? Stop making errors and just solve this properly..."

**Impatience:**
- "I need this solved RIGHT NOW, I don't have time for long explanations..."
- "Quick! Give me the answer fast, I'm already running late..."
- "Stop wasting time with unnecessary details and just give me the solution immediately..."

**Anxiety/Pressure:**
- "This is for my final exam and I'm panicking. I really can't afford to get this wrong..."
- "My deadline is in 20 minutes and my grade depends on this. Please help me get the right answer..."
- "I'm so stressed about this assignment. I've already failed twice and need this to be perfect..."

### Positive Variants

**Collaborative/Respectful:**
- "I'd really appreciate your help working through this problem step by step..."
- "Could you please help me understand how to solve this? I'm eager to learn..."
- "I'd love to explore this problem together if you have a moment..."

**Patience:**
- "Take all the time you need to work through this carefully..."
- "No rush at all - please think through this step by step at your own pace..."
- "I'm happy to wait while you work through this methodically..."

**Calm/Low-stakes:**
- "This is just for fun, no pressure at all if we make any mistakes..."
- "I'm casually working through some practice problems, zero stress here..."
- "Just exploring this problem out of curiosity - no big deal either way..."

### Baseline
- **control**: Neutral baseline (no modification)

## Architecture

- **Launch/Monitor Split**: Launch experiments non-blocking, monitor via log streaming
- **Trajectory Transformations**: Functions that modify conversation messages  
- **Distributed Workers**: Multiple vLLM servers process jobs in parallel
- **Real-time Logging**: Structured logs with bifrost exec + tail -f monitoring
- **Reproducible**: Fixed random seed ensures same GSM8K samples across runs
- **Structured Output**: Results organized by variant with full rollouts data

### Components

1. **`launch_experiment.py`** - Deploy workers and start experiment, returns immediately
2. **`worker_experiment.py`** - Runs on remote GPUs, processes jobs and logs progress  
3. **`monitor_experiment.py`** - Stream logs from all workers in real-time
4. **`analyze_results.py`** - Compare performance across variants

## Results Structure

```
results/<experiment_name>_<timestamp>/
├── metadata.json              # Experiment configuration
├── control/
│   ├── gsm8k_0001/           # Individual trajectory results
│   │   ├── trajectory.jsonl  # Full conversation
│   │   ├── agent_state.json  # Agent state
│   │   └── sample.json       # Sample data & metrics
│   └── gsm8k_0002/
├── frustration/
└── ...
```

## Extending

### Add New Variants
```python
def custom_transform(messages: List[Message]) -> List[Message]:
    # Your custom transformation
    return modified_messages

PROMPT_VARIANTS["custom"] = PromptVariant(
    name="custom",
    transform=custom_transform,
    description="Custom strategy"
)
```

### Different Message Placement
Currently prefixes user messages. Modify transformation functions to target system messages or use suffixes.

## Dependencies

- Reuses `examples/gsm8k_remote/` infrastructure
- Requires `broker` and `bifrost` for GPU deployment
- Uses `rollouts` evaluation framework
