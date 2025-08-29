# GSM8K Remote Evaluation

GSM8K evaluation using remote vLLM server deployed via broker/bifrost.

This mirrors `gsm8k_local` but uses a Qwen3-0.6B vLLM server on remote GPU instead of Anthropic API. The evaluation framework and trajectory saving remain identical.

## Features

- **Remote GPU deployment**: Automatically provisions GPU via broker and deploys vLLM server via bifrost
- **Qwen3-0.6B model**: Uses `willcb/Qwen3-0.6B` for math evaluation
- **Streaming evaluation**: Results are saved as evaluation progresses
- **Dual result saving**: Results saved both locally and on remote GPU
- **Optional cleanup**: `--keep-running` flag to preserve GPU instance
- **Configurable vLLM**: Flags for GPU memory utilization, model length, etc.

## Usage

### Basic usage:
```bash
python examples/gsm8k_remote/deploy_and_evaluate.py --samples 3 --mode no-tools
```

### With calculator tools:
```bash
python examples/gsm8k_remote/deploy_and_evaluate.py --samples 3 --mode with-tools --parallel 2
```

### Keep GPU running:
```bash
python examples/gsm8k_remote/deploy_and_evaluate.py --samples 10 --keep-running
```

### Custom GPU/vLLM settings:
```bash
python examples/gsm8k_remote/deploy_and_evaluate.py \
  --samples 5 \
  --min-vram 16 \
  --max-price 0.60 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096
```

## Arguments

### GSM8K Evaluation
- `--samples N`: Number of samples to evaluate (default: 3)
- `--mode {no-tools,with-tools}`: Evaluation mode (default: no-tools)
- `--parallel N`: Number of parallel evaluations (default: 1)

### Remote Deployment
- `--keep-running`: Keep GPU instance running after evaluation
- `--min-vram N`: Minimum VRAM in GB (default: 12)
- `--max-price X`: Maximum price per hour (default: 0.40)

### vLLM Configuration
- `--gpu-memory-utilization X`: GPU memory utilization (default: 0.6)
- `--max-model-len N`: Maximum model length (default: 2048)

## Output Structure

Results are saved in the same format as `gsm8k_local`:

```
examples/gsm8k_remote/results/gsm8k_remote_no_tools_nsamples_3_TIMESTAMP/
├── report.json              # Summary metrics
├── samples/                 # Individual sample results
│   ├── gsm8k_0001.json
│   └── ...
├── trajectories/            # Full conversation trajectories  
│   ├── gsm8k_0001.jsonl
│   └── ...
└── agent_states/           # Detailed agent states
    ├── gsm8k_0001.json
    └── ...
```

Results are also synced to `~/gsm8k_remote_results` on the remote GPU.

## Implementation Notes

- **Model**: Uses `willcb/Qwen3-0.6B` (0.6B parameter model, fits well in 12GB VRAM)
- **Deployment**: Reuses broker/bifrost patterns from `examples/deploy_inference_server/`
- **Evaluation**: Reuses rollouts evaluation framework from `gsm8k_local`
- **Streaming**: Evaluation results are saved progressively during execution
- **Fault tolerance**: Results synced to remote GPU for recovery
- **TODO**: Configuration management should be unified across deployment patterns

## Dependencies

- `broker` client for GPU provisioning
- `bifrost` client for code deployment
- `rollouts` evaluation framework
- `datasets` library for GSM8K dataset