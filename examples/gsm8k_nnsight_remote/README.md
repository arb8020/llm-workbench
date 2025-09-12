# GSM8K nnsight Remote Evaluation

GSM8K evaluation with **residual stream activation collection** using remote nnsight server.

This example combines GSM8K math evaluation with model interpretability - it runs standard GSM8K problems while simultaneously collecting activation patterns from the model's residual stream for probe training.

## ⚡ Fast Path: Single‑Pass NNsight Server (Debug First)

If you just want to validate NNsight activations end‑to‑end as quickly as possible (no GSM8K loop), use the push‑button deploy + smoke test:

```bash
uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py --model willcb/Qwen3-0.6B
```

Idempotent options:
- Reuse by id: `--gpu-id <id>`
- Reuse by name: `--reuse --name nnsight-singlepass-server` (default name)
  - Will push new code, restart tmux session, and re-run smoke test

What this does:
- Provisions a GPU via broker and exposes port 8001
- Pushes the repo and installs extras (`examples_gsm8k_nnsight_remote`)
- Starts `server_singlepass.py` in a tmux session
- Waits for `/health`, loads the model, hits `/v1/chat/completions` with simple savepoints
- Verifies activation `.pt` files exist in `/tmp/nnsight_activations`
- Leaves the GPU running so you can SSH in and debug

Useful follow‑ups it prints:
- Attach tmux: `bifrost exec <ssh> 'tmux attach -t nnsight-singlepass'`
- Tail logs: `bifrost exec <ssh> 'tail -n 200 -f ~/nnsight_singlepass.log'`
- Health: `curl -s http://<host>:8001/health`
- Activation dir: `/tmp/nnsight_activations`

Server code: `examples/gsm8k_nnsight_remote/server_singlepass.py`

### Idempotency, Cleanup, Port, and GPU Reuse

The deploy script can reuse an existing GPU to speed up iteration and avoid re‑provisioning:

- Precedence:
  1) If `--gpu-id <id>` is provided, the script reuses exactly that instance.
  2) Else, if `--reuse` is set, it searches for a RUNNING instance with `--name` (default `nnsight-singlepass-server`) and reuses it if found.
  3) Otherwise, it provisions a new GPU.

- What “reuse” means:
  - Always pushes the latest code to `~/.bifrost/workspace` using `uv sync` with extras.
  - Restarts the server by killing the `nnsight-singlepass` tmux session if present and starting a fresh one.
  - Kills any process bound to the chosen `--port` before starting (prevents stale listeners).
  - Runs the same health check + smoke test and verifies activation files exist on the remote.
  - Leaves the instance running so you can SSH in and continue debugging.

- Port selection:
  - Default `--port 8001`. If you suspect another service (e.g., nginx) is bound to that port, pick a different one and the script will expose and bind that port end‑to‑end: `--port 8011`.

- Examples:
  - Reuse exact instance: `uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py --model willcb/Qwen3-0.6B --gpu-id gpu_12345`
  - Reuse by name: `uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py --model willcb/Qwen3-0.6B --reuse --name nnsight-singlepass-server`
  - Change port: `uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py --model willcb/Qwen3-0.6B --port 8011`

Cleanup and speed options on reuse:
- `--skip-sync`: skips dependency bootstrap entirely (fastest). It runs the server via the existing workspace virtualenv at `~/.bifrost/workspace/.venv`. If that venv is missing, the script fails loudly and tells you to run once without `--skip-sync` (or with `--frozen-sync`).
- `--frozen-sync`: uses `uv sync --frozen` to respect `uv.lock` without re‑resolving or updating (faster and reproducible). Good middle ground after the first full sync.
- `--fresh`: wipes the existing `~/.bifrost/workspace/.venv`, resets/cleans the workspace (`git reset --hard origin/main && git clean -xdf`), kills any processes on `--port`, then performs a clean install/start. This mimics a brand‑new GPU setup while reusing the same instance.
- Example fast reuse: `uv run python examples/gsm8k_nnsight_remote/deploy_and_smoke_test.py --model willcb/Qwen3-0.6B --reuse --skip-sync`

Health checks:
- The script waits for `http://localhost:<port>/health` on the GPU and also validates the OpenAPI spec contains the expected routes (`/models/load`, `/v1/chat/completions`). This prevents confusing 200s from unrelated services.
- If startup fails, it tails `~/nnsight_singlepass.log`.
- It then attempts the external proxy `/health`; if blocked by the provider, it continues using direct remote calls.

## 🎯 Key Features

- **Activation Collection**: Extracts `input_layernorm.output` and `post_attention_layernorm.output` from Qwen3-0.6B
- **Push-button Deployment**: Automatically provisions GPU and deploys nnsight server via broker/bifrost
- **OpenAI API Compatible**: Works with existing evaluation frameworks (rollouts, tau-bench, etc.)
- **GSM8K Evaluation**: Standard math problem evaluation with optional calculator tools
- **Probe Training Ready**: Saves activations in format suitable for linear probe training

## 🚀 Quick Start for New Users

### Step 1: Run Your First Evaluation
```bash
# Basic GSM8K evaluation with activation collection (stored on GPU)
python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
  --samples 3 \
  --collect-activations \
  --keep-running
```

This command will:
- 🔧 Provision a GPU instance automatically
- 🚀 Deploy the nnsight server with Qwen3-0.6B
- 📊 Run 3 GSM8K math problems
- 🧠 **Collect activations and store them on the GPU** (default behavior)
- 🔒 Keep the GPU running so you can access stored activations

### Step 2: Check Your Stored Activations
After the evaluation completes, you'll see output like:
```
🌐 nnsight server available at: http://194.68.245.163:8001
🎉 Success! nnsight GSM8K evaluation completed
🔒 Keeping GPU instance running: gpu_12345
```

Use the server URL to list your stored activations:
```bash
# List all activation sessions
curl http://194.68.245.163:8001/v1/activations/sessions

# Get details about a specific session
curl http://194.68.245.163:8001/v1/activations/session/{session_id}
```

### Step 3: Access the GPU to Analyze Activations
Connect to your GPU instance to work with the stored activations:
```bash
# SSH to the GPU (replace with your GPU details)
bifrost ssh gpu_12345

# Once on the GPU, explore the activation files
ls -la /tmp/nnsight_activations/

# Load activations in Python
python3 -c "
import torch
import glob

# Find activation files
files = glob.glob('/tmp/nnsight_activations/activations_*.pt')
print(f'Found {len(files)} activation files')

for file in files[:2]:  # Show first 2
    tensor = torch.load(file, map_location='cpu')
    print(f'{file}: shape={tensor.shape}, dtype={tensor.dtype}')
"
```

## 🔄 Advanced Usage

### Transfer activations back locally (old behavior):
```bash
python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
  --samples 10 \
  --collect-activations \
  --transfer-activations
```

### With calculator tools:
```bash
python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
  --samples 10 \
  --mode with-tools \
  --collect-activations \
  --keep-running
```

### Multiple experiments on same GPU:
```bash
python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
  --samples 20 \
  --collect-activations \
  --gpu-id gpu_12345  # Use existing GPU
```

## 📊 What Gets Collected

When `--collect-activations` is enabled, the system collects:

1. **Standard GSM8K Results**: Problem/answer pairs, correctness scores, trajectories
2. **Residual Stream Activations**: Model internal representations at key points:
   - `input_layernorm.output` (pre-attention residual stream)
   - `post_attention_layernorm.output` (pre-MLP residual stream)
   - From layers [8, 12, 16] of Qwen3-0.6B (configurable)

## 📁 Output Structure

```
examples/gsm8k_nnsight_remote/results/gsm8k_nnsight_no_tools_nsamples_10_activations_20240912_140530/
├── summary.json                 # Evaluation metrics + activation collection info
├── dataset.jsonl               # GSM8K problems used
├── samples/                    # Individual results with activations
│   ├── gsm8k_0001.json        # Standard result + activation data
│   └── ...
├── trajectories/               # Full conversation logs
│   ├── gsm8k_0001.jsonl
│   └── ...
└── agent_states/              # Detailed agent states
    ├── gsm8k_0001.json
    └── ...
```

### Activation Data Format

Each result includes activation tensors in the `activations` field:
```json
{
  "layer_8_input_layernorm_output": {
    "format": "sampled_tensor",
    "shape": [1, 15, 512], 
    "dtype": "torch.float32",
    "data": [[...]],  // Actual tensor values
    "sample_info": "first_50_features"
  }
}
```

## 🔧 Arguments

### Evaluation Parameters
- `--samples N`: Number of GSM8K samples (default: 3)
- `--mode {no-tools,with-tools}`: Use calculator tools (default: no-tools)  
- `--parallel N`: Parallel evaluations (default: 1)
- `--collect-activations`: Enable activation collection (stored on GPU by default)
- `--transfer-activations`: Transfer activations back locally instead of GPU storage 

### GPU Management
- `--gpu-id ID`: Use existing GPU instance
- `--keep-running`: Keep GPU after evaluation
- `--min-vram N`: Minimum VRAM GB (default: 16)
- `--max-price X`: Max price/hour (default: 0.60)

### Other
- `--verbose`: Detailed logging

## 🧠 Model & Activation Details

- **Model**: `willcb/Qwen3-0.6B` (same as gsm8k_remote example)
- **Architecture**: 24-layer transformer, ~600M parameters  
- **Activation Points**: Layer normalization outputs (residual stream)
- **Default Layers**: [8, 12, 16] (early, middle, late representations)
- **Tensor Format**: JSON-serializable with shape/dtype metadata

## 🔬 Research Applications

This setup enables research on:

1. **Mathematical Reasoning**: How do residual stream activations evolve during math problem solving?
2. **Tool Use**: What activation patterns emerge when using calculator tools vs. direct reasoning?
3. **Linear Probes**: Train probes on activations to predict problem difficulty, solution steps, etc.
4. **Interpretability**: Understand how Qwen3 processes mathematical concepts

## 📚 Dependencies

- `broker` client for GPU provisioning
- `bifrost` client for code deployment  
- `rollouts` evaluation framework
- `datasets` for GSM8K data
- `nnsight` for activation extraction
- `torch` + `transformers` + `accelerate`

## 💡 Usage Tips

1. **Start Small**: Try `--samples 3` first to test the pipeline
2. **Monitor Memory**: 16GB VRAM handles Qwen3-0.6B + activation collection  
3. **Save Activations**: Use `--keep-running` for multiple activation collection runs
4. **Batch Processing**: Higher `--parallel` values speed up evaluation
5. **Tool Analysis**: Compare `--mode no-tools` vs `with-tools` activation patterns

## 🔄 Comparison with gsm8k_remote

| Feature | gsm8k_remote | gsm8k_nnsight_remote |
|---------|-------------|---------------------|
| **Server** | vLLM | nnsight |  
| **Activations** | ❌ | ✅ |
| **Model** | Qwen3-0.6B | Qwen3-0.6B |
| **API** | OpenAI Compatible | OpenAI Compatible |
| **Tools** | Calculator | Calculator |
| **Performance** | Fast | Slower (due to activation extraction) |

## 🎯 Next Steps

After running evaluation:

1. **Analyze Results**: Check `summary.json` for accuracy metrics
2. **Inspect Activations**: Look at tensor shapes and patterns in `samples/`
3. **Train Probes**: Use collected activations for linear probe experiments  
4. **Scale Up**: Run on larger sample sizes for robust results
5. **Compare Modes**: Analyze difference between tool vs. no-tool activation patterns

## 🗄️ Remote Activation Storage

When using `--remote-storage-only`, activations are stored on the GPU instance instead of being transferred back:

### Activation Management Endpoints
```bash
# List all stored activation sessions
curl http://SERVER_IP:8001/v1/activations/sessions

# Get details about a specific session  
curl http://SERVER_IP:8001/v1/activations/session/{session_id}

# Delete activation files for a session
curl -X DELETE http://SERVER_IP:8001/v1/activations/session/{session_id}
```

### Benefits of Remote Storage
- **No transfer bottleneck**: Large activations stay on GPU
- **Faster processing**: Run analyses directly on GPU tensors
- **Memory efficient**: Stream processing without local storage limits
- **Scalable**: Handle activation collections too large for local machines

### Remote Processing Workflow
1. Run evaluation with `--collect-activations --keep-running` (activations stored on GPU by default)
2. Use management endpoints to list stored activation sessions
3. Connect to GPU via bifrost/ssh to run custom analysis scripts
4. Load activations directly from `/tmp/nnsight_activations/` for probe training

## 🔬 Working with Stored Activations

### Example: Training a Linear Probe on GPU
```bash
# SSH to your GPU instance
bifrost ssh gpu_12345

# Create analysis script on the GPU
cat > analyze_activations.py << 'EOF'
import torch
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load all activation files
files = glob.glob('/tmp/nnsight_activations/activations_*.pt')
print(f"Found {len(files)} activation files")

# Load activations for layer 12, input layernorm
layer12_files = [f for f in files if 'layer_12_input_layernorm_output' in f]
activations = []

for file in layer12_files:
    tensor = torch.load(file, map_location='cuda')  # Keep on GPU
    # Take mean over sequence length: [1, seq_len, 512] -> [512]
    pooled = tensor.mean(dim=1).squeeze(0)  # [512]
    activations.append(pooled.cpu().numpy())

if activations:
    X = np.stack(activations)  # [n_samples, 512]
    print(f"Activation matrix shape: {X.shape}")
    print(f"Sample activations ready for probe training!")
else:
    print("No layer 12 activations found")
EOF

# Run the analysis
python3 analyze_activations.py
```

### Example: Comparing Tool vs No-Tool Activations
```bash
# Run two evaluations on the same GPU
python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
  --samples 5 --mode no-tools --collect-activations --gpu-id gpu_12345

python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py \
  --samples 5 --mode with-tools --collect-activations --gpu-id gpu_12345

# SSH to GPU and compare activation patterns
bifrost ssh gpu_12345
python3 -c "
import torch
import glob

files = glob.glob('/tmp/nnsight_activations/activations_*.pt')
sessions = set(f.split('_')[1] for f in files)
print(f'Found {len(sessions)} activation sessions: {sessions}')

# Compare activations between sessions
for session in list(sessions)[:2]:
    session_files = [f for f in files if f'_{session}_' in f]
    print(f'Session {session}: {len(session_files)} files')
"
```

## ⚠️ Architecture Limitations

**Current implementation is hardcoded for standard transformer architecture:**

- **Assumes**: `model.layers[i].input_layernorm.output` and `model.layers[i].post_attention_layernorm.output`
- **Works with**: Qwen3, most standard transformers with typical layer normalization
- **May not work with**: Models with different layer structures, custom normalization schemes, or non-standard architectures

### 🔧 Multi-Architecture Support

For broader model support, see `examples/outlier_features_moe` which provides:
- **Dynamic path detection**: Automatically discovers layer structures
- **Architecture mapping**: `layer_analysis_results.json` with model-specific paths
- **Pattern templates**: Configurable hook point targeting

**Example multi-architecture config:**
```json
{
  "architecture": "Qwen3MoeForCausalLM",
  "layer_access_pattern": "model.layers[i]", 
  "pre_attention_path": "model.layers.{i}.input_layernorm",
  "pre_mlp_path": "model.layers.{i}.post_attention_layernorm"
}
```

To extend this example for new architectures, modify `nnsight_server.py:129-143` to use dynamic path resolution instead of hardcoded component names.

---

This example bridges evaluation and interpretability - you get both performance metrics AND the internal representations needed for mechanistic analysis!
