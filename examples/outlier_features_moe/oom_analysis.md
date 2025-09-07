# OOM Analysis - Code Snippets and Error

## GPU Configuration at Time of OOM

```
Sun Sep  7 01:55:45 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          On  |   00000000:00:06.0 Off |                    0 |
| N/A   42C    P0             69W /  300W |   58661MiB /  81920MiB |      1%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100 80GB PCIe          On  |   00000000:00:0B.0 Off |                    0 |
| N/A   48C    P0             69W /  300W |   58659MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```

- GPU 0: 58,661 MiB / 81,920 MiB (71.6% utilized)
- GPU 1: 58,659 MiB / 81,920 MiB (71.6% utilized)

## OOM Error

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 1 has a total capacity of 79.14 GiB of which 840.75 MiB is free. Process 1068993 has 78.31 GiB memory in use. Of the allocated memory 77.81 GiB is allocated by PyTorch, and 9.84 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

Full stack trace:
```
    output = module._old_forward(*args, **kwargs)
  File "/root/.bifrost/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
```

## Analysis Configuration

```
============================================================
FULL OUTLIER ANALYSIS PIPELINE
============================================================
Model: Qwen/Qwen3-30B-A3B
Dataset: HuggingFaceFW/fineweb-edu
Sequences: 4 x 2048 tokens
Batch size: 1
Layers: All layers (will be determined from model)
Threshold: 6.0
Save dir: ./full_analysis_results
```

Model structure:
```
Auto-detected 48 layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
```

Loading progress:
```
Fetching 16 files: 100%|██████████| 16/16 [01:32<00:00,  5.76s/it]
Loading checkpoint shards:  19%|█▉        | 3/16 [04:12<18:31, 85.51s/it]
```

## Main Analysis Loop Structure

**File: `run_full_analysis.py`**
```python
# Step 1: Extract activations in batches
num_batches = args.num_sequences // args.batch_size
all_run_dirs = []

try:
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + args.batch_size
        batch_texts = text_sequences[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch_texts)} sequences)")
        
        run_dir, metadata = extract_activations(
            model_name=args.model,
            texts=batch_texts,
            layers=args.layers,
            save_dir=args.save_dir
        )
        all_run_dirs.append(run_dir)
        print(f"✅ Batch {batch_idx + 1} completed: {run_dir}")
```

## Model Loading in extract_activations Function

**File: `extract_activations.py`**
```python
def extract_activations(
    model_name="allenai/OLMoE-1B-7B-0125-Instruct",
    texts=None,
    layers=None,
    save_dir="./activations"
):
    # ... validation code ...
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    llm = LanguageModel(model_name, device_map="auto")
    
    # Determine layers if not specified
    if layers is None:
        try:
            num_layers = len(llm.model.layers)
            layers = list(range(num_layers))
            print(f"Auto-detected {num_layers} layers: {layers}")
        except AttributeError:
            raise ValueError(f"Could not auto-detect layers for model {model_name}. Please specify --layers explicitly.")
    
    # Extract activations using pure batch function
    activations = extract_activations_batch(llm, texts, layers)
```

## Activation Extraction Function

**File: `extract_activations.py`**
```python
def extract_activations_batch(model, texts: list[str], layers: list[int]) -> dict[str, torch.Tensor]:
    """
    Pure function: extract activations from batch of texts.
    """
    activations = {}
    with model.trace(texts) as tracer:
        for layer_idx in layers:
            ln_into_attn = model.model.layers[layer_idx].input_layernorm.output.save()
            ln_into_mlp = model.model.layers[layer_idx].post_attention_layernorm.output.save()
            
            activations[f"layer_{layer_idx}_ln_attn"] = ln_into_attn
            activations[f"layer_{layer_idx}_ln_mlp"] = ln_into_mlp
    
    # Convert proxies to tensors
    result = {}
    for layer_name, activation_proxy in activations.items():
        tensor = activation_proxy.detach().cpu()
        assert tensor.dim() == 3, f"Expected 3D tensor for {layer_name}, got shape {tensor.shape}"
        result[layer_name] = tensor
    
    return result
```

## Execution Flow

1. Main analysis starts with 4 sequences, batch size 1 = 4 batches
2. For each batch (4 iterations):
   - Call `extract_activations()` 
   - Inside `extract_activations()`: `llm = LanguageModel(model_name, device_map="auto")`
   - Model loading completes successfully
   - During forward pass in `extract_activations_batch()`, OOM occurs on GPU 1