# Outlier Features Analysis - Updated Handoff Document

## Recent Progress Summary

**✅ MAJOR BREAKTHROUGHS**: 
1. **Fixed NVIDIA GPU Manufacturer Filtering**: Added proper manufacturer field to GPUOffer type and query interface
2. **Multi-GPU Support Added**: Implemented clean gpu_count parameter (1-8 GPUs) with explicit API design
3. **GPU Filter Feature**: Added `--gpu-filter` parameter for exact GPU selection (e.g., "A100", "H100")
4. **Memory Leak Identified and Fixed**: Found critical issue where model was loaded 4x per batch
5. **Memory-Optimized Pipeline**: Implemented chunked activation extraction with balanced device mapping

**🔄 CURRENT STATUS**: Testing memory-optimized version on 2x A100 80GB

## Critical Bug Fix: Memory Leak

### The Problem
Analysis was failing with OOM errors even on 2x A100 (160GB total VRAM) because:

```python
# BAD: Model loaded for every batch (4x model in memory!)
for batch_idx in range(num_batches):  # 4 batches
    run_dir, metadata = extract_activations(  # Loads 156B model EVERY TIME!
        model_name=args.model,  # 4 × 156B = 624B parameters in GPU memory
        texts=batch_texts,
        layers=args.layers,
        save_dir=args.save_dir
    )
```

### Root Cause Analysis
- **GPU 1**: 78.3GB / 79.1GB used when trying to allocate 1.16GB more
- **Peak vs steady-state**: nvidia-smi showed 58.6GB, but forward pass peaked at 78.3GB
- **Model reloading**: Each batch loaded a fresh 156B model instance
- **All-layer extraction**: Saved activations from all 48 layers simultaneously
- **Unbalanced device mapping**: device_map="auto" overpacked GPU 1

## New Memory-Optimized Implementation

### Key Changes
1. **Load model once**: Moved LanguageModel construction outside batch loop
2. **Balanced device mapping**: Using `device_map="balanced"` with memory limits
3. **Chunked extraction**: Process 8 layers at a time instead of all 48
4. **Disabled KV cache**: `llm.model.config.use_cache = False`
5. **Immediate CPU transfer**: Activations moved to CPU and saved immediately
6. **Memory cleanup**: `torch.cuda.empty_cache()` after each chunk

### New Code Structure
```python
# GOOD: Model loaded once with optimized settings
llm = LanguageModel(
    model_name,
    device_map="balanced",
    max_memory={"cuda:0": "76GiB", "cuda:1": "76GiB"},
    torch_dtype=torch.bfloat16
)
llm.model.config.use_cache = False

# Process in chunks of 8 layers
for layers_chunk in chunk(layers, 8):
    with torch.inference_mode(), llm.trace(texts) as tracer:
        # Extract only 8 layers at once
        # Immediately save to disk and clear GPU
```

## Infrastructure Improvements

### 1. Multi-GPU Provisioning Fixed
- **Added gpu_count parameter**: Clean API design instead of hacky kwargs
- **Updated broker client**: Explicit parameter in GPUClient.create()
- **Fixed parameter conflict**: Resolved "multiple values for gpu_count" error

### 2. NVIDIA Manufacturer Filtering
- **Added manufacturer field**: To GPUOffer type and GPUQuery interface  
- **Fixed data population**: RunPod provider now sets manufacturer from GPU type info
- **Query interface**: `client.manufacturer == 'Nvidia'` now works

### 3. GPU Selection Features
- **GPU filter**: `--gpu-filter "A100"` for exact GPU type selection
- **GPU count**: `--gpu-count 2` for multi-GPU instances
- **Keep alive**: `--keep-running` prevents auto-cleanup

## Files Modified

### Core Analysis Pipeline
- `run_full_analysis.py`: Restructured to load model once, use chunked extraction
- `extract_activations.py`: Added `extract_activations_optimized()` function

### Broker Infrastructure  
- `broker/broker/types.py`: Added manufacturer field to GPUOffer
- `broker/broker/query.py`: Added manufacturer QueryField
- `broker/broker/client.py`: Added manufacturer property and gpu_count parameter
- `broker/broker/api.py`: Added gpu_count parameter to create() function
- `broker/broker/providers/runpod.py`: Fixed manufacturer data population

### Deployment Scripts
- `deploy_and_analyze.py`: Added gpu_count and gpu_filter parameters

## Current Test Run

**Instance**: `5z40wlao5262ts` (2x A100 80GB)
**SSH**: `root@38.128.233.200:22719`
**Status**: Dependencies installing, will test memory-optimized pipeline
**Key Test Points**:
- Model loads once with balanced device mapping
- Activations extracted in 8-layer chunks (48 layers = 6 chunks)
- Peak memory stays under 76GB per GPU
- No OOM during forward passes

### Monitoring Commands
```bash
# Check analysis progress
bifrost exec 'root@38.128.233.200:22719' 'cd ~/.bifrost/workspace/examples/outlier_features_moe && tail -10 outlier_analysis.log'

# Monitor GPU memory usage
bifrost exec 'root@38.128.233.200:22719' 'nvidia-smi'

# Check if analysis is still running
bifrost exec 'root@38.128.233.200:22719' 'tmux list-sessions | grep outlier-analysis'

# View full log if needed
bifrost exec 'root@38.128.233.200:22719' 'cd ~/.bifrost/workspace/examples/outlier_features_moe && cat outlier_analysis.log'
```

## Success Metrics

1. ✅ **Multi-GPU Provisioning**: 2x A100 deployment working
2. ✅ **NVIDIA Filtering**: Manufacturer filtering implemented 
3. ✅ **Memory Leak Fixed**: Model loaded once vs 4x
4. 🔄 **OOM Prevention**: Testing chunked extraction approach
5. ⏳ **Analysis Completion**: Waiting for successful outlier detection results

## Key File Locations

### Analysis Scripts
- **Main runner**: `examples/outlier_features_moe/run_full_analysis.py`
- **Memory-optimized extractor**: `examples/outlier_features_moe/extract_activations.py` 
- **Deployment wrapper**: `examples/outlier_features_moe/deploy_and_analyze.py`
- **Analysis logic**: `examples/outlier_features_moe/analyze_activations.py`
- **Dataset utilities**: `examples/outlier_features_moe/dataset_utils.py`

### Infrastructure Files
- **GPU Client**: `broker/broker/client.py`
- **GPU Types**: `broker/broker/types.py` 
- **GPU Query**: `broker/broker/query.py`
- **GPU API**: `broker/broker/api.py`
- **RunPod Provider**: `broker/broker/providers/runpod.py`

### Documentation
- **Original handoff**: `examples/outlier_features_moe/HANDOFF_CLAUDE.md`
- **Updated handoff**: `examples/outlier_features_moe/HANDOFF_CLAUDE_UPDATE.md` (this file)
- **OOM analysis**: `examples/outlier_features_moe/oom_analysis.md`

## Quick Management Commands

```bash
# Check all GPU instances
broker instances list

# Manual cleanup (if needed)
broker instances terminate 5z40wlao5262ts

# Check background tasks
jobs  # or ps aux | grep deploy_and_analyze

# Start new optimized analysis (if needed)
python examples/outlier_features_moe/deploy_and_analyze.py --model "Qwen/Qwen3-30B-A3B" --gpu-count 2 --gpu-filter "A100" --min-vram 80 --min-cpu-ram 80 --num-sequences 4 --keep-running

# Test smaller model first (for debugging)
python examples/outlier_features_moe/deploy_and_analyze.py --model "allenai/OLMoE-1B-7B-0125-Instruct" --gpu-count 1 --num-sequences 2
```

## Next Steps

1. **Validate memory optimization**: Confirm no OOM with new approach
2. **Results verification**: Ensure outlier detection methodology still works
3. **Performance optimization**: Fine-tune chunk size if needed
4. **Multi-model testing**: Test on other MoE architectures

## Technical Insights

- **MoE Memory Requirements**: Even with "effective" 11B parameters, actual memory usage ~60GB+ due to routing overhead
- **Device Mapping Critical**: "auto" can create imbalanced loads, "balanced" + limits essential
- **Peak vs Steady Memory**: Forward pass memory >> model weight memory
- **Chunking Essential**: Processing all 48 layers simultaneously causes memory spikes

## Cost Optimization

- **VRAM Estimation**: Updated to account for MoE routing and activation peaks
- **Multi-GPU Strategy**: 2x 80GB more cost-effective than single 180GB+ GPU
- **Memory Efficiency**: Chunked approach enables analysis on smaller GPU configurations

---

**Priority**: Monitor current test run for successful completion without OOM errors. The memory-optimized approach should finally enable successful outlier analysis on the Qwen3-30B-A3B model.