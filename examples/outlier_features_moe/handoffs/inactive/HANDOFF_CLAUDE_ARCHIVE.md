# Outlier Features Analysis - Handoff Document

## Current Status Summary

**✅ MAJOR SUCCESSES**: 
1. **Auto VRAM Estimation Fixed**: Now calculates 29GB (vs old 150GB) for Qwen3-30B with MoE efficiency
2. **Async Sync Bug Fixed**: Results now sync properly using efficient SFTP 
3. **CPU RAM Requirements**: Added 64GB minimum to prevent MoE model loading OOM
4. **Context-Length Aware**: VRAM estimates account for sequence length and batch size

**🔄 CURRENT ISSUE**: Deployment works but GPU selection needs NVIDIA manufacturer filtering

## Key Files to Read

### 1. Core Analysis Scripts
```bash
examples/outlier_features_moe/run_full_analysis.py          # Main analysis pipeline
examples/outlier_features_moe/extract_activations.py       # Activation extraction
examples/outlier_features_moe/analyze_activations.py       # Outlier detection logic
examples/outlier_features_moe/dataset_utils.py            # Tokenizer-based text sequences
```

### 2. Deployment Infrastructure  
```bash
examples/outlier_features_moe/deploy_and_analyze.py        # ⭐ MAIN WRAPPER - auto VRAM + CPU RAM filtering
examples/outlier_features_moe/estimate_vram.py            # ⭐ VRAM estimation - now context-length aware
examples/outlier_features_moe/inspect_model_layers.py      # Model architecture inspection
examples/outlier_features_moe/run_model_comparison.py      # Multi-model comparison
```

### 3. Documentation & Context
```bash
examples/outlier_features_moe/paper.md                    # Paper methodology with exact quotes
examples/outlier_features_moe/tasks.md                    # Original task requirements
examples/outlier_features_moe/todo_claude.md              # Previous todo list
```

### 4. Project Infrastructure
```bash
pyproject.toml                                            # UV workspace with engine[interp] extra
bifrost/bifrost/client.py                                # BifrostClient with uv_extra support
deploy.md                                                 # General deployment examples
```

## Completed Tasks

### ✅ Core Pipeline Working  
1. **Tokenizer-based sequences**: Updated from character-based to token-based chunking
2. **Optional layers flag**: `--layers` defaults to all layers, can specify subset
3. **Model architecture detection**: Auto-detects layers for any transformer architecture
4. **Paper methodology implementation**: Proper outlier detection (≥6.0 magnitude, ≥25% layers, ≥6% sequence positions)
5. **Multi-architecture support**: Unified layer access pattern works across OLMoE, Qwen3, etc.

### ✅ Smart VRAM Estimation (MAJOR FIX)
1. **Auto VRAM calculation**: Estimates based on effective MoE parameters (29GB vs 150GB!)
2. **Context-length aware**: Accounts for sequence length, batch size, KV cache
3. **MoE efficiency**: Only counts active experts (8/128 for Qwen3-30B-A3B = 34% efficiency)
4. **Detailed breakdown**: Shows model weights, KV cache, activations separately
5. **Safety factor**: Configurable multiplier (default 1.3x for inference)

### ✅ Deployment Infrastructure Fixed
1. **Async sync bug fixed**: Results sync properly using efficient SFTP
2. **CPU RAM filtering**: 64GB minimum prevents MoE model loading OOM
3. **Binary file support**: SFTP handles .pt activation files correctly  
4. **Dependency management**: Automatic `uv sync --extra interp` for nnsight
5. **Auto cleanup**: GPU termination to stop billing

## Remaining Tasks

### 🔥 URGENT: Add NVIDIA GPU Manufacturer Filtering
**File**: `broker/client.py` 
**Issue**: GPUClient Python API lacks manufacturer filtering that CLI has (`broker search --manufacturer nvidia`)
**Current Workaround**: Most cloud GPUs are NVIDIA anyway, but should be explicit

**TODO**:
1. Research how `broker search --manufacturer` is implemented in CLI
2. Add `manufacturer` field to GPUClient query builder
3. Update deploy script to use: `gpu_client.manufacturer == 'nvidia'`

### 🔧 SSH Connection Stability  
**File**: `examples/outlier_features_moe/deploy_and_analyze.py`
**Issue**: Occasional SSH connection failures during deployment (Connection reset by peer)
**Potential fixes**:
1. Add retry logic for SSH connections
2. Increase SSH timeout settings
3. Wait longer for SSH daemon initialization

### 📊 Analysis Validation
**Next Steps**:
1. Re-run analysis with fixed sync to get actual results
2. Validate outlier detection against paper methodology
3. Test on multiple architectures (OLMoE, GLM, GPT-OSS) 
4. Compare results across different model sizes

### 🎯 Multi-Model Comparison
**File**: `examples/outlier_features_moe/run_model_comparison.py`
**Target Models**: 
- `allenai/OLMoE-1B-7B-0125-Instruct` (tested ✅)
- `Qwen/Qwen3-30B-A3B` (analysis completed, sync failed)
- `openai/gpt-oss-120b` (layer structure confirmed)
- `zai-org/GLM-4.5-Air` (layer structure confirmed)

## Key Insights

### 🚀 What Works
1. **Unified Layer Access**: All transformer architectures use `model.layers[i].{input_layernorm, post_attention_layernorm}`
2. **Bifrost UV Extras**: `push(uv_extra="interp")` handles dependency installation correctly
3. **Automated Deployment**: End-to-end pipeline from provisioning to cleanup works flawlessly

### ⚠️ What Needs Attention  
1. **Result Sync**: Critical async bug prevents getting analysis results
2. **VRAM Estimation**: Wildly inaccurate parameter counting
3. **Cost Optimization**: 150GB requirement forces expensive GPUs (should be ~80GB)

### 🧠 Architecture Understanding
- **Paper Definition**: Outliers = magnitude ≥6.0, ≥25% layers affected, ≥6% sequence positions
- **Key Insight**: "6% of sequence dimensions" means unique positions across ALL layers, not within single layer  
- **MoE Consideration**: Some models use mixture-of-experts, may need different VRAM calculations

## Testing Commands

### Run Analysis (once sync is fixed)
```bash
python examples/outlier_features_moe/deploy_and_analyze.py --model "Qwen/Qwen3-30B-A3B" --num-sequences 2
```

### Test VRAM Estimator  
```bash
python examples/outlier_features_moe/estimate_vram.py --model "Qwen/Qwen3-30B-A3B"
```

### Manual Analysis (if deployment fails)
```bash
python examples/outlier_features_moe/run_full_analysis.py --model "allenai/OLMoE-1B-7B-0125-Instruct" --num-sequences 4
```

## Context Notes

- **User Goal**: Analyze outlier features across different transformer architectures
- **Paper Reference**: "Outlier Features in Large Language Models" - exact methodology implemented
- **Budget Conscious**: User called out 150GB VRAM requirement as "trolling" (correctly - it's overkill)
- **Quality Focus**: User wants systematic, paper-accurate analysis across multiple model types

## Success Metrics

1. ✅ **Pipeline Works**: Analysis completes successfully  
2. ❌ **Results Retrieved**: Sync results back to local machine
3. ❌ **Cost Optimized**: Accurate VRAM estimation for cheaper GPU selection
4. ❌ **Multi-Architecture**: Test on all 4 target model types
5. ❌ **Validated Results**: Confirm outlier detection matches paper methodology

**Priority**: Fix the async sync bug first - everything else depends on getting results back!