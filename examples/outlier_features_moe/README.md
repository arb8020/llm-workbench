# Outlier Features Analysis for Mixture-of-Experts Models

A systematic analysis pipeline for detecting outlier features in large MoE language models, implementing the methodology from "Outlier Features in Large Language Models" research.

## 🎯 Quick Start

**Run analysis on a model:**
```bash
python deploy_and_analyze.py \
    --model "Qwen/Qwen3-30B-A3B" \
    --gpu-count 2 \
    --gpu-filter "A100" \
    --min-vram 80 \
    --num-sequences 4
```

**Local testing (smaller model):**
```bash
python run_full_analysis.py \
    --model "allenai/OLMoE-1B-7B-0125-Instruct" \
    --num-sequences 2
```

## 📊 Results Summary

### Completed Analysis: Qwen3-30B-A3B

- **Total Parameters**: 30.5B (3.3B active per token)
- **Systematic Outliers Found**: 24 features
- **Phase Shift Status**: Above phase shift (systematic outliers present)

**Top Outlier Features:**

| Feature | Max Magnitude | % Layers | % Sequence Positions | Occurrences |
|---------|---------------|----------|---------------------|-------------|
| 1461    | 33.75        | 35.4%    | 46.0%              | 5,894       |
| 1475    | 28.25        | 33.3%    | 38.6%              | 7,136       |
| 571     | 18.12        | 35.4%    | 75.4%              | 24,975      |

## 🏗 Architecture

### Core Analysis Pipeline
- `run_full_analysis.py` - Main analysis orchestrator
- `extract_activations.py` - Memory-optimized activation extraction
- `analyze_activations.py` - Outlier detection with paper-accurate methodology
- `deploy_and_analyze.py` - Automated cloud GPU deployment

### Utilities (in scripts/)
- `scripts/dataset_utils.py` - Tokenizer-based sequence generation
- `scripts/estimate_vram.py` - VRAM estimation for MoE models

## 🔬 Methodology

Implements exact criteria from outlier features research:
- **Magnitude Threshold**: ≥6.0 activation magnitude
- **Layer Coverage**: ≥25% of transformer layers affected
- **Sequence Coverage**: ≥6% of sequence positions affected

**Memory-Optimized Processing:**
- Single model load per analysis (not per batch)
- Chunked layer extraction (8 layers at a time)
- Balanced multi-GPU device mapping
- Immediate CPU transfer and disk saves

## 📋 Target Models

### Analyzed
- ✅ **Qwen/Qwen3-30B-A3B** (30.5B total, 3.3B active, 10.8% ratio) - **24 systematic outliers found**

### Currently Running
- 🔄 **openai/gpt-oss-120b** (117B total, 5.1B active, 4.4% ratio)

### Planned Analysis - Complete MoE Survey

**Small Scale:**
- **allenai/OLMoE-1B-7B-0125** (7B total, 1B active, 14.3% ratio)

**Medium Scale:**  
- **zai-org/GLM-4.5-Air** (106B total, 12B active, 11.3% ratio)

**Latest 2025 Models:**
- **meta-llama/Llama-4-Maverick** (400B total, 17B active, 4.3% ratio) - *Meta's first MoE*
- **meta-llama/Llama-4-Scout** (109B total, 17B active, 15.6% ratio) - *10M context*
- **deepseek-ai/DeepSeek-V3.1** (671B total, 37B active, 5.5% ratio) - *Aug 2025 hybrid*
- **deepseek-ai/DeepSeek-V3-0324** (671B total, 37B active, 5.5% ratio) - *Improved reasoning*

**Extreme Scale:**
- **moonshotai/Kimi-K2-Instruct** (1000B total, 32B active, 3.2% ratio)

## 🚀 Key Optimizations

### Repository Size Reduction
- **Before**: 1.4GB (219 × 8.4MB .pt files)
- **After**: 240KB (99.86% reduction via `git filter-repo`)
- **Impact**: Deployment in seconds vs 15+ minutes

### Multi-GPU Support
- Automatic 2×A100 provisioning with balanced memory limits
- NVIDIA manufacturer filtering
- Real-time memory monitoring

### Result Sync Optimization
- Downloads only metadata and logs (not large .pt files)
- Saves 800MB+ of bandwidth per analysis

## 📁 Directory Structure

```
examples/outlier_features_moe/
├── README.md                    # This file
├── run_full_analysis.py         # Main analysis pipeline
├── deploy_and_analyze.py        # Cloud deployment wrapper
├── extract_activations.py       # Activation extraction
├── analyze_activations.py       # Outlier detection
├── layer_analysis_results.json # Model layer structures
├── remote_results/             # Synced analysis results
├── handoffs/                   # Development handoff docs
├── docs/                       # Research papers & analysis
└── scripts/                    # Utility scripts
    ├── dataset_utils.py        # Dataset utilities
    ├── estimate_vram.py        # VRAM estimation
    └── ...                     # Other utility scripts
```

## 🔧 Requirements

- Python 3.8+ with `uv` package manager
- CUDA-capable GPUs (tested on 2×A100 80GB)
- Cloud GPU access via broker/bifrost (for large models)

**Install dependencies:**
```bash
uv sync --extra interp
```

## 💾 Results Format

Analysis generates:
- **Systematic outlier summary** with feature dimensions, magnitudes, and coverage
- **Cross-batch consistency** tracking (features appearing in multiple input batches)
- **Layer distribution patterns** (early vs late layer concentration)
- **Sequence position analysis** (% of positions affected per feature)

## 🎯 Research Applications

- **Phase transition analysis**: Determine if models are above/below outlier phase shift
- **Architecture comparison**: Compare outlier patterns across MoE designs  
- **Scale effects**: Study how outliers change with parameter count
- **Efficiency analysis**: Relate outliers to active/total parameter ratios

## 🐛 Known Limitations

- Requires substantial VRAM (30-80GB+ for large models)
- Analysis runtime: 10-30 minutes depending on model size
- Remote sync occasionally fails (results remain on remote instance)

## 📚 References

Based on methodology from "Outlier Features in Large Language Models" research, with optimizations for Mixture-of-Experts architectures and large-scale analysis.