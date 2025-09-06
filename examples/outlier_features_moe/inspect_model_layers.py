#!/usr/bin/env python3
"""
Local script to inspect model layer structure from HuggingFace safetensors files.
Much faster than loading models - just fetches the weight mapping to understand layer names.

Usage:
    python inspect_model_layers.py
"""

import requests
import json
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class LayerInfo:
    """Information about a model's layer structure."""
    model_name: str
    model_type: str
    architecture: str
    num_layers: int
    layer_access_pattern: str
    pre_attention_path: str
    pre_mlp_path: str
    sample_layer_names: List[str]

# Target models for outlier analysis
MODELS = [
    "allenai/OLMoE-1B-7B-0125-Instruct",     # olmoe
    "Qwen/Qwen3-30B-A3B",                    # qwen3_moe  
    "openai/gpt-oss-120b",                   # gpt_oss
    "zai-org/GLM-4.5-Air",                  # glm4_moe
]

def fetch_config(model_name: str) -> Dict:
    """Fetch model config.json from HuggingFace."""
    url = f"https://huggingface.co/{model_name}/raw/main/config.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to fetch config for {model_name}: {e}")
        return {}

def fetch_safetensors_index(model_name: str) -> Dict:
    """Fetch model.safetensors.index.json from HuggingFace."""
    url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors.index.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to fetch safetensors index for {model_name}: {e}")
        return {}

def analyze_layer_structure(model_name: str) -> LayerInfo:
    """Analyze layer structure from config and safetensors index."""
    print(f"\n🔍 Analyzing {model_name}...")
    
    # Fetch config
    config = fetch_config(model_name)
    if not config:
        return None
    
    model_type = config.get('model_type', 'unknown')
    architecture = config.get('architectures', ['unknown'])[0] if config.get('architectures') else 'unknown'
    num_layers = config.get('num_hidden_layers', 0)
    
    print(f"  Model type: {model_type}")
    print(f"  Architecture: {architecture}")  
    print(f"  Layers: {num_layers}")
    
    # Fetch safetensors index for layer names
    safetensors_index = fetch_safetensors_index(model_name)
    if not safetensors_index:
        print("  ⚠️ Could not fetch layer structure")
        return LayerInfo(
            model_name=model_name,
            model_type=model_type,
            architecture=architecture,
            num_layers=num_layers,
            layer_access_pattern="unknown",
            pre_attention_path="unknown",
            pre_mlp_path="unknown",
            sample_layer_names=[]
        )
    
    # Analyze weight map for layer patterns
    weight_map = safetensors_index.get('weight_map', {})
    layer_names = set()
    
    # Look for layer normalization patterns
    input_layernorm_pattern = None
    post_attention_layernorm_pattern = None
    
    for param_name in weight_map.keys():
        if 'layer' in param_name:
            layer_names.add(param_name)
            
        # Detect layer access patterns
        if 'input_layernorm' in param_name and input_layernorm_pattern is None:
            # Extract pattern like "model.layers.{i}.input_layernorm"
            parts = param_name.split('.')
            if 'layers' in parts:
                layer_idx = parts.index('layers')
                pattern_parts = parts[:layer_idx+2] + ['input_layernorm']  # model.layers.{i}.input_layernorm
                input_layernorm_pattern = '.'.join(pattern_parts)
        
        if 'post_attention_layernorm' in param_name and post_attention_layernorm_pattern is None:
            parts = param_name.split('.')
            if 'layers' in parts:
                layer_idx = parts.index('layers')
                pattern_parts = parts[:layer_idx+2] + ['post_attention_layernorm']
                post_attention_layernorm_pattern = '.'.join(pattern_parts)
    
    # Determine layer access pattern
    sample_layer_names = sorted(list(layer_names))[:10]  # First 10 for inspection
    
    if input_layernorm_pattern and post_attention_layernorm_pattern:
        layer_access_pattern = "model.layers[i]"
        # Normalize pattern by replacing any number with {i}
        import re
        pre_attention_path = re.sub(r'\.(\d+)\.', '.{i}.', input_layernorm_pattern)
        pre_mlp_path = re.sub(r'\.(\d+)\.', '.{i}.', post_attention_layernorm_pattern)
        print(f"  ✅ Pattern: {layer_access_pattern}")
        print(f"  ✅ Pre-attention: {pre_attention_path}")
        print(f"  ✅ Pre-MLP: {pre_mlp_path}")
    else:
        layer_access_pattern = "unknown"
        pre_attention_path = "unknown"
        pre_mlp_path = "unknown"
        print(f"  ❓ Could not determine layer access pattern")
        print(f"     Sample layer names: {sample_layer_names[:3]}")
    
    return LayerInfo(
        model_name=model_name,
        model_type=model_type,
        architecture=architecture,
        num_layers=num_layers,
        layer_access_pattern=layer_access_pattern,
        pre_attention_path=pre_attention_path,
        pre_mlp_path=pre_mlp_path,
        sample_layer_names=sample_layer_names
    )

def generate_summary_table(layer_infos: List[LayerInfo]):
    """Generate a summary table of all models."""
    print(f"\n{'='*80}")
    print("LAYER STRUCTURE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<30} {'Type':<15} {'Layers':<8} {'Pre-Attention Access':<40}")
    print("-" * 93)
    
    for info in layer_infos:
        if info:
            print(f"{info.model_name.split('/')[-1]:<30} "
                  f"{info.model_type:<15} "
                  f"{info.num_layers:<8} "
                  f"{info.pre_attention_path:<40}")
    
    print("-" * 93)
    
    # Check for consistency
    unique_patterns = set()
    unique_pre_attention = set()
    unique_pre_mlp = set()
    
    for info in layer_infos:
        if info and info.layer_access_pattern != "unknown":
            unique_patterns.add(info.layer_access_pattern)
            unique_pre_attention.add(info.pre_attention_path.replace('{i}', 'X'))
            unique_pre_mlp.add(info.pre_mlp_path.replace('{i}', 'X'))
    
    print(f"\n🎯 CONSISTENCY ANALYSIS:")
    print(f"   Unique layer patterns: {len(unique_patterns)} → {list(unique_patterns)}")
    print(f"   Unique pre-attention patterns: {len(unique_pre_attention)} → {list(unique_pre_attention)}")
    print(f"   Unique pre-MLP patterns: {len(unique_pre_mlp)} → {list(unique_pre_mlp)}")
    
    if len(unique_patterns) == 1 and len(unique_pre_attention) == 1:
        print(f"   ✅ ALL MODELS USE SAME PATTERN - No adapter needed!")
    else:
        print(f"   ⚠️  Different patterns detected - Adapter system needed")

def save_results(layer_infos: List[LayerInfo], filename: str = "layer_analysis_results.json"):
    """Save results to JSON file for later use."""
    results = []
    for info in layer_infos:
        if info:
            results.append({
                "model_name": info.model_name,
                "model_type": info.model_type,
                "architecture": info.architecture,
                "num_layers": info.num_layers,
                "layer_access_pattern": info.layer_access_pattern,
                "pre_attention_path": info.pre_attention_path,
                "pre_mlp_path": info.pre_mlp_path,
                "sample_layer_names": info.sample_layer_names[:5]  # Limit sample size
            })
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")

def main():
    """Main analysis pipeline."""
    print("🔬 LOCAL MODEL LAYER ANALYSIS")
    print("Fetching layer structure from HuggingFace safetensors files...")
    print(f"Analyzing {len(MODELS)} models...")
    
    layer_infos = []
    
    for model_name in MODELS:
        try:
            info = analyze_layer_structure(model_name)
            layer_infos.append(info)
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error with {model_name}: {e}")
            layer_infos.append(None)
    
    # Generate summary
    generate_summary_table([info for info in layer_infos if info])
    
    # Save results
    save_results([info for info in layer_infos if info])
    
    print(f"\n✅ Analysis complete!")
    print("\n📋 Next steps:")
    print("1. Use layer_analysis_results.json to configure model adapters")
    print("2. Update extract_activations.py with confirmed layer patterns")
    print("3. Test outlier analysis across all 4 model types")

if __name__ == "__main__":
    main()