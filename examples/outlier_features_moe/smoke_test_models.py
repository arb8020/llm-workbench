#!/usr/bin/env python3
"""
Smoke test script to inspect model architectures without loading weights.
Goal: Print all layer structures to understand residual stream access points.

Usage:
    python smoke_test_models.py
"""

from transformers import AutoConfig, AutoModel
import torch
from typing import Dict, List

# Target models for outlier analysis - final 4 architectures
MODELS = [
    "allenai/OLMoE-1B-7B-0125-Instruct",     # olmoe
    "Qwen/Qwen3-30B-A3B",                    # qwen3_moe  
    "openai/gpt-oss-120b",                   # gpt_oss
    "zai-org/GLM-4.5-Air",                  # glm4_moe (using Air for efficiency)
]

def print_model_structure(model_name: str, max_depth: int = 3):
    """Print the structure of a model without loading weights."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Load config only (no weights)
        print("Loading config...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Print basic config info
        print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
        print(f"Architecture: {config.architectures[0] if hasattr(config, 'architectures') else 'unknown'}")
        print(f"Hidden size: {getattr(config, 'hidden_size', 'unknown')}")
        print(f"Num layers: {getattr(config, 'num_hidden_layers', 'unknown')}")
        print(f"Num attention heads: {getattr(config, 'num_attention_heads', 'unknown')}")
        
        # Try to load model structure (config only, no weights)
        print(f"\nLoading model structure (no weights)...")
        model = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="meta"  # This loads structure without weights
        )
        
        print(f"\nModel structure:")
        print_module_tree(model, max_depth=max_depth)
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        print(f"Error type: {type(e).__name__}")

def print_module_tree(module, prefix="", max_depth=3, current_depth=0):
    """Recursively print the module tree structure."""
    if current_depth >= max_depth:
        return
        
    for name, child in module.named_children():
        print(f"{prefix}├── {name}: {type(child).__name__}")
        
        # Print some key attributes if they exist
        if hasattr(child, 'weight') and child.weight is not None:
            print(f"{prefix}│   └── weight: {child.weight.shape}")
        
        # Recurse into children
        if current_depth < max_depth - 1:
            child_prefix = prefix + "│   "
            print_module_tree(child, child_prefix, max_depth, current_depth + 1)

def find_layer_patterns(model_name: str):
    """Try to find common layer access patterns."""
    try:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="meta"
        )
        
        print(f"\n🔍 Layer pattern analysis for {model_name}:")
        
        # Look for common layer container names
        layer_containers = []
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['layer', 'block', 'transformer']):
                if hasattr(module, '__len__') or 'ModuleList' in str(type(module)):
                    layer_containers.append((name, type(module).__name__, len(list(module.children()))))
        
        for name, type_name, child_count in layer_containers:
            print(f"  {name}: {type_name} (length: {child_count})")
        
        # Look for normalization layers that might be our target
        norm_layers = []
        for name, module in model.named_modules():
            if any(norm in type(module).__name__.lower() for norm in ['norm', 'layernorm']):
                norm_layers.append((name, type(module).__name__))
        
        print(f"\n🎯 Potential residual stream access points:")
        for name, type_name in norm_layers[:20]:  # Limit output
            print(f"  {name}: {type_name}")
            
    except Exception as e:
        print(f"❌ Pattern analysis failed for {model_name}: {e}")

def main():
    print("🔬 SMOKE TEST: Model Architecture Analysis")
    print("Goal: Find residual stream access points across different model architectures")
    print(f"Testing {len(MODELS)} model architectures...")
    
    for model_name in MODELS:
        try:
            print_model_structure(model_name)
            find_layer_patterns(model_name)
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error with {model_name}: {e}")
            continue
    
    print(f"\n✅ Smoke test complete!")
    print("\n📋 Next steps:")
    print("1. Review layer naming patterns above")
    print("2. Identify residual stream access points for each architecture")
    print("3. Create mapping table: {model_type: (pre_attn_path, pre_mlp_path)}")

if __name__ == "__main__":
    main()