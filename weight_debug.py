#!/usr/bin/env python3
"""Quick smoke test to debug weight loading issue."""

from engine.core.utils.weights import load_and_print_gpt2_weights_jax

def main():
    print("🔍 Loading GPT-2 weights...")
    weights = load_and_print_gpt2_weights_jax()
    
    print(f"\n📊 Loaded {len(weights)} weight tensors")
    print(f"📋 Available keys: {sorted(list(weights.keys())[:10])}{'...' if len(weights) > 10 else ''}")
    
    # Check if we have the expected keys vs what we actually get
    expected_keys = ['wte', 'wpe', 'ln_f.weight', 'ln_f.bias']
    actual_keys = list(weights.keys())
    
    print(f"\n🎯 Looking for expected keys:")
    for key in expected_keys:
        if key in weights:
            print(f"  ✅ {key}: {weights[key].shape}")
        else:
            # Look for similar keys
            similar = [k for k in actual_keys if key.split('.')[-1] in k or key in k]
            print(f"  ❌ {key}: not found, similar: {similar[:3]}")

if __name__ == "__main__":
    main()