#!/usr/bin/env python3
"""
External testing script for JAX GPT-2 implementation.

This script imports the GPT-2 implementation and tests it against HuggingFace,
providing a clean separation between the educational skeleton code and testing.

Usage:
    python engine/scripts/dev/compare_jax_gpt2.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

# Debug prints for import resolution
print(f"🔍 Current working directory: {Path.cwd()}")
print(f"🔍 Python path: {sys.path[:3]}...")  # First 3 entries
print(f"🔍 Attempting imports...")

# Import the GPT2 implementation
try:
    from hello_gpt2_jax import (
        GPT2Config, GPT2State, 
        gpt2_forward, init_dummy_weights, load_and_print_real_weights,
        compare_logits, get_hf_logits
    )
    print("✅ Successfully imported GPT2 implementation")
except ImportError as e:
    print(f"❌ Failed to import GPT2 implementation: {e}")
    print("💡 Make sure to run from the engine/scripts/dev directory")
    sys.exit(1)


def test_gpt2_phases():
    """Test different phases of GPT-2 implementation."""
    print("🧪 Testing JAX GPT-2 Implementation Phases")
    print("=" * 60)
    
    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    # Use GPU if available, otherwise CPU
    device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
    print(f"Using device: {device}")
    
    # Test input: "Hello world"
    test_input = np.array([[15496, 995]])  # "Hello world" tokens for GPT-2
    print(f"Test input shape: {test_input.shape}")
    print(f"Test tokens: {test_input.tolist()}")
    
    config = GPT2Config()
    print(f"Model config: {config}")
    
    with jax.default_device(device):
        # Get HuggingFace reference logits
        print("\n📚 Getting HuggingFace reference logits...")
        try:
            hf_logits = get_hf_logits(test_input, model_name="gpt2")
            print(f"HF logits shape: {hf_logits.shape}")
            print(f"HF logits range: [{hf_logits.min():.3f}, {hf_logits.max():.3f}]")
            hf_available = True
        except Exception as e:
            print(f"⚠️  HuggingFace not available: {e}")
            hf_logits = np.random.randn(1, 2, 50257) * 0.1  # dummy fallback
            hf_available = False
        
        # Test with dummy weights (Phase 1)
        print("\n🎲 Phase 1: Testing with dummy weights...")
        dummy_weights = init_dummy_weights(config)
        jax_input = jnp.array(test_input)
        jax_logits = gpt2_forward(dummy_weights, jax_input, config)
        jax_logits_np = np.array(jax_logits)
        
        print(f"JAX logits shape: {jax_logits_np.shape}")
        print(f"JAX logits range: [{jax_logits_np.min():.3f}, {jax_logits_np.max():.3f}]")
        
        # Compare Phase 1 results
        if hf_available:
            print("\n📊 Comparing dummy implementation vs HuggingFace...")
            comparison = compare_logits(
                jax_logits_np, 
                hf_logits,
                rtol=1e-3,
                atol=1e-5,
                verbose=True
            )
            
            print(f"Phase 1 - Max difference: {comparison.get('max_abs_diff', 'N/A'):.6f}")
            print(f"Phase 1 - All close: {comparison.get('all_close', False)}")
        
        # Try to load real weights for future phases
        print("\n📦 Attempting to load real GPT-2 weights...")
        try:
            real_weights = load_and_print_real_weights()
            if real_weights:
                print("✅ Real weights loaded successfully")
                print("💡 Next step: Modify gpt2_forward() to use real_weights")
                print("💡 Implement embedding lookup, attention, and MLP layers")
            else:
                print("⚠️  Real weights not available - using dummy weights only")
        except Exception as e:
            print(f"⚠️  Could not load real weights: {e}")
            print("💡 This is expected if comparison utilities are not available")
        
        # Educational guidance
        print("\n" + "=" * 60)
        print("📚 Educational Notes:")
        print("1. This script tests your GPT-2 implementation")
        print("2. Start with Phase 1 (dummy logits) to verify basic structure")
        print("3. Progress through phases: embeddings → attention → MLP → full model")
        print("4. Each phase should get closer to HuggingFace reference")
        print("5. Final goal: JAX implementation matches HuggingFace exactly")
        
        if not hf_available:
            print("\n⚠️  HuggingFace comparison not available")
            print("💡 Install transformers library for full comparison features")


def benchmark_performance():
    """Simple performance benchmark."""
    print("\n⚡ Performance Benchmark")
    print("-" * 30)
    
    config = GPT2Config()
    dummy_weights = init_dummy_weights(config)
    
    # Warm up
    test_input = jnp.array([[15496, 995]])
    for _ in range(3):
        _ = gpt2_forward(dummy_weights, test_input, config)
    
    # Benchmark
    import time
    start_time = time.time()
    n_runs = 10
    
    for _ in range(n_runs):
        logits = gpt2_forward(dummy_weights, test_input, config)
        logits.block_until_ready()  # Ensure computation completes
    
    elapsed = time.time() - start_time
    print(f"Average time per forward pass: {elapsed/n_runs*1000:.2f}ms")
    print(f"Total time for {n_runs} runs: {elapsed:.3f}s")


if __name__ == "__main__":
    print("🚀 JAX GPT-2 External Testing Suite")
    print("=" * 60)
    
    try:
        test_gpt2_phases()
        benchmark_performance()
        
        print("\n✅ Testing completed successfully!")
        print("💡 Modify hello_gpt2_jax.py to implement different phases")
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()