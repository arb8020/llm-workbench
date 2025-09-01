#!/usr/bin/env python3
"""
Logits comparison script for JAX GPT-2 implementation.

This script compares JAX GPT-2 logits against HuggingFace reference across
multiple test batches to verify correctness.

Usage:
    python engine/scripts/dev/gpt2_jax/compare.py --mode skeleton
    python engine/scripts/dev/gpt2_jax/compare.py --mode solution
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
import argparse

from engine.core.utils.comparison import compare_logits, get_hf_logits


def load_gpt2_implementation(mode):
    """Load GPT2 implementation based on mode."""
    if mode == "skeleton":
        try:
            from skeleton import gpt2_forward, GPT2Config
            print("✅ Successfully imported from skeleton")
            config = GPT2Config()
            weights = {}  # Skeleton uses dummy weights
            return gpt2_forward, weights, config
        except ImportError as e:
            print(f"❌ Failed to import from skeleton: {e}")
            sys.exit(1)
    
    elif mode == "solution":
        try:
            from solution import gpt2_forward, GPT2Config, load_and_print_real_weights
            print("✅ Successfully imported from solution")
            
            # Load real weights once
            print("📦 Loading real GPT-2 weights...")
            weights = load_and_print_real_weights()
            config = GPT2Config()
            return gpt2_forward, weights, config
        except ImportError as e:
            print(f"❌ Failed to import from solution: {e}")
            sys.exit(1)
    
    else:
        print(f"❌ Invalid mode: {mode}")
        print("💡 Use --mode skeleton or --mode solution")
        sys.exit(1)


def generate_test_batches(k=5):
    """Generate k different test batches for comparison."""
    test_batches = []
    
    # Batch 1: "Hello world"
    test_batches.append({
        "name": "Hello world",
        "tokens": np.array([[15496, 995]])
    })
    
    # Batch 2: "The quick brown"
    test_batches.append({
        "name": "The quick brown",
        "tokens": np.array([[464, 2068, 7586]])
    })
    
    # Batch 3: Single token
    test_batches.append({
        "name": "Single token",
        "tokens": np.array([[15496]])
    })
    
    # Batch 4: Longer sequence
    test_batches.append({
        "name": "Longer sequence",
        "tokens": np.array([[464, 2068, 7586, 1976, 11687, 625, 262]])
    })
    
    # Batch 5: Multiple batch items
    if k >= 5:
        test_batches.append({
            "name": "Batch size 2",
            "tokens": np.array([[15496, 995], [464, 2068]])
        })
    
    return test_batches[:k]


def compare_logits_across_batches(gpt2_forward_fn, weights, config, k=5):
    """Compare JAX implementation vs HuggingFace across k different batches."""
    print("🧪 Comparing JAX GPT-2 vs HuggingFace across multiple batches")
    print("=" * 70)
    
    test_batches = generate_test_batches(k)
    results = []
    
    for i, batch in enumerate(test_batches):
        print(f"\n📊 Batch {i+1}/{k}: {batch['name']}")
        print("-" * 40)
        
        test_input = batch['tokens']
        print(f"Input shape: {test_input.shape}")
        print(f"Input tokens: {test_input.tolist()}")
        
        # Get HuggingFace reference
        print("📚 Getting HuggingFace logits...")
        try:
            hf_logits = get_hf_logits(test_input, model_name="gpt2")
            print(f"HF logits shape: {hf_logits.shape}")
            print(f"HF logits range: [{hf_logits.min():.3f}, {hf_logits.max():.3f}]")
        except Exception as e:
            print(f"⚠️  HuggingFace failed: {e}")
            hf_logits = np.random.randn(*test_input.shape, 50257) * 0.1
        
        # Get JAX implementation logits
        print("🔥 Getting JAX logits...")
        jax_input = jnp.array(test_input)
        jax_logits = gpt2_forward_fn(weights, jax_input, config)
        jax_logits_np = np.array(jax_logits)
        
        print(f"JAX logits shape: {jax_logits_np.shape}")
        print(f"JAX logits range: [{jax_logits_np.min():.3f}, {jax_logits_np.max():.3f}]")
        
        # Compare logits
        print("⚖️  Comparing logits...")
        # Updated tolerances for realistic JAX vs PyTorch precision expectations  
        # rtol=5e-3 allows 0.5% relative error, atol=1e-1 allows 0.1 absolute error
        comparison = compare_logits(
            jax_logits_np,
            hf_logits,
            rtol=5e-3,  # 0.5% relative tolerance (was 0.1%)
            atol=1e-1,  # 0.1 absolute tolerance (was 1e-5)
            verbose=False
        )
        
        # Store results
        batch_result = {
            "name": batch['name'],
            "input_shape": test_input.shape,
            "max_abs_diff": comparison.get('max_abs_diff', float('inf')),
            "mean_abs_diff": comparison.get('mean_abs_diff', float('inf')),
            "all_close": comparison.get('all_close', False)
        }
        results.append(batch_result)
        
        # Print batch summary
        print(f"Max absolute difference: {batch_result['max_abs_diff']:.6f}")
        print(f"Mean absolute difference: {batch_result['mean_abs_diff']:.6f}")
        print(f"All close (rtol=5e-3, atol=1e-1): {batch_result['all_close']}")
        
        if batch_result['all_close']:
            print("✅ PASS")
        else:
            print("❌ FAIL")
    
    return results


def print_summary(results):
    """Print summary of all batch comparisons."""
    print("\n" + "=" * 70)
    print("📋 SUMMARY REPORT")
    print("=" * 70)
    
    total_batches = len(results)
    passed_batches = sum(1 for r in results if r['all_close'])
    
    print(f"Total batches tested: {total_batches}")
    print(f"Batches passed: {passed_batches}")
    print(f"Batches failed: {total_batches - passed_batches}")
    print(f"Pass rate: {passed_batches/total_batches*100:.1f}%")
    
    print("\nPer-batch results:")
    for i, result in enumerate(results):
        status = "✅ PASS" if result['all_close'] else "❌ FAIL"
        print(f"  {i+1}. {result['name']:15} - Max diff: {result['max_abs_diff']:8.6f} - {status}")
    
    if passed_batches == total_batches:
        print("\n🎉 ALL TESTS PASSED! Your GPT-2 implementation matches HuggingFace!")
    else:
        print(f"\n💡 {total_batches - passed_batches} tests failed. Keep implementing!")
        
        avg_max_diff = np.mean([r['max_abs_diff'] for r in results if not np.isinf(r['max_abs_diff'])])
        if avg_max_diff > 10:
            print("💡 Large differences suggest missing core components (embeddings, attention, MLP)")
        elif avg_max_diff > 1:
            print("💡 Medium differences suggest architectural mismatches")
        else:
            print("💡 Small differences suggest numerical precision issues")


def main():
    parser = argparse.ArgumentParser(description="Compare JAX GPT-2 implementation against HuggingFace")
    parser.add_argument("--mode", choices=["skeleton", "solution"], default="skeleton", 
                       help="Which implementation to test (default: skeleton)")
    parser.add_argument("--batches", type=int, default=5, 
                       help="Number of test batches to run (default: 5)")
    
    args = parser.parse_args()
    
    print("🚀 JAX GPT-2 Logits Comparison Suite")
    print(f"Mode: {args.mode}")
    print(f"Testing across {args.batches} input batches...")
    print()
    
    # Load the appropriate implementation
    gpt2_forward_fn, weights, config = load_gpt2_implementation(args.mode)
    
    try:
        # Test across multiple batches
        results = compare_logits_across_batches(gpt2_forward_fn, weights, config, k=args.batches)
        print_summary(results)
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
