# Key Entropix Files Needed

Based on the entropix repository structure, we need these key files:

## Core Implementation Files:
1. **entropix/model.py** - Main JAX transformer implementation
   - Contains: attention, feed_forward, xfmr, KV cache logic
   - Key features: RoPE, RMS norm, sharding

2. **entropix/cache.py** - KV cache implementation (if exists)
   - KV cache data structures and operations

3. **entropix/sampler.py** - Entropy-based sampling logic
   - Sampling strategies and entropy calculations

4. **entropix/config.py** - Model configuration
   - Model parameters and hyperparameters

5. **entropix/weights.py** - Weight loading utilities
   - Functions to load/convert model weights

## Files to create/modify in our implementation:
- `solution_entropix_jax.py` - Pure JAX implementation with KV cache
- `solution_entropix_torch.py` - Pure PyTorch implementation (optional)
- `compare_entropix.py` - Updated comparison with multi-token sampling
- `entropix_faithful_impl.py` - Most faithful reproduction

## Key features to implement:
1. ✅ RoPE (already have)
2. ✅ RMS norm (already have) 
3. ✅ Grouped-query attention (have basic version)
4. ❌ KV cache (missing - critical)
5. ❌ Multi-token sampling (missing)
6. ❌ JAX sharding (missing)
7. ❌ Entropy-based sampling (missing)