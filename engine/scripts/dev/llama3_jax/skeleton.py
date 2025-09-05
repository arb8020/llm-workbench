#!/usr/bin/env python3
"""
Minimal Llama3 JAX implementation skeleton.

Students should implement llama3_forward() to match HuggingFace Llama3 logits.

Usage:
    python engine/scripts/dev/llama3_jax/skeleton.py
"""

import jax
import jax.numpy as jnp
import einops
import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from jax import Array
from engine.core.utils.weights import load_llama_weights, download_llama_weights

"""
B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
"""
def convert_hf_weights_to_jax_format(hf_weights: Dict[str, Array]) -> Dict[str, Array]:
    """Convert HuggingFace weight names to our expected format."""
    converted = {}
    
    for name, param in hf_weights.items():
        # Keep original names but add some friendly aliases
        converted[name] = param
        
        # Add friendly aliases that our code expects
        if name == 'model.embed_tokens.weight':
            converted['embed_tokens'] = param
        elif name == 'model.norm.weight':
            converted['norm'] = param
        elif name == 'lm_head.weight':
            converted['lm_head'] = param
    
    return converted

def load_and_print_real_weights() -> Dict[str, Array]:
    """Load real LLaMA weights and print some info about them."""
    print("📦 Loading real LLaMA-3.1-8B weights from HuggingFace...")
    
    # Download and load weights
    model_dir = download_llama_weights("meta-llama/Llama-3.1-8B-Instruct")
    weights_obj = load_llama_weights(model_dir)
    
    # Convert to JAX arrays and print info
    hf_weights = {}
    print("\n🔍 Original HuggingFace weight shapes:")
    for name, param in weights_obj.params.items():
        hf_weights[name] = jnp.array(param)
        if any(key in name for key in ['embed_tokens', 'norm', 'lm_head', 'layers.0.']):
            print(f"  {name}: {param.shape}")
    
    # Convert to our expected format
    weights = convert_hf_weights_to_jax_format(hf_weights)
    
    print(f"\n📊 Total parameters: {len(weights_obj.params):,}")
    total_params = sum(p.size for p in weights_obj.params.values())
    print(f"📈 Total parameter count: {total_params:,}")
    
    # Print some converted weight names for debugging
    print(f"\n🔄 Sample converted weight names:")
    sample_names = [k for k in weights.keys() if any(x in k for x in ['embed_tokens', 'norm', 'layers.0.', 'lm_head'])][:5]
    for name in sample_names:
        print(f"  {name}: {weights[name].shape}")
    
    return weights


@dataclass(frozen=True)
class Llama3Config:
    """Configuration for Llama3 model."""
    vocab_size: int = 128256
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8  # for grouped-query attention
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    training: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

@dataclass(frozen=True)
class Llama3State:
    

# rms norm 
# rms norm
# swiglu
# kv cache
# gqa
# rope

# how many blocks in llama 3.1-8b, concise, ignore notes
""" Llama 3.1-8b has 32 transformer blocks/layers."""

def llama3_forward(input_ids: jnp.ndarray, weights: Dict[str, Array], config: Llama3Config) -> jnp.ndarray:
    """Forward pass through Llama3 model"""
    batch_size, seq_len = input_ids.shape
    return jnp.zeros((batch_size, seq_len, config.vocab_size))

if __name__ == "__main__":
    config = Llama3Config(training=True)
    test_input = jnp.array([[1, 2, 3, 4, 5]])  # dummy tokens
    
    # Load actual Llama3 weights
    print("🚀 LLaMA-3.1 JAX Implementation - Loading real weights")
    print("Starting with real weight loading...")
    print()
    
    try:
        real_weights = load_and_print_real_weights()
        
        print("\n🔥 Running LLaMA forward pass with real weights...")
        logits = llama3_forward(test_input, real_weights, config)
        print(f"Output shape: {logits.shape}")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Script completed!")
