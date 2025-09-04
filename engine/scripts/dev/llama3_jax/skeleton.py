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
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from jax import Array

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


def llama3_forward(input_ids: jnp.ndarray, weights: Dict[str, Array], config: Llama3Config) -> jnp.ndarray:
    """Forward pass through Llama3 model"""
    # TODO: Implement the full Llama3 forward pass
    # 1. Token embeddings (no positional embeddings in Llama3)
    # 2. Precompute RoPE frequencies
    # 3. Pass through transformer blocks
    # 4. Final RMS norm
    # 5. Language modeling head (often tied to input embeddings)
    
    batch_size, seq_len = input_ids.shape
    
    # Placeholder return - replace with actual implementation
    return jnp.zeros((batch_size, seq_len, config.vocab_size))


if __name__ == "__main__":
    config = Llama3Config(training=True)
    test_input = jnp.array([[1, 2, 3, 4, 5]])  # dummy tokens
    
    # TODO: Load actual Llama3 weights
    dummy_weights = {}
    
    logits = llama3_forward(test_input, dummy_weights, config)
    print(f"Output shape: {logits.shape}")
