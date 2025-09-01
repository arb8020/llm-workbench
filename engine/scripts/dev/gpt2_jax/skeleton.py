#!/usr/bin/env python3
"""
Minimal GPT-2 JAX implementation skeleton.

Students should implement gpt2_forward() to match HuggingFace GPT-2 logits.

Usage:
    python engine/scripts/dev/hello_gpt2_jax_skeleton.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from jax import Array


@dataclass(frozen=True)
class GPT2Config:
    """Configuration for GPT-2 model."""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_positions: int = 1024
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True
    freqs_cis: Optional[Array] = None  # Rotary position embeddings (for RoPE-enabled variants)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"


def gpt2_forward(weights: Dict[str, Array], input_ids: jnp.ndarray, config: GPT2Config) -> jnp.ndarray:
    """
    GPT-2 forward pass function.
    
    Args:
        weights: Model weights dictionary
        input_ids: Token IDs of shape (batch_size, seq_len)
        config: Model configuration
        
    Returns:
        logits: Output logits of shape (batch_size, seq_len, vocab_size)
    """
    batch_size, seq_len = input_ids.shape
    
    # TODO: Implement GPT-2 forward pass
    # This should eventually match HuggingFace GPT-2 exactly
    # Use weights and config to build the actual model
    
    # If using rotary embeddings, assert they're provided
    # if config.freqs_cis is not None:
    #     assert config.freqs_cis is not None, "freqs_cis must be provided in config for RoPE-enabled models"
    
    # Placeholder: return random logits
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (batch_size, seq_len, config.vocab_size)) * 0.1


if __name__ == "__main__":
    # Simple test
    config = GPT2Config()
    dummy_weights = {}  # Skeleton uses dummy weights
    test_input = jnp.array([[15496, 995]])  # "Hello world" tokens
    logits = gpt2_forward(dummy_weights, test_input, config)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")