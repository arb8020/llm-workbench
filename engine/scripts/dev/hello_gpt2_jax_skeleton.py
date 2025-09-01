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
from typing import Dict

def gpt2_forward(input_ids: jnp.ndarray) -> jnp.ndarray:
    """
    GPT-2 forward pass function.
    
    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        
    Returns:
        logits: Output logits of shape (batch_size, seq_len, 50257)
    """
    batch_size, seq_len = input_ids.shape
    vocab_size = 50257
    
    # TODO: Implement GPT-2 forward pass
    # This should eventually match HuggingFace GPT-2 exactly
    
    # Placeholder: return random logits
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (batch_size, seq_len, vocab_size)) * 0.1


if __name__ == "__main__":
    # Simple test
    test_input = jnp.array([[15496, 995]])  # "Hello world" tokens
    logits = gpt2_forward(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")