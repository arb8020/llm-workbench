#!/usr/bin/env python3
"""
Debug script to isolate the numerical precision differences between JAX and HuggingFace GPT-2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path

# Add the path for imports
sys.path.insert(0, str(Path("engine/scripts/dev/gpt2_jax")))

try:
    from engine.engine.core.utils.comparison import get_hf_logits
    from solution import gpt2_forward, GPT2Config, load_and_print_real_weights
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def debug_layer_outputs():
    """Debug individual layer outputs to find where the difference starts."""
    
    # Load weights
    print("📦 Loading weights...")
    weights = load_and_print_real_weights()
    config = GPT2Config()
    
    # Simple test input
    input_ids = jnp.array([[15496, 995]])  # "Hello world"
    print(f"Input: {input_ids}")
    
    # Get HF reference
    hf_logits = get_hf_logits(np.array(input_ids), "gpt2")
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"HF first few logits: {hf_logits[0, 0, :5]}")
    
    # Get JAX output
    jax_logits = gpt2_forward(weights, input_ids, config)
    print(f"JAX logits shape: {jax_logits.shape}")
    print(f"JAX first few logits: {jax_logits[0, 0, :5]}")
    
    # Compare first few logits
    diff = np.abs(np.array(jax_logits) - hf_logits)
    print(f"Max diff in first 10 logits: {diff[0, 0, :10].max()}")
    print(f"Diff in first few logits: {diff[0, 0, :5]}")

if __name__ == "__main__":
    debug_layer_outputs()