#!/usr/bin/env python3
"""
JAX GPT-2 implementation that starts as dummy and can be gradually filled out.

This script begins with dummy outputs and can be incrementally improved until
it matches HuggingFace GPT-2 logits exactly.

Usage:
    python engine/scripts/hello_gpt2_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, NamedTuple, Tuple, Optional
from jax import Array
import sys
from pathlib import Path

# Debug prints for import resolution
print(f"🔍 Current working directory: {Path.cwd()}")
print(f"🔍 Python path: {sys.path[:3]}...")  # First 3 entries
print(f"🔍 Attempting imports...")

try:
    from engine.engine.core.utils.comparison import compare_logits, get_hf_logits
    from engine.engine.core.utils.weights import load_gpt2_weights, download_gpt2_weights
    print("✅ Successfully imported from engine.engine.core.utils")
except ImportError as e:
    print(f"❌ Failed engine.engine.core.utils: {e}")
    try:
        from engine.core.utils.comparison import compare_logits, get_hf_logits
        from engine.core.utils.weights import load_gpt2_weights, download_gpt2_weights
        print("✅ Successfully imported from engine.core.utils")
    except ImportError as e2:
        print(f"❌ Failed engine.core.utils: {e2}")
        print("🚨 Using fallback dummy implementations")
        def compare_logits(*args, **kwargs):
            return {"message": "comparison not available", "all_close": False, "max_abs_diff": float('inf')}
        def get_hf_logits(*args, **kwargs):
            return np.random.randn(1, 2, 50257) * 0.1
        def load_gpt2_weights(*args, **kwargs):
            return None
        def download_gpt2_weights(*args, **kwargs):
            return None


class GPT2Config(NamedTuple):
    """Configuration for GPT-2 model."""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_positions: int = 1024
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True


class GPT2State(NamedTuple):
    """
    Immutable state container for GPT-2 inference.
    
    Pattern inspired by Google's JAX Scaling Book (2025):
    https://jax-ml.github.io/scaling-book/inference/
    
    Used for stateful generation with KV caching. For simple forward passes,
    this may not be necessary - just pass weights, input_ids, config directly.
    """
    input_ids: Array
    position: int
    kv_cache: Optional[Array] = None


def init_dummy_weights(config: GPT2Config) -> Dict[str, Array]:
    """Initialize dummy weights. Replace with real GPT-2 weights later."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)
    
    weights = {
        # Embeddings
        "wte": jax.random.normal(keys[0], (config.vocab_size, config.d_model)) * 0.02,
        "wpe": jax.random.normal(keys[1], (config.n_positions, config.d_model)) * 0.02,
        
        # Final layer norm
        "ln_f.weight": jnp.ones(config.d_model),
        "ln_f.bias": jnp.zeros(config.d_model),
        
        # Language model head
        "lm_head": jax.random.normal(keys[2], (config.d_model, config.vocab_size)) * 0.02,
    }
    
    # Add transformer layers
    for i in range(config.n_layers):
        layer_key = jax.random.split(keys[3 + i % 7], 8)
        
        # Attention weights
        weights[f"h.{i}.attn.c_attn.weight"] = jax.random.normal(layer_key[0], (config.d_model, 3 * config.d_model)) * 0.02
        weights[f"h.{i}.attn.c_attn.bias"] = jnp.zeros(3 * config.d_model)
        weights[f"h.{i}.attn.c_proj.weight"] = jax.random.normal(layer_key[1], (config.d_model, config.d_model)) * 0.02
        weights[f"h.{i}.attn.c_proj.bias"] = jnp.zeros(config.d_model)
        
        # MLP weights
        weights[f"h.{i}.mlp.c_fc.weight"] = jax.random.normal(layer_key[2], (config.d_model, 4 * config.d_model)) * 0.02
        weights[f"h.{i}.mlp.c_fc.bias"] = jnp.zeros(4 * config.d_model)
        weights[f"h.{i}.mlp.c_proj.weight"] = jax.random.normal(layer_key[3], (4 * config.d_model, config.d_model)) * 0.02
        weights[f"h.{i}.mlp.c_proj.bias"] = jnp.zeros(config.d_model)
        
        # Layer norms
        weights[f"h.{i}.ln_1.weight"] = jnp.ones(config.d_model)
        weights[f"h.{i}.ln_1.bias"] = jnp.zeros(config.d_model)
        weights[f"h.{i}.ln_2.weight"] = jnp.ones(config.d_model)
        weights[f"h.{i}.ln_2.bias"] = jnp.zeros(config.d_model)
    
    return weights


# ==============================================================================
# Modular Component Functions
# 
# These pure functions follow the functional programming patterns from:
# - Google JAX Scaling Book: https://jax-ml.github.io/scaling-book/inference/
# - NVIDIA Triton JAX example: https://github.com/triton-inference-server/python_backend/tree/r22.10/examples/jax
# 
# NVIDIA Triton approach is simpler - they focus on the service wrapper:
# ```python
# def AddSub(input_0: jnp.ndarray, input_1: jnp.ndarray) -> List[jnp.ndarray]:
#     output_0 = jnp.add(input_0, input_1)
#     output_1 = jnp.subtract(input_0, input_1)
#     return [output_0, output_1]
# ```
# 
# Our approach combines both: pure functional components (like Triton) 
# with modular architecture patterns (like Scaling Book).
# ==============================================================================

def embedding_lookup(weights: Dict[str, Array], tokens: Array, weight_name: str) -> Array:
    """Look up embeddings for tokens."""
    return weights[weight_name][tokens]


def linear_transform(x: Array, weights: Dict[str, Array], weight_prefix: str) -> Array:
    """Apply linear transformation: x @ W + b"""
    weight = weights[f"{weight_prefix}.weight"] if f"{weight_prefix}.weight" in weights else weights[weight_prefix]
    output = jnp.matmul(x, weight)
    
    # Add bias if it exists
    bias_key = f"{weight_prefix}.bias"
    if bias_key in weights:
        output = output + weights[bias_key]
    
    return output


def layer_norm(x: Array, weights: Dict[str, Array], layer_name: str, 
               epsilon: float = 1e-5) -> Array:
    """Apply layer normalization."""
    gamma = weights[f"{layer_name}.weight"]
    beta = weights[f"{layer_name}.bias"]
    
    # Compute mean and variance over the last dimension
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    
    # Normalize
    normalized = (x - mean) / jnp.sqrt(var + epsilon)
    
    return gamma * normalized + beta


def gelu(x: Array) -> Array:
    """GELU activation function."""
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def multi_head_attention(x: Array, weights: Dict[str, Array], layer_idx: int, 
                        config: GPT2Config) -> Array:
    """Multi-head self-attention."""
    batch_size, seq_len, d_model = x.shape
    
    # Compute Q, K, V
    qkv = linear_transform(x, weights, f"h.{layer_idx}.attn.c_attn")
    q, k, v = jnp.split(qkv, 3, axis=-1)
    
    # Reshape for multi-head attention
    head_dim = d_model // config.n_heads
    q = q.reshape(batch_size, seq_len, config.n_heads, head_dim)
    k = k.reshape(batch_size, seq_len, config.n_heads, head_dim)
    v = v.reshape(batch_size, seq_len, config.n_heads, head_dim)
    
    # Transpose for attention computation: (batch, heads, seq_len, head_dim)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    
    # Scaled dot-product attention
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(head_dim)
    
    # Causal mask
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1) * -1e9
    scores = scores + mask[None, None, :, :]
    
    # Softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply attention to values
    attn_output = jnp.matmul(attn_weights, v)
    
    # Transpose back and reshape
    attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
    attn_output = attn_output.reshape(batch_size, seq_len, d_model)
    
    # Output projection
    output = linear_transform(attn_output, weights, f"h.{layer_idx}.attn.c_proj")
    
    return output


def mlp_block(x: Array, weights: Dict[str, Array], layer_idx: int) -> Array:
    """MLP block with GELU activation."""
    # First linear layer
    hidden = linear_transform(x, weights, f"h.{layer_idx}.mlp.c_fc")
    
    # GELU activation
    hidden = gelu(hidden)
    
    # Second linear layer
    output = linear_transform(hidden, weights, f"h.{layer_idx}.mlp.c_proj")
    
    return output


def transformer_block(x: Array, weights: Dict[str, Array], layer_idx: int, 
                     config: GPT2Config) -> Array:
    """Single transformer block with attention and MLP."""
    # Pre-norm for attention
    attn_input = layer_norm(x, weights, f"h.{layer_idx}.ln_1", config.layer_norm_epsilon)
    
    # Multi-head attention with residual connection
    attn_output = multi_head_attention(attn_input, weights, layer_idx, config)
    x = x + attn_output
    
    # Pre-norm for MLP
    mlp_input = layer_norm(x, weights, f"h.{layer_idx}.ln_2", config.layer_norm_epsilon)
    
    # MLP with residual connection
    mlp_output = mlp_block(mlp_input, weights, layer_idx)
    x = x + mlp_output
    
    return x


# ==============================================================================
# Phase Functions
# ==============================================================================

def gpt2_forward(weights: Dict[str, Array], input_ids: Array, 
                config: GPT2Config) -> Array:
    """
    Forward pass through GPT-2.
    
    Args:
        weights: Model weights dictionary
        input_ids: Input token IDs of shape (batch_size, seq_len)
        config: Model configuration
        
    Returns:
        logits: Logits of shape (batch_size, seq_len, vocab_size)
    """
    batch_size, seq_len = input_ids.shape
    print(f"🔥 Running actual GPT-2 forward pass")
    
    # Token embeddings
    token_embeddings = embedding_lookup(weights, input_ids, "wte")
    
    # Position embeddings
    positions = jnp.arange(seq_len)[None, :]  # (1, seq_len)
    position_embeddings = embedding_lookup(weights, positions, "wpe")
    
    # Combine embeddings
    x = token_embeddings + position_embeddings
    
    # Apply transformer blocks
    for layer_idx in range(config.n_layers):
        x = transformer_block(x, weights, layer_idx, config)
    
    # Final layer normalization
    x = layer_norm(x, weights, "ln_f", config.layer_norm_epsilon)
    
    # Language model head (reuse token embedding weights)
    if "lm_head" in weights:
        logits = linear_transform(x, weights, "lm_head")
    else:
        # GPT-2 shares weights between token embedding and language model head
        logits = jnp.matmul(x, weights["wte"].T)
    
    return logits


def phase1_dummy_logits(batch_size: int, seq_len: int, config: GPT2Config) -> Array:
    """Phase 1: Return completely random logits."""
    print("🎲 Phase 1: Using dummy random logits")
    key = jax.random.PRNGKey(123)
    return jax.random.normal(key, (batch_size, seq_len, config.vocab_size)) * 0.1


def convert_hf_weights_to_jax_format(hf_weights: Dict[str, Array]) -> Dict[str, Array]:
    """Convert HuggingFace weight names to our expected format."""
    converted = {}
    
    for name, param in hf_weights.items():
        # Remove 'transformer.' prefix if present
        clean_name = name.replace('transformer.', '')
        converted[clean_name] = param
        
        # Add friendly aliases that our code expects
        if clean_name == 'wte.weight':
            converted['wte'] = param
        elif clean_name == 'wpe.weight':
            converted['wpe'] = param
    
    return converted


def load_and_print_real_weights() -> Dict[str, Array]:
    """Load real GPT-2 weights and print some info about them."""
    print("📦 Loading real GPT-2 weights from HuggingFace...")
    
    # Download and load weights
    model_dir = download_gpt2_weights("gpt2")
    weights_obj = load_gpt2_weights(model_dir)
    
    # Convert to JAX arrays and print info
    hf_weights = {}
    print("\n🔍 Original HuggingFace weight shapes:")
    for name, param in weights_obj.params.items():
        hf_weights[name] = jnp.array(param)
        if any(key in name for key in ['wte', 'wpe', 'ln_f', 'h.0.']):
            print(f"  {name}: {param.shape}")
    
    # Convert to our expected format
    weights = convert_hf_weights_to_jax_format(hf_weights)
    
    print(f"\n📊 Total parameters: {len(weights_obj.params):,}")
    total_params = sum(p.size for p in weights_obj.params.values())
    print(f"📈 Total parameter count: {total_params:,}")
    
    # Print some converted weight names for debugging
    print(f"\n🔄 Sample converted weight names:")
    sample_names = [k for k in weights.keys() if any(x in k for x in ['wte', 'wpe', 'h.0.', 'ln_f'])][:5]
    for name in sample_names:
        print(f"  {name}: {weights[name].shape}")
    
    return weights


def test_gpt2_comparison():
    """Test our JAX GPT-2 against HuggingFace implementation."""
    print("🧪 Testing JAX GPT-2 vs HuggingFace GPT-2")
    print("=" * 50)
    
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
    
    with jax.default_device(device):
        # Get HuggingFace reference logits
        print("\n📚 Getting HuggingFace reference logits...")
        hf_logits = get_hf_logits(test_input, model_name="gpt2")
        print(f"HF logits shape: {hf_logits.shape}")
        print(f"HF logits range: [{hf_logits.min():.3f}, {hf_logits.max():.3f}]")
        
        # Load real weights and print info
        print("\n📦 Loading real GPT-2 weights...")
        real_weights = load_and_print_real_weights()
        
        # Get our JAX model logits with real weights
        print("\n🔥 Getting JAX model logits with real weights...")
        config = GPT2Config()
        jax_input = jnp.array(test_input)
        jax_logits = gpt2_forward(real_weights, jax_input, config)
        jax_logits_np = np.array(jax_logits)
        
        print(f"JAX logits shape: {jax_logits_np.shape}")
        print(f"JAX logits range: [{jax_logits_np.min():.3f}, {jax_logits_np.max():.3f}]")
        
        # Compare the two
        print("\n📊 Comparing logits...")
        comparison = compare_logits(
            jax_logits_np, 
            hf_logits,
            rtol=1e-3,
            atol=1e-5,
            verbose=True
        )
        
        # Next steps guidance
        print("\n" + "=" * 50)
        if comparison.get('all_close', False):
            print("🎉 SUCCESS! JAX model matches HuggingFace!")
        else:
            print("📋 Next steps to improve accuracy:")
            print("1. Use real_weights instead of dummy_weights in gpt2_forward()")
            print("2. Implement proper GPT-2 forward pass with loaded weights")
            print("3. Real weights are now loaded - just need to use them!")
            
            max_diff = comparison.get('max_abs_diff', float('inf'))
            if max_diff > 10:
                print("💡 Large difference suggests missing core components")
            elif max_diff > 1:
                print("💡 Medium difference suggests architecture mismatch") 
            else:
                print("💡 Small difference suggests numerical precision issues")


if __name__ == "__main__":
    print("🚀 JAX GPT-2 Implementation - Phase by Phase")
    print("Starting with dummy implementation...")
    print()
    
    try:
        test_gpt2_comparison()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Script completed!")
