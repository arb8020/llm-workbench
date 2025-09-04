#!/usr/bin/env python3
"""
JAX Llama3 implementation that matches HuggingFace Llama3 logits.

This implementation demonstrates critical architectural differences between
GPT-2 and Llama3, including:

LLAMA3 ARCHITECTURAL FEATURES:
1. RMS Normalization: Instead of LayerNorm, uses Root Mean Square normalization
   - Simpler than LayerNorm: no bias, only scale parameter
   - Formula: x * weight / sqrt(mean(x²) + eps)

2. Rotary Position Embedding (RoPE): No learned positional embeddings
   - Position information encoded directly in attention through rotation
   - Frequencies: θᵢ = 10000^(-2i/d) for i in [0, d/2)
   - Applied to query/key vectors before attention computation

3. Grouped-Query Attention (GQA): More efficient than standard multi-head attention
   - Fewer key/value heads than query heads (8 KV heads vs 32 Q heads for 8B model)
   - Key/value heads are repeated to match query head count during computation

4. SwiGLU Activation: Instead of GELU, uses Swish-Gated Linear Units
   - Formula: SwiGLU(x) = Swish(x @ W_gate) ⊙ (x @ W_up) @ W_down
   - Where Swish(x) = x * sigmoid(x)

5. Weight Tying: Input embeddings and output projection often share weights

Usage:
    python engine/scripts/dev/llama3_jax/solution.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple
from jax import Array
from dataclasses import dataclass
from engine.core.utils.comparison import compare_logits, get_hf_logits
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# Force consistent precision
jax.config.update("jax_enable_x64", False)

@dataclass(frozen=True)
class Llama3Config:
    """Configuration for Llama3 model."""
    vocab_size: int = 128256
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
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


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)"""
    # RMS norm: x * weight / sqrt(mean(x²) + eps)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normalized = x / jnp.sqrt(variance + eps)
    return normalized * weight


def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 500000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute rotary position embedding frequencies"""
    # Create frequency tensor: θᵢ = base^(-2i/d) for i in [0, d/2)
    freqs = 1.0 / (theta ** (jnp.arange(0, d_head, 2).astype(jnp.float32) / d_head))
    
    # Create position indices
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    
    # Compute angles: position * frequency for each position and frequency
    angles = jnp.outer(positions, freqs)  # [seq_len, d_head//2]
    
    # Convert to complex exponentials and then to cos/sin
    cos_angles = jnp.cos(angles)
    sin_angles = jnp.sin(angles)
    
    return cos_angles, sin_angles


def apply_rotary_pos_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary position embedding to input tensor"""
    # x shape: [batch, seq_len, n_heads, head_dim]
    # cos, sin shape: [seq_len, head_dim//2]
    
    batch_size, seq_len, n_heads, head_dim = x.shape
    
    # Reshape x to separate real/imaginary parts
    # Split into pairs: [x0, x1, x2, x3, ...] -> [(x0, x1), (x2, x3), ...]
    x_pairs = x.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
    
    # Extract real and imaginary parts
    x_real, x_imag = x_pairs[..., 0], x_pairs[..., 1]
    
    # Expand cos/sin to match tensor dimensions
    cos = cos[:seq_len].reshape(1, seq_len, 1, head_dim // 2)
    sin = sin[:seq_len].reshape(1, seq_len, 1, head_dim // 2)
    
    # Apply rotation: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    x_real_rot = x_real * cos - x_imag * sin
    x_imag_rot = x_real * sin + x_imag * cos
    
    # Recombine pairs and reshape back
    x_rot_pairs = jnp.stack([x_real_rot, x_imag_rot], axis=-1)
    x_rot = x_rot_pairs.reshape(batch_size, seq_len, n_heads, head_dim)
    
    return x_rot


def grouped_query_attention(x: jnp.ndarray, 
                           w_q: jnp.ndarray, w_k: jnp.ndarray, w_v: jnp.ndarray, w_o: jnp.ndarray,
                           cos: jnp.ndarray, sin: jnp.ndarray,
                           config: Llama3Config) -> jnp.ndarray:
    """Grouped-Query Attention mechanism used in Llama3"""
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // config.n_heads
    n_rep = config.n_heads // config.n_kv_heads  # How many times to repeat each KV head
    
    # Project to Q, K, V - fix einsum and ensure correct dimensions
    q_proj = jnp.einsum('bld,dd->bld', x, w_q)  # [batch, seq, d_model]
    k_proj = jnp.einsum('bld,kd->blk', x, w_k)  # [batch, seq, kv_dim] 
    v_proj = jnp.einsum('bld,kd->blk', x, w_v)  # [batch, seq, kv_dim]
    
    # Reshape to heads
    q = q_proj.reshape(batch_size, seq_len, config.n_heads, head_dim)
    k = k_proj.reshape(batch_size, seq_len, config.n_kv_heads, head_dim)
    v = v_proj.reshape(batch_size, seq_len, config.n_kv_heads, head_dim)
    
    # Apply RoPE to Q and K separately
    q_rot = apply_rotary_pos_emb(q, cos, sin)
    k_rot = apply_rotary_pos_emb(k, cos, sin)
    
    # Repeat K and V heads to match Q heads (grouped-query attention)
    k_rot = jnp.repeat(k_rot, n_rep, axis=2)  # [batch, seq_len, n_heads, head_dim]
    v = jnp.repeat(v, n_rep, axis=2)          # [batch, seq_len, n_heads, head_dim]
    
    # Compute attention scores
    scores = jnp.einsum('blhd,bmhd->bhlm', q_rot, k_rot) / jnp.sqrt(head_dim)
    
    # Apply causal mask
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    scores = jnp.where(mask == 1, jnp.finfo(scores.dtype).min, scores)
    
    # Softmax attention weights
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply attention to values
    attn_output = jnp.einsum('bhlm,bmhd->blhd', attn_weights, v)
    
    # Reshape and project output
    attn_output = attn_output.reshape(batch_size, seq_len, d_model)
    output = jnp.einsum('bld,dd->bld', attn_output, w_o)
    
    return output


def swiglu_ffn(x: jnp.ndarray, 
               w_gate: jnp.ndarray, w_up: jnp.ndarray, w_down: jnp.ndarray) -> jnp.ndarray:
    """SwiGLU feedforward network (Shazeer, 2020)"""
    # SwiGLU(x) = Swish(x @ w_gate) * (x @ w_up) @ w_down
    # where Swish(x) = x * sigmoid(x)
    gate = jnp.einsum('bld,fd->blf', x, w_gate)
    up = jnp.einsum('bld,fd->blf', x, w_up)
    
    # Apply Swish activation to gate
    swish_gate = gate * jax.nn.sigmoid(gate)
    
    # Element-wise multiply with up projection, then down project
    hidden = swish_gate * up
    output = jnp.einsum('blf,df->bld', hidden, w_down)
    
    return output


def llama3_block(x: jnp.ndarray, layer_idx: int, weights: Dict[str, Array], 
                cos: jnp.ndarray, sin: jnp.ndarray, config: Llama3Config) -> jnp.ndarray:
    """Single Llama3 transformer block"""
    
    # Pre-attention RMS norm
    x_norm = rms_norm(x, weights[f'model.layers.{layer_idx}.input_layernorm.weight'], config.rms_norm_eps)
    
    # Self-attention
    attn_output = grouped_query_attention(
        x_norm,
        weights[f'model.layers.{layer_idx}.self_attn.q_proj.weight'],
        weights[f'model.layers.{layer_idx}.self_attn.k_proj.weight'], 
        weights[f'model.layers.{layer_idx}.self_attn.v_proj.weight'],
        weights[f'model.layers.{layer_idx}.self_attn.o_proj.weight'],
        cos, sin, config
    )
    
    # Residual connection
    x = x + attn_output
    
    # Pre-FFN RMS norm
    x_norm = rms_norm(x, weights[f'model.layers.{layer_idx}.post_attention_layernorm.weight'], config.rms_norm_eps)
    
    # SwiGLU FFN
    ffn_output = swiglu_ffn(
        x_norm,
        weights[f'model.layers.{layer_idx}.mlp.gate_proj.weight'],
        weights[f'model.layers.{layer_idx}.mlp.up_proj.weight'],
        weights[f'model.layers.{layer_idx}.mlp.down_proj.weight']
    )
    
    # Residual connection
    x = x + ffn_output
    
    return x


def llama3_forward(input_ids: jnp.ndarray, weights: Dict[str, Array], config: Llama3Config) -> jnp.ndarray:
    """Forward pass through Llama3 model"""
    batch_size, seq_len = input_ids.shape
    
    # Token embeddings (no positional embeddings in Llama3)
    x = weights['model.embed_tokens.weight'][input_ids]  # [batch, seq_len, d_model]
    
    # Precompute RoPE frequencies
    head_dim = config.d_model // config.n_heads
    cos, sin = precompute_rope_freqs(head_dim, config.max_seq_len, config.rope_theta)
    
    # Pass through transformer blocks
    for layer_idx in range(config.n_layers):
        x = llama3_block(x, layer_idx, weights, cos, sin, config)
    
    # Final RMS norm
    x = rms_norm(x, weights['model.norm.weight'], config.rms_norm_eps)
    
    # Language modeling head (often tied to input embeddings)
    logits = jnp.einsum('bld,vd->blv', x, weights['lm_head.weight'])
    
    return logits


def load_and_print_real_weights(model_name: str = "huggyllama/llama-7b"):
    """Load real Llama weights from HuggingFace and convert to JAX format"""
    print(f"🦙 Loading Llama model: {model_name}")
    
    # Load the model - remove device_map to avoid accelerate requirement for testing
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        # device_map="cpu"  # Keep on CPU to avoid GPU memory issues
    )
    
    # Convert PyTorch weights to JAX format
    jax_weights = {}
    
    print("🔄 Converting PyTorch weights to JAX format...")
    for name, param in model.named_parameters():
        # Convert to numpy then JAX array
        jax_weights[name] = jnp.array(param.detach().cpu().numpy())
        print(f"  {name}: {jax_weights[name].shape}")
    
    print(f"✅ Loaded {len(jax_weights)} weight tensors")
    return jax_weights


def test_architecture():
    """Test the architecture with dummy weights"""
    print("🧪 Testing Llama3 JAX architecture with dummy weights...")
    
    # Create config and dummy weights
    config = Llama3Config(
        vocab_size=1000,  # Smaller for testing
        d_model=512,      # Smaller for testing
        n_layers=2,       # Smaller for testing
        n_heads=8,
        n_kv_heads=2,
        training=True
    )
    
    # Create dummy weights
    dummy_weights = {}
    dummy_weights['model.embed_tokens.weight'] = jnp.zeros((config.vocab_size, config.d_model))
    dummy_weights['model.norm.weight'] = jnp.ones((config.d_model,))
    dummy_weights['lm_head.weight'] = jnp.zeros((config.vocab_size, config.d_model))
    
    # Layer weights
    for i in range(config.n_layers):
        dummy_weights[f'model.layers.{i}.input_layernorm.weight'] = jnp.ones((config.d_model,))
        dummy_weights[f'model.layers.{i}.post_attention_layernorm.weight'] = jnp.ones((config.d_model,))
        
        # Attention weights - fix dimensions for grouped-query attention
        head_dim = config.d_model // config.n_heads
        kv_dim = head_dim * config.n_kv_heads  # Total dimension for K/V projections
        
        dummy_weights[f'model.layers.{i}.self_attn.q_proj.weight'] = jnp.zeros((config.d_model, config.d_model))
        dummy_weights[f'model.layers.{i}.self_attn.k_proj.weight'] = jnp.zeros((kv_dim, config.d_model))
        dummy_weights[f'model.layers.{i}.self_attn.v_proj.weight'] = jnp.zeros((kv_dim, config.d_model))
        dummy_weights[f'model.layers.{i}.self_attn.o_proj.weight'] = jnp.zeros((config.d_model, config.d_model))
        
        # FFN weights
        ffn_dim = config.d_model * 4  # Standard scaling
        dummy_weights[f'model.layers.{i}.mlp.gate_proj.weight'] = jnp.zeros((ffn_dim, config.d_model))
        dummy_weights[f'model.layers.{i}.mlp.up_proj.weight'] = jnp.zeros((ffn_dim, config.d_model))
        dummy_weights[f'model.layers.{i}.mlp.down_proj.weight'] = jnp.zeros((config.d_model, ffn_dim))
    
    # Test input
    test_input = jnp.array([[1, 2, 3, 4, 5]])
    
    print("🔥 Running forward pass with dummy weights...")
    try:
        logits = llama3_forward(test_input, dummy_weights, config)
        print(f"✅ Success! Output shape: {logits.shape}")
        print(f"Expected shape: {test_input.shape + (config.vocab_size,)}")
        
        if logits.shape == test_input.shape + (config.vocab_size,):
            print("✅ Output shape is correct!")
            return True
        else:
            print("❌ Output shape mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_against_hf():
    """Validate JAX implementation against HuggingFace"""
    print("🧪 Validating JAX Llama against HuggingFace...")
    
    # Test input
    test_input = jnp.array([[1, 2, 3, 4, 5]])  # Simple token sequence
    
    # Get HuggingFace reference FIRST (memory efficient)
    print("📚 Getting HuggingFace reference...")
    hf_logits = get_hf_logits(np.array(test_input), model_name="huggyllama/llama-7b")
    print(f"HF model loaded and unloaded. Cached logits shape: {hf_logits.shape}")
    
    # Load JAX weights and create config  
    print("📦 Loading JAX weights...")
    weights = load_and_print_real_weights()
    config = Llama3Config(training=True)
    
    # Get JAX logits
    print("🔥 Running JAX forward pass...")
    jax_logits = llama3_forward(test_input, weights, config)
    jax_logits_np = np.array(jax_logits)
    
    # Compare
    print("⚖️ Comparing logits...")
    comparison = compare_logits(jax_logits_np, hf_logits, rtol=5e-3, atol=1e-1, verbose=True)
    
    if comparison['all_close']:
        print("✅ SUCCESS: JAX implementation matches HuggingFace!")
    else:
        print("❌ MISMATCH: Need to debug implementation")
        print(f"Max difference: {comparison['max_abs_diff']:.6f}")
        print(f"Mean difference: {comparison['mean_abs_diff']:.6f}")


if __name__ == "__main__":
    # First test architecture with dummy weights
    if test_architecture():
        print("\n" + "="*50)
        print("Architecture test passed! Now testing against HuggingFace...")
        validate_against_hf()
    else:
        print("❌ Architecture test failed. Fix implementation first.")