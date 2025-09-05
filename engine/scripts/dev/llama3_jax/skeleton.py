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

"""
KV Cache implementation for Llama3-style transformer.

Dimension key:
- B: batch size
- L: sequence length (current input)
- M: memory/context length (max sequence length in cache)
- Ly: layer index
- G: number of key-value heads (grouped query attention)
- H: number of query heads
- K: head dimension (d_kv)
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp


class KVCache(NamedTuple):
    """Key-Value cache for multi-layer transformer."""
    keys_LyBMGK: jax.Array    # (layers, batch, max_context_len, kv_heads, head_dim)
    values_LyBMGK: jax.Array  # (layers, batch, max_context_len, kv_heads, head_dim)


class Llama3State(NamedTuple):
    """State container for Llama3 model."""
    cache: KVCache
    current_position: int
    max_sequence_length: int


def create_kv_cache(
    num_layers: int,
    batch_size: int,
    max_sequence_length: int,
    kv_heads: int,
    head_dim: int
) -> KVCache:
    """Initialize an empty KV cache with zeros."""
    shape = (num_layers, batch_size, max_sequence_length, kv_heads, head_dim)
    zeros = jnp.zeros(shape, dtype=jnp.bfloat16)
    return KVCache(keys_LyBMGK=zeros, values_LyBMGK=zeros)


def write_to_cache(
    cache: KVCache,
    layer_idx: int,
    position: int,
    keys_BLGK: jax.Array,
    values_BLGK: jax.Array
) -> KVCache:
    """"""
    # Convert to bfloat16 for storage efficiency
    keys_bf16_BLGK = jnp.bfloat16(keys_BLGK)
    values_bf16_BLGK = jnp.bfloat16(values_BLGK)
    
    seq_len = keys_BLGK.shape[1]
    
    # Expand to match cache dimensions
    keys_expanded_1BLGK = keys_bf16_BLGK[jnp.newaxis, :, :, :, :]  
    updated_keys_LyBMGK = jax.lax.dynamic_update_slice(
        cache.keys_LyBMGK,
        keys_expanded_1BLGK,
        (layer_idx, 0, position, 0, 0)
    )
    
    # expand to match cache dimensions 
    values_expanded_1BLGK = values_bf16_BLGK[jnp.newaxis, :, :, :, :]  
    updated_values_LyBMGK = jax.lax.dynamic_update_slice(
        cache.values_LyBMGK,
        values_expanded_1BLGK,
        (layer_idx, 0, position, 0, 0)
    )
    
    return KVCache(keys_LyBMGK=updated_keys_LyBMGK, values_LyBMGK=updated_values_LyBMGK)


def read_kvcache(
    cache: KVCache,
    layer_idx: int,
    repetition_factor: int
) -> Tuple[jax.Array, jax.Array]:
    keys_BMGK = cache.keys_LyBMGK[layer_idx]      
    values_BMGK = cache.values_LyBMGK[layer_idx]  
    
    keys_BMHK = replicate_kv_heads(keys_BMGK, repetition_factor)
    values_BMHK = replicate_kv_heads(values_BMGK, repetition_factor)
    
    return keys_BMHK, values_BMHK


def replicate_kv_heads(tensor_BMGK: jax.Array, repetition_factor: int) -> jax.Array:
    batch, seq_len, kv_heads, head_dim = tensor_BMGK.shape
    
    # add repetition dimension and broadcast
    tensor_expanded_BMG1K = tensor_BMGK[:, :, :, jnp.newaxis, :]  
    tensor_replicated_BMGRK = jnp.broadcast_to(
        tensor_expanded_BMG1K,
        (batch, seq_len, kv_heads, repetition_factor, head_dim)
    )
    
    # merge kv_heads and repetition dimensions
    query_heads = kv_heads * repetition_factor
    tensor_BMHK = tensor_replicated_BMGRK.reshape(batch, seq_len, query_heads, head_dim)
    
    return tensor_BMHK


def update_llama3_state(
    state: Llama3State,
    layer_idx: int,
    keys_BLGK: jax.Array,
    values_BLGK: jax.Array
) -> Llama3State:
    seq_len = keys_BLGK.shape[1]
    
    # write to cache
    updated_cache = write_to_cache(
        state.cache,
        layer_idx,
        state.current_position,
        keys_BLGK,
        values_BLGK
    )
    
    # advance position (clamp to max sequence length)
    new_position = jnp.minimum(
        state.current_position + seq_len,
        state.max_sequence_length
    )
    
    return Llama3State(
        cache=updated_cache,
        current_position=int(new_position),
        max_sequence_length=state.max_sequence_length
    )


def update_and_read_kvcache(
    state: Llama3State,
    layer_idx: int,
    keys_BLGK: jax.Array,
    values_BLGK: jax.Array,
    repetition_factor: int
) -> Tuple[jax.Array, jax.Array, Llama3State]:
    """"""
    # update state with new keys/values
    updated_state = update_llama3_state(state, layer_idx, keys_BLGK, values_BLGK)
    
    # read back all keys/values for this layer (with replication)
    keys_BMHK, values_BMHK = read_kvcache(
        updated_state.cache,
        layer_idx,
        repetition_factor
    )
    
    return keys_BMHK, values_BMHK, updated_state





# @dataclass(frozen=True)
# class Llama3State:
    

# rms norm 
# rms norm
# swiglu
# kv cache
# gqa
# rope

# how many blocks in llama 3.1-8b, concise, ignore notes
# """ Llama 3.1-8b has 32 transformer blocks/layers."""


"""
🔍 Original HuggingFace weight shapes:
  model.embed_tokens.weight: (128256, 4096)
  model.layers.0.input_layernorm.weight: (4096,)
  model.layers.0.mlp.down_proj.weight: (4096, 14336)
  model.layers.0.mlp.gate_proj.weight: (14336, 4096)
  model.layers.0.mlp.up_proj.weight: (14336, 4096)
  model.layers.0.post_attention_layernorm.weight: (4096,)
  model.layers.0.self_attn.k_proj.weight: (1024, 4096)
  model.layers.0.self_attn.o_proj.weight: (4096, 4096)
  model.layers.0.self_attn.q_proj.weight: (4096, 4096)
  model.layers.0.self_attn.v_proj.weight: (1024, 4096)
  model.layers.1.input_layernorm.weight: (4096,)
  model.layers.1.post_attention_layernorm.weight: (4096,)
  model.layers.2.input_layernorm.weight: (4096,)
  model.layers.2.post_attention_layernorm.weight: (4096,)
  model.layers.3.input_layernorm.weight: (4096,)
  model.layers.3.post_attention_layernorm.weight: (4096,)
  model.layers.4.input_layernorm.weight: (4096,)
  model.layers.4.post_attention_layernorm.weight: (4096,)
  model.layers.5.input_layernorm.weight: (4096,)
  model.layers.5.post_attention_layernorm.weight: (4096,)
  model.layers.6.input_layernorm.weight: (4096,)
  model.layers.6.post_attention_layernorm.weight: (4096,)
  model.layers.7.input_layernorm.weight: (4096,)
  model.layers.7.post_attention_layernorm.weight: (4096,)
  model.layers.8.input_layernorm.weight: (4096,)
  model.layers.8.post_attention_layernorm.weight: (4096,)
  model.layers.10.input_layernorm.weight: (4096,)
  model.layers.10.post_attention_layernorm.weight: (4096,)
  model.layers.11.input_layernorm.weight: (4096,)
  model.layers.11.post_attention_layernorm.weight: (4096,)
  model.layers.12.input_layernorm.weight: (4096,)
  model.layers.12.post_attention_layernorm.weight: (4096,)
  model.layers.13.input_layernorm.weight: (4096,)
  model.layers.13.post_attention_layernorm.weight: (4096,)
  model.layers.14.input_layernorm.weight: (4096,)
  model.layers.14.post_attention_layernorm.weight: (4096,)
  model.layers.15.input_layernorm.weight: (4096,)
  model.layers.15.post_attention_layernorm.weight: (4096,)
  model.layers.16.input_layernorm.weight: (4096,)
  model.layers.16.post_attention_layernorm.weight: (4096,)
  model.layers.17.input_layernorm.weight: (4096,)
  model.layers.17.post_attention_layernorm.weight: (4096,)
  model.layers.18.input_layernorm.weight: (4096,)
  model.layers.18.post_attention_layernorm.weight: (4096,)
  model.layers.19.input_layernorm.weight: (4096,)
  model.layers.19.post_attention_layernorm.weight: (4096,)
  model.layers.9.input_layernorm.weight: (4096,)
  model.layers.9.post_attention_layernorm.weight: (4096,)
  model.layers.20.input_layernorm.weight: (4096,)
  model.layers.20.post_attention_layernorm.weight: (4096,)
  model.layers.21.input_layernorm.weight: (4096,)
  model.layers.21.post_attention_layernorm.weight: (4096,)
  model.layers.22.input_layernorm.weight: (4096,)
  model.layers.22.post_attention_layernorm.weight: (4096,)
  model.layers.23.input_layernorm.weight: (4096,)
  model.layers.23.post_attention_layernorm.weight: (4096,)
  model.layers.24.input_layernorm.weight: (4096,)
  model.layers.24.post_attention_layernorm.weight: (4096,)
  model.layers.25.input_layernorm.weight: (4096,)
  model.layers.25.post_attention_layernorm.weight: (4096,)
  model.layers.26.input_layernorm.weight: (4096,)
  model.layers.26.post_attention_layernorm.weight: (4096,)
  model.layers.27.input_layernorm.weight: (4096,)
  model.layers.27.post_attention_layernorm.weight: (4096,)
  model.layers.28.input_layernorm.weight: (4096,)
  model.layers.28.post_attention_layernorm.weight: (4096,)
  model.layers.29.input_layernorm.weight: (4096,)
  model.layers.29.post_attention_layernorm.weight: (4096,)
  model.layers.30.input_layernorm.weight: (4096,)
  model.layers.30.post_attention_layernorm.weight: (4096,)
  lm_head.weight: (128256, 4096)
  model.layers.31.input_layernorm.weight: (4096,)
  model.layers.31.post_attention_layernorm.weight: (4096,)
  model.norm.weight: (4096,)
"""

def llama_extract_block_weights(layer_idx: int, weights: Dict[str, Array]) -> Dict[str, Array]:
    """helper function to extract weights for a LLaMA block at given layer index"""
    return {
        'attention_norm': {
            'weight': weights[f"layers.{layer_idx}.attention_norm.weight"],
        },
        'attention': {
            'wq': {
                'weight': weights[f"layers.{layer_idx}.attention.wq.weight"],
            },
            'wk': {
                'weight': weights[f"layers.{layer_idx}.attention.wk.weight"],
            },
            'wv': {
                'weight': weights[f"layers.{layer_idx}.attention.wv.weight"],
            },
            'wo': {
                'weight': weights[f"layers.{layer_idx}.attention.wo.weight"],
            }
        },
        'ffn_norm': {
            'weight': weights[f"layers.{layer_idx}.ffn_norm.weight"],
        },
        'feed_forward': {
            'w1': {
                'weight': weights[f"layers.{layer_idx}.feed_forward.w1.weight"],
            },
            'w2': {
                'weight': weights[f"layers.{layer_idx}.feed_forward.w2.weight"],
            },
            'w3': {
                'weight': weights[f"layers.{layer_idx}.feed_forward.w3.weight"],
            }
        }
    }


def swiglu(x_BLD: jnp.ndarray, gate_proj_HD: jnp.ndarray, up_proj_HD: jnp.ndarray, down_proj_DH: jnp.ndarray):
    """Swiglu: Simple gated linear unit variant used in PaLM
    Shazeer (2020) https://arxiv.org/abs/2002.05202"""

    gate_BLH = jax.nn.silu(jnp.einsum('bld,hd->blh', x_BLD, gate_proj_HD)) 
    projected_up_BLH = jnp.einsum('bld,hd->blh', x_BLD, up_proj_HD)
    gated_output_BLH = gate_BLH * projected_up_BLH 
    back_down_BLD = jnp.einsum('blh,dh->bld')

    return

def initialize_rotation_matrices(dim: int, seq_len: int, theta: float = 500000.0, use_ntk_scaling: bool = True):
    """
    precomputes complex rotation factors for RoPE to avoid redundant computation during attention
    captures range of patterns by using diff frequencies depending on the embedding dimension's index
    su et al, 2021: https://arxiv.org/abs/2104.09864
    """
    # early dim -> high freq -> rotates fast -> local patterns
    # later dim -> low freq -> rotates slow -> global patterns
    frequencies = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))

    # NTK aware scaling
    if use_ntk_scaling:
        frequencies = apply_ntk_scaling(frequencies)

    # multiply each pos by each frequency
    positions = jnp.arange(seq_len)
    frequencies = jnp.outer(positions, frequencies)

    # convert to complex rotation factors with euler's formula
    return jnp.exp(1j * frequencies) # 1j is the imaginary unit

def apply_ntk_scaling(frequencies: jax.Array, scaling_factor: int = 8, low_freq_factor: int = 1, high_freq_factor: int = 4, original_context: int = 4096):
  return jax.vmap(scale_frequencies)(frequencies)

def apply_rotary_emb(
    q_BLHK: jnp.ndarray,
    k_BLGK: jnp.ndarray,
    rot_LK2_complex: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    
    # honestly could drop this part
    Bq, Lq, H, Kq = q_BLHK.shape
    Bk, Lk, G, Kk = k_BLGK.shape
    assert Bq == Bk,     f"B mismatch: {Bq} vs {Bk}"
    assert Lq == Lk,     f"L mismatch: {Lq} vs {Lk}"
    assert Kq == Kk,     f"K mismatch: {Kq} vs {Kk}"
    assert Kq % 2 == 0,  f"K must be even, got {Kq}"
    assert rot_LK2_complex.shape == (Lq, Kq // 2), \
        f"rot shape must be [L, K/2] = [{Lq}, {Kq//2}], got {rot_LK2_complex.shape}"


    B, L, K = Bq, Lq, Kq

    # cast rotation to matching complex precision
    rot_LK2_complex = rot_LK2_complex[:L]


    rot_LK2_complex = rot_LK2_complex.astype(
        jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    )

    # split into real/imaginary pairs
    paired_q_BLH_K2_2 = q_BLHK.reshape(B, L, H, K // 2, 2).astype(dtype)
    paired_k_BLG_K2_2 = k_BLGK.reshape(B, L, G, K // 2, 2).astype(dtype)

    # convert a/b pairs to z = a + bi
    q_c_BLH_K2 = jax.lax.complex(q_BLH_K2_2[..., 0], q_BLH_K2_2[..., 1])
    k_c_BLG_K2 = jax.lax.complex(k_BLG_K2_2[..., 0], k_BLG_K2_2[..., 1])

    # apply rotations (broadcast over B and heads)
    q_rot_c_BLH_K2 = q_c_BLH_K2 * rot_LK2_C[None, :, None, :]  
    k_rot_c_BLG_K2 = k_c_BLG_K2 * rot_LK2_C[None, :, None, :]  

    # concatenate real and imaginary
    q_realimag_concat_BLH_K2_2 = jnp.stack((jnp.real(q_rot_c_BLH_K2), jnp.imag(q_rot_c_BLH_K2)), axis=-1)  
    k_realimag_concat_BLG_K2_2 = jnp.stack((jnp.real(k_rot_c_BLG_K2), jnp.imag(k_rot_c_BLG_K2)), axis=-1)  

    # final reshape back into input dims
    q_out_BLHK = q_ri_BLH_K2_2.reshape(B, L, H, K).astype(dtype)  
    k_out_BLGK = k_ri_BLG_K2_2.reshape(B, L, G, K).astype(dtype)  

    return xq_out, xk_out


def grouped_query_attn(x_BLD: jnp.ndarray, 
    q_proj_DD: jnp.ndarray,       
    k_proj_KD: jnp.ndarray,       
    v_proj_KD: jnp.ndarray,       
    o_proj_DD: jnp.ndarray,       
    config: Llama3Config,
    rotation_matrices: jnp.ndarray, 
    cur_pos: int = 0, 
    block_idx: int = 0,
    kv_cache: Optional[KVCache] = None) -> jnp.ndarray:
    """
    model.layers.0.self_attn.k_proj.weight: (1024, 4096)
    model.layers.0.self_attn.o_proj.weight: (4096, 4096)
    model.layers.0.self_attn.q_proj.weight: (4096, 4096)
    model.layers.0.self_attn.v_proj.weight: (1024, 4096)

    """

    B, L, D = x_BLD.shape
    H = config.num_heads
    G = config.num_kv_heads
    head_dim = D // H
    K = head_dim
    R = H // G

    # project
    q_BLD = jnp.einsum('bld,dd->bld', x_BLD, q_proj_DD)
    k_BLK = jnp.einsum('bld,kd->blk', x_BLD, k_proj_KD)
    v_BLK = jnp.einsum('bld,kd->blk', x_BLD, v_proj_KD)

    # split into heads
    head_dim = config.d_model // config.n_heads

    q_BLHK = q_BLD.reshape(B, L, H, K)
    k_BLGK = k_BLK.reshape(B, L, H, K)
    v_BLGK = v_BLK.reshape(B, L, H, K)

    
    # rope
    # bring in the kv cache
    # get scores with qk
    # weight values by scores

    
    out_BLD = jnp.einsum('', , o_proj_DD)
    return 






    


    pass



def rms_norm(x_BLD: jnp.ndarray, weight: jnp.ndarray, epsilon: float = 1e-5) -> jnp.ndarray:

    """RMSNorm: A Simple Yet Effective Normalization Method For Deep Neural Networks
    Zhang et al. (2019) https://arxiv.org/abs/1910.07467"""
    
    squared_inputs = jax.lax.pow(x_BLD, 2)
    mean_squared = squared_inputs.mean(-1, keepdims=True)
    stabilized_mean = mean_squared + epsilon
    rms = jax.lax.rsqrt(stabilized_mean)
    normalized = weight * (inputs * rms)
    return normalized


def project_input_ids(input_ids: jnp.ndarray, weights: Dict[str, Array], config: Llama3Config) -> jnp.ndarray:
    projected_BLD = weights['model.embed_tokens.weight'][input_ids]

    return projected_BLD

def llama3_block(x_BLD: jnp.ndarray, layer_idx: int, weights: Dict[str, Array], config: Llama3Config) -> jnp.ndarray:
    
    block_weights = llama_extract_block_weights(layer_idx, weights)

    # rms norm gqa
    # rms norm swiglu
    # maintain residual
    
    return x_BLD + ffn_output_BLD


def llama3_forward(input_ids: jnp.ndarray, weights: Dict[str, Array], config: Llama3Config) -> jnp.ndarray:
    """Forward pass through Llama3 model"""
    batch_size, seq_len = input_ids.shape
    vocab_size = config.vocab_size

    project_BLD = project_input_ids(input_ids, weights, config)

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
