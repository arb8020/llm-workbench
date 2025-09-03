#!/usr/bin/env python3
"""
Minimal GPT-2 JAX implementation skeleton.

Students should implement gpt2_forward() to match HuggingFace GPT-2 logits.

Usage:
    python engine/scripts/dev/hello_gpt2_jax_skeleton.py
"""

import jax
import jax.numpy as jnp
import einops
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from jax import Array
from engine.core.utils.weights import load_gpt2_weights, download_gpt2_weights, load_and_print_gpt2_weights_jax, validate_gpt2_weights

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
class GPT2Config:
    """Configuration for GPT-2 model."""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_positions: int = 1024
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True
    training: bool = False
    freqs_cis: Optional[Array] = None  
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"


def _validate_ffn_shapes(x_BLD: jnp.ndarray, 
                        weight_in_DF: jnp.ndarray, 
                        bias_in_F: jnp.ndarray,
                        weight_out_FD: jnp.ndarray, 
                        bias_out_D: jnp.ndarray):
    
    D = x_BLD.shape[-1]  
    assert weight_in_DF.shape[0] == D, "weight_in_DF's first dimension must match x_BLD's last dimension"
    F = weight_in_DF.shape[1]  
    assert bias_in_F.shape[0] == F, "bias_in_F dimension must match weight_in_DF's second dimension"
    
    assert weight_out_FD.shape[0] == F, "weight_out_FD's first dimension must match weight_in_DF's second dimension"
    assert weight_out_FD.shape[1] == D, "weight_out_FD's second dimension must match x_BLD's last dimension"
    assert bias_out_D.shape[0] == D, "bias_out_D dimension must match x_BLD's last dimension"

def _validate_linear_shapes(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> None:
    assert x.shape[-1] == weight.shape[0], f"x shape {x.shape} incompatible with weight shape {weight.shape}"
    assert weight.shape[1] == bias.shape[0], f"weight shape {weight.shape} incompatible with bias shape {bias.shape}"

def _validate_layer_norm_shapes(x_BLD: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray):
    assert gamma.shape[-1] == x_BLD.shape[-1], "gamma's last dimension must match x_BLD's last dimension"
    assert beta.shape[-1] == x_BLD.shape[-1], "beta's last dimension must match x_BLD's last dimension"

def _validate_attention_shapes(x_BMD: jnp.ndarray, w_qkv_3DD: jnp.ndarray, b_qkv_3D: jnp.ndarray, 
                             w_out_DD: jnp.ndarray, b_out_D: jnp.ndarray, config: GPT2Config):
    """Validate all attention layer input shapes match expected dimensions."""
    B, M, D = x_BMD.shape
    
    # Validate input
    assert len(x_BMD.shape) == 3, f"x_BMD must be 3D, got shape {x_BMD.shape}"
    assert D == config.d_model, f"x_BMD last dim {D} must match config.d_model {config.d_model}"
    
    # Validate Q,K,V weights
    assert w_qkv_3DD.shape == (3, D, D), f"w_qkv_3DD shape {w_qkv_3DD.shape} must be (3, {D}, {D})"
    assert b_qkv_3D.shape == (3, D), f"b_qkv_3D shape {b_qkv_3D.shape} must be (3, {D})"
    
    # Validate output projection
    assert w_out_DD.shape == (D, D), f"w_out_DD shape {w_out_DD.shape} must be ({D}, {D})"
    assert b_out_D.shape == (D,), f"b_out_D shape {b_out_D.shape} must be ({D},)"
    
    # Validate head configuration
    assert D % config.n_heads == 0, f"d_model {D} must be divisible by n_heads {config.n_heads}"
    head_dim = D // config.n_heads
    assert head_dim > 0, f"Head dimension {head_dim} must be positive"
    




def gelu(x):
    """Hendrycks & Gimpel (2016) https://arxiv.org/abs/1606.08415"""
    # Using the approximation formula for GELU
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + jnp.tanh(
        jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))
    ))
    return x * cdf

def gelu_exact(x):
    """Hendrycks & Gimpel (2016) https://arxiv.org/abs/1606.08415"""

    return 0.5 * x * (1 + jnp.erf(x / jnp.sqrt(2.0)))

def gelu_hf(x):
    """Hendrycks & Gimpel (2016) https://arxiv.org/abs/1606.08415"""
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 *
  jnp.power(x, 3.0))))



def project_and_embed(input_ids: jnp.ndarray, weights: Dict[str, Array], config: GPT2Config) -> jnp.ndarray:
    """Radford et al. (2019) https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"""

    projected_BLD = weights['wte.weight'][input_ids]
    _, seq_len = input_ids.shape
    position_embeddings = weights['wpe.weight'][:seq_len]
    projected_embedded_BLD = projected_BLD + position_embeddings
    
    return projected_embedded_BLD

def layer_norm(x_BLD: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, epsilon: float = 1e-5) -> jnp.ndarray:
    """Ba et al. (2016) https://arxiv.org/abs/1607.06450"""

    _validate_layer_norm_shapes(x_BLD, gamma, beta)

    mean_BL1 = jnp.mean(x_BLD, axis=-1, keepdims=True)
    variance_BL1 = jnp.var(x_BLD, axis=-1, keepdims=True, ddof=0)

    demeaned_BLD = (x_BLD - mean_BL1) # BLD auto broadcasts over BL1
    demeaned_centered_BLD = demeaned_BLD / jnp.sqrt(variance_BL1 + epsilon)

    gamma_scaled_BLD = demeaned_centered_BLD * gamma
    beta_shifted_BLD = gamma_scaled_BLD + beta

    final_BLD = beta_shifted_BLD 

    return final_BLD


def ffn(x_BLD: jnp.ndarray, 
        weight_in_DF: jnp.ndarray, 
        bias_in_F: jnp.ndarray,
        weight_out_FD: jnp.ndarray, 
        bias_out_D: jnp.ndarray,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Vaswani et al. (2017) https://arxiv.org/abs/1706.03762"""

    _validate_ffn_shapes(x_BLD, weight_in_DF, bias_in_F, weight_out_FD, bias_out_D) 
    hidden_BLF = linear(x_BLD, weight_in_DF, bias_in_F)
    activated_BLF = activation_fn(hidden_BLF)
    output_BLD = linear(activated_BLF, weight_out_FD, bias_out_D)

    return output_BLD


def linear(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Goodfellow et al. (2016) http://www.deeplearningbook.org"""
    _validate_linear_shapes(x, weight, bias)
    return x @ weight + bias

def multihead_attn(x_BMD: jax.Array, w_qkv_3DD: jax.Array, b_qkv_3D: jax.Array, w_out_DD: jax.Array, b_out_D: jax.Array, config: GPT2Config, training: bool = False) -> jnp.ndarray:
    """Vaswani et al. (2017) https://arxiv.org/abs/1706.03762"""
    
    # Validate shapes before proceeding
    _validate_attention_shapes(x_BMD, w_qkv_3DD, b_qkv_3D, w_out_DD, b_out_D, config)
    
    H = config.n_heads
    K = config.d_model // config.n_heads
    
    x_BLD = x_BMD
    if not training:
        x_BLD = x_BMD[:, -1:, :] # take just the last token

    # split W_qkv
    w_qkv_3DHK = einops.rearrange(w_qkv_3DD, 'THREE D (H K) -> THREE D H K', H=H, K=K)
    w_q_DHK, w_k_DHK, w_v_DHK = w_qkv_3DHK[0], w_qkv_3DHK[1], w_qkv_3DHK[2]

    # split biases
    b_q_DHK = einops.rearrange(b_qkv_3D[0], '(H K) -> H K', H=H, K=K)
    b_k_DHK = einops.rearrange(b_qkv_3D[1], '(H K) -> H K', H=H, K=K)
    b_v_DHK = einops.rearrange(b_qkv_3D[2], '(H K) -> H K', H=H, K=K)

    # project into query/key/value
    query_BLHK = jnp.einsum('BLD,DHK->BLHK', x_BLD, w_q_DHK) + b_q_DHK
    key_BMHK = jnp.einsum('BMD,DHK->BMHK', x_BMD, w_k_DHK) + b_k_DHK
    value_BMHK = jnp.einsum('BMD,DHK->BMHK', x_BMD, w_v_DHK) + b_v_DHK
   
    # compute cos similarity over B and H. result matrix should be LM (would be MM for training)
    similarity_score_BHLM = jnp.einsum('BLHK,BMHK->BHLM', query_BLHK, key_BMHK)
    
    b, l, h, k = query_BLHK.shape
   
    scaled_score_BHLM = similarity_score_BHLM / (k**0.5) # scale by attention k/v dim

    # causal mask
    l, m = scaled_score_BHLM.shape[-2:]

    block_upper_LM = jnp.triu(jnp.ones((l, m)), k=1) # triu takes in a matrix as input
    causal_mask_LM = jnp.where(block_upper_LM == 1, jnp.finfo(scaled_score_BHLM.dtype).min, 0.0) # -inf for blocked values, 0 otherwise

    causal_mask_BHLM = einops.rearrange(causal_mask_LM, 'L M -> 1 1 L M')
    
    masked_score_BHLM = scaled_score_BHLM + causal_mask_BHLM 

    softmaxed_score_BHLM = jax.nn.softmax(masked_score_BHLM, axis=-1)

    weights_BLHK = jnp.einsum('BHLM,BMHK->BLHK', softmaxed_score_BHLM, value_BMHK) # dot over BH to BHLK, reshape to BL
    
    weights_BLD = einops.rearrange(weights_BLHK, 'B L H K -> B L (H K)')
    
    attn_out_BLD = jnp.einsum('BLD,DD->BLD', weights_BLD, w_out_DD) + b_out_D

    return attn_out_BLD

def multihead_attn_new(q_BLD: jax.Array, k_BLD: jax.Array, v_BLD: jax.Array, 
                      w_out_DD: jax.Array, b_out_D: jax.Array, config: GPT2Config) -> jnp.ndarray:
    """New attention function that takes Q,K,V directly (like solution.py approach)"""
    
    batch_size, seq_len, d_model = q_BLD.shape
    head_dim = d_model // config.n_heads
    
    # Reshape for multi-head attention: (B, L, H, K)
    q = q_BLD.reshape(batch_size, seq_len, config.n_heads, head_dim)
    k = k_BLD.reshape(batch_size, seq_len, config.n_heads, head_dim)  
    v = v_BLD.reshape(batch_size, seq_len, config.n_heads, head_dim)
    
    # Transpose to (B, H, L, K) for attention computation
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    
    # Scaled dot-product attention (like solution.py)
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
    scale_factor = jnp.float32(head_dim) ** 0.5
    scores = scores / scale_factor
    
    # Causal mask (like solution.py)
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask_value = jnp.finfo(scores.dtype).min
    scores = jnp.where(causal_mask[None, None, :, :], scores, mask_value)
    
    # Softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_weights = attn_weights.astype(v.dtype)
    
    # Apply attention to values
    attn_output = jnp.matmul(attn_weights, v)
    
    # Transpose back and reshape: (B, H, L, K) -> (B, L, H, K) -> (B, L, D)
    attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
    attn_output = attn_output.reshape(batch_size, seq_len, d_model)
    
    # Output projection
    output = jnp.matmul(attn_output, w_out_DD) + b_out_D
    
    return output

def gpt2_extract_block_weights(layer_idx: int, weights: Dict[str, Array]) -> Dict[str, Array]:
    """helper function to extract weights for a GPT2 block at given layer index"""
    return {
        'ln_1': {
            'weight': weights[f"h.{layer_idx}.ln_1.weight"],
            'bias': weights[f"h.{layer_idx}.ln_1.bias"]
        },
        'attn': {
            'c_attn': {
                'weight': weights[f"h.{layer_idx}.attn.c_attn.weight"],
                'bias': weights[f"h.{layer_idx}.attn.c_attn.bias"]
            },
            'c_proj': {
                'weight': weights[f"h.{layer_idx}.attn.c_proj.weight"],
                'bias': weights[f"h.{layer_idx}.attn.c_proj.bias"]
            }
        },
        'ln_2': {
            'weight': weights[f"h.{layer_idx}.ln_2.weight"],
            'bias': weights[f"h.{layer_idx}.ln_2.bias"]
        },
        'mlp': {
            'c_fc': {
                'weight': weights[f"h.{layer_idx}.mlp.c_fc.weight"],
                'bias': weights[f"h.{layer_idx}.mlp.c_fc.bias"]
            },
            'c_proj': {
                'weight': weights[f"h.{layer_idx}.mlp.c_proj.weight"],
                'bias': weights[f"h.{layer_idx}.mlp.c_proj.bias"]
            }
        }
    }

def gpt2_block(x_BLD: jnp.ndarray, layer_idx: int, weights: Dict[str, Array], config: GPT2Config) -> jnp.ndarray:
    """Radford et al. (2019) https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"""
    
    block_weights = gpt2_extract_block_weights(layer_idx, weights)
    
    normed_x_BLD = layer_norm(x_BLD, block_weights['ln_1']['weight'], block_weights['ln_1']['bias'], config.layer_norm_epsilon)
    
    # Apply linear transformation first, then split (like solution.py)
    c_attn_weight = block_weights['attn']['c_attn']['weight']  # Shape: [D, 3*D]
    c_attn_bias = block_weights['attn']['c_attn']['bias']      # Shape: [3*D]
    
    # Compute QKV in one shot: x @ W + b, then split
    qkv_BL3D = jnp.matmul(normed_x_BLD, c_attn_weight) + c_attn_bias
    q_BLD, k_BLD, v_BLD = jnp.split(qkv_BL3D, 3, axis=-1)
    
    # Get output projection weights
    w_out_DD = block_weights['attn']['c_proj']['weight']
    b_out_D = block_weights['attn']['c_proj']['bias']
    
    # Use new QKV computation in attention (need to update multihead_attn to accept Q,K,V directly)
    attn_output_BLD = multihead_attn_new(q_BLD, k_BLD, v_BLD, w_out_DD, b_out_D, config)
    
    x_BLD = x_BLD + attn_output_BLD
    
    normed_x_BLD = layer_norm(x_BLD, block_weights['ln_2']['weight'], block_weights['ln_2']['bias'], config.layer_norm_epsilon)
    
    
    ffn_output_BLD = ffn(normed_x_BLD, 
                     block_weights['mlp']['c_fc']['weight'],
                     block_weights['mlp']['c_fc']['bias'],
                     block_weights['mlp']['c_proj']['weight'],
                     block_weights['mlp']['c_proj']['bias'], gelu_hf)
    
    return x_BLD + ffn_output_BLD

def gpt2_forward(input_ids: jnp.ndarray, weights: Dict[str, Array], config: GPT2Config) -> jnp.ndarray:
    """Radford et al. (2019) https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"""
    batch_size, seq_len = input_ids.shape
    vocab_size = config.vocab_size

    # Validate required weight keys exist
    required_keys = [
        'wte.weight', 'wpe.weight', 'ln_f.weight', 'ln_f.bias'
    ]
    # Add layer-specific keys
    for i in range(config.n_layers):
        layer_keys = [
            f'h.{i}.ln_1.weight', f'h.{i}.ln_1.bias',
            f'h.{i}.attn.c_attn.weight', f'h.{i}.attn.c_attn.bias',
            f'h.{i}.attn.c_proj.weight', f'h.{i}.attn.c_proj.bias',
            f'h.{i}.ln_2.weight', f'h.{i}.ln_2.bias',
            f'h.{i}.mlp.c_fc.weight', f'h.{i}.mlp.c_fc.bias',
            f'h.{i}.mlp.c_proj.weight', f'h.{i}.mlp.c_proj.bias'
        ]
        required_keys.extend(layer_keys)
    
    validate_gpt2_weights(weights, required_keys)

    projected_embedded_BLD = project_and_embed(input_ids, weights, config)
    x_BLD = projected_embedded_BLD

    for layer_idx in range(config.n_layers):
        x_BLD = gpt2_block(x_BLD, layer_idx, weights, config)

    x_BLD = layer_norm(x_BLD, 
                      weights['ln_f.weight'], 
                      weights['ln_f.bias'], 
                      config.layer_norm_epsilon)

    logits_BLV = jnp.einsum('BLD,VD->BLV', x_BLD, weights['wte.weight'])
    
    return logits_BLV


if __name__ == "__main__":
    
    # print real weights
    real_weights = load_and_print_gpt2_weights_jax() 
    
    config = GPT2Config(training=True)
    test_input = jnp.array([[15496, 995]])  # "Hello world" tokens
    
    logits = gpt2_forward(test_input, real_weights, config)
    
    
