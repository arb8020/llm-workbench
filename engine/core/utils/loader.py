"""
Backend-agnostic GPT-2 weight loader using Noam Shazeer's naming conventions.

Dimension key:
B: batch size
L: sequence length  
M: memory length (length of sequence being attended to)
D: model dimension (d_model)
V: vocabulary size
F: feed-forward hidden size (4*D for GPT-2)
H: number of attention heads
K: size of each attention key/value (D/H)
N: number of layers
"""

from transformers import GPT2Model, GPT2Config
from typing import Dict, NamedTuple, Any, Optional, Literal
import numpy as np


class GPT2Params(NamedTuple):
    """GPT-2 parameters using shape suffix notation (backend-agnostic)."""
    # Token embeddings
    wte_VD: np.ndarray  # token embeddings
    wpe_MD: np.ndarray  # position embeddings
    
    # Layer parameters (N layers)
    ln1_scale_ND: np.ndarray  # layer norm 1 scale
    ln1_bias_ND: np.ndarray   # layer norm 1 bias
    
    # Attention parameters
    w_qkv_N3DD: np.ndarray    # combined QKV projection
    w_out_NDD: np.ndarray     # output projection
    
    ln2_scale_ND: np.ndarray  # layer norm 2 scale
    ln2_bias_ND: np.ndarray   # layer norm 2 bias
    
    # FFN parameters
    w_ffn1_NDF: np.ndarray    # first FFN layer (D -> 4D)
    b_ffn1_NF: np.ndarray     # first FFN bias
    w_ffn2_NFD: np.ndarray    # second FFN layer (4D -> D)
    b_ffn2_ND: np.ndarray     # second FFN bias
    
    # Final layer norm
    ln_f_scale_D: np.ndarray  # final layer norm scale
    ln_f_bias_D: np.ndarray   # final layer norm bias
    
    # Config
    n_layers: int
    n_heads: int
    n_embd: int
    vocab_size: int
    max_seq_len: int


def load_gpt2_weights(model_name: str = "gpt2") -> GPT2Params:
    """
    Load GPT-2 weights from HuggingFace and convert to JAX arrays with shape suffixes.
    
    Args:
        model_name: Name of the GPT-2 model variant ("gpt2", "gpt2-medium", etc.)
    
    Returns:
        GPT2Params with all model weights in JAX format
    """
    # Load HuggingFace model
    model = GPT2Model.from_pretrained(model_name)
    config = model.config
    
    N = config.n_layer
    H = config.n_head
    D = config.n_embd
    V = config.vocab_size
    M = config.n_positions
    F = D * 4  # FFN hidden size is 4*D in GPT-2
    K = D // H
    
    # Extract embeddings
    wte_VD = model.wte.weight.detach().numpy()
    wpe_MD = model.wpe.weight.detach().numpy()
    
    # Initialize layer arrays
    ln1_scale_ND = []
    ln1_bias_ND = []
    w_qkv_N3DD = []
    w_out_NDD = []
    ln2_scale_ND = []
    ln2_bias_ND = []
    w_ffn1_NDF = []
    b_ffn1_NF = []
    w_ffn2_NFD = []
    b_ffn2_ND = []
    
    # Extract weights from each layer
    for layer_idx in range(N):
        layer = model.h[layer_idx]
        
        # Layer norm 1
        ln1_scale_ND.append(layer.ln_1.weight.detach().numpy())
        ln1_bias_ND.append(layer.ln_1.bias.detach().numpy())
        
        # Attention - combine QKV into single tensor
        # GPT-2 uses Conv1D which has transposed weights
        q_weight = layer.attn.c_attn.weight[:, :D].detach().numpy().T  # (D, D)
        k_weight = layer.attn.c_attn.weight[:, D:2*D].detach().numpy().T  # (D, D)
        v_weight = layer.attn.c_attn.weight[:, 2*D:].detach().numpy().T  # (D, D)
        
        qkv_weight = np.stack([q_weight, k_weight, v_weight], axis=0)  # (3, D, D)
        w_qkv_N3DD.append(qkv_weight)
        
        # Output projection
        w_out_NDD.append(layer.attn.c_proj.weight.detach().numpy().T)
        
        # Layer norm 2
        ln2_scale_ND.append(layer.ln_2.weight.detach().numpy())
        ln2_bias_ND.append(layer.ln_2.bias.detach().numpy())
        
        # FFN layers
        w_ffn1_NDF.append(layer.mlp.c_fc.weight.detach().numpy().T)  # (D, 4D)
        b_ffn1_NF.append(layer.mlp.c_fc.bias.detach().numpy())
        w_ffn2_NFD.append(layer.mlp.c_proj.weight.detach().numpy().T)  # (4D, D)
        b_ffn2_ND.append(layer.mlp.c_proj.bias.detach().numpy())
    
    # Final layer norm
    ln_f_scale_D = model.ln_f.weight.detach().numpy()
    ln_f_bias_D = model.ln_f.bias.detach().numpy()
    
    # Stack all layer-wise tensors
    ln1_scale_ND = np.stack(ln1_scale_ND)
    ln1_bias_ND = np.stack(ln1_bias_ND)
    w_qkv_N3DD = np.stack(w_qkv_N3DD)
    w_out_NDD = np.stack(w_out_NDD)
    ln2_scale_ND = np.stack(ln2_scale_ND)
    ln2_bias_ND = np.stack(ln2_bias_ND)
    w_ffn1_NDF = np.stack(w_ffn1_NDF)
    b_ffn1_NF = np.stack(b_ffn1_NF)
    w_ffn2_NFD = np.stack(w_ffn2_NFD)
    b_ffn2_ND = np.stack(b_ffn2_ND)
    
    return GPT2Params(
        wte_VD=wte_VD,
        wpe_MD=wpe_MD,
        ln1_scale_ND=ln1_scale_ND,
        ln1_bias_ND=ln1_bias_ND,
        w_qkv_N3DD=w_qkv_N3DD,
        w_out_NDD=w_out_NDD,
        ln2_scale_ND=ln2_scale_ND,
        ln2_bias_ND=ln2_bias_ND,
        w_ffn1_NDF=w_ffn1_NDF,
        b_ffn1_NF=b_ffn1_NF,
        w_ffn2_NFD=w_ffn2_NFD,
        b_ffn2_ND=b_ffn2_ND,
        ln_f_scale_D=ln_f_scale_D,
        ln_f_bias_D=ln_f_bias_D,
        n_layers=N,
        n_heads=H,
        n_embd=D,
        vocab_size=V,
        max_seq_len=M
    )


def get_test_params(D: int = 768, H: int = 12, N: int = 12, V: int = 50257, M: int = 1024, seed: int = 42) -> GPT2Params:
    """
    Create random test parameters for GPT-2 with proper shapes.
    Default values match GPT-2 base model.
    
    Args:
        D: Model dimension (768 for GPT-2 base)
        H: Number of heads (12 for GPT-2 base)
        N: Number of layers (12 for GPT-2 base)
        V: Vocabulary size (50257 for GPT-2)
        M: Max sequence length (1024 for GPT-2)
        seed: Random seed for reproducibility
    
    Returns:
        GPT2Params with random weights for testing
    """
    F = D * 4  # FFN hidden size
    rng = np.random.RandomState(seed)
    
    return GPT2Params(
        wte_VD=rng.randn(V, D) * 0.02,
        wpe_MD=rng.randn(M, D) * 0.02,
        ln1_scale_ND=np.ones((N, D)),
        ln1_bias_ND=np.zeros((N, D)),
        w_qkv_N3DD=rng.randn(N, 3, D, D) * 0.02,
        w_out_NDD=rng.randn(N, D, D) * 0.02,
        ln2_scale_ND=np.ones((N, D)),
        ln2_bias_ND=np.zeros((N, D)),
        w_ffn1_NDF=rng.randn(N, D, F) * 0.02,
        b_ffn1_NF=np.zeros((N, F)),
        w_ffn2_NFD=rng.randn(N, F, D) * 0.02,
        b_ffn2_ND=np.zeros((N, D)),
        ln_f_scale_D=np.ones(D),
        ln_f_bias_D=np.zeros(D),
        n_layers=N,
        n_heads=H,
        n_embd=D,
        vocab_size=V,
        max_seq_len=M
    )


def convert_params_to_backend(params: GPT2Params, backend: Literal["jax", "torch", "numpy"] = "numpy") -> GPT2Params:
    """
    Convert GPT2Params to a specific backend format.
    
    Args:
        params: GPT2Params with numpy arrays
        backend: Target backend ("jax", "torch", or "numpy")
    
    Returns:
        GPT2Params with arrays converted to the specified backend
    """
    if backend == "numpy":
        return params
    
    elif backend == "jax":
        import jax.numpy as jnp
        return GPT2Params(
            wte_VD=jnp.array(params.wte_VD),
            wpe_MD=jnp.array(params.wpe_MD),
            ln1_scale_ND=jnp.array(params.ln1_scale_ND),
            ln1_bias_ND=jnp.array(params.ln1_bias_ND),
            w_qkv_N3DD=jnp.array(params.w_qkv_N3DD),
            w_out_NDD=jnp.array(params.w_out_NDD),
            ln2_scale_ND=jnp.array(params.ln2_scale_ND),
            ln2_bias_ND=jnp.array(params.ln2_bias_ND),
            w_ffn1_NDF=jnp.array(params.w_ffn1_NDF),
            b_ffn1_NF=jnp.array(params.b_ffn1_NF),
            w_ffn2_NFD=jnp.array(params.w_ffn2_NFD),
            b_ffn2_ND=jnp.array(params.b_ffn2_ND),
            ln_f_scale_D=jnp.array(params.ln_f_scale_D),
            ln_f_bias_D=jnp.array(params.ln_f_bias_D),
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            n_embd=params.n_embd,
            vocab_size=params.vocab_size,
            max_seq_len=params.max_seq_len
        )
    
    elif backend == "torch":
        import torch
        return GPT2Params(
            wte_VD=torch.from_numpy(params.wte_VD),
            wpe_MD=torch.from_numpy(params.wpe_MD),
            ln1_scale_ND=torch.from_numpy(params.ln1_scale_ND),
            ln1_bias_ND=torch.from_numpy(params.ln1_bias_ND),
            w_qkv_N3DD=torch.from_numpy(params.w_qkv_N3DD),
            w_out_NDD=torch.from_numpy(params.w_out_NDD),
            ln2_scale_ND=torch.from_numpy(params.ln2_scale_ND),
            ln2_bias_ND=torch.from_numpy(params.ln2_bias_ND),
            w_ffn1_NDF=torch.from_numpy(params.w_ffn1_NDF),
            b_ffn1_NF=torch.from_numpy(params.b_ffn1_NF),
            w_ffn2_NFD=torch.from_numpy(params.w_ffn2_NFD),
            b_ffn2_ND=torch.from_numpy(params.b_ffn2_ND),
            ln_f_scale_D=torch.from_numpy(params.ln_f_scale_D),
            ln_f_bias_D=torch.from_numpy(params.ln_f_bias_D),
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            n_embd=params.n_embd,
            vocab_size=params.vocab_size,
            max_seq_len=params.max_seq_len
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose from 'jax', 'torch', or 'numpy'.")