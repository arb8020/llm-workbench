"""
Simple logits comparison utility.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, LlamaForCausalLM, AutoModelForCausalLM
from typing import Dict, Any
from pathlib import Path
try:
    import llama_stack
    LLAMA_STACK_AVAILABLE = True
except ImportError:
    LLAMA_STACK_AVAILABLE = False


def compare_logits(
    logits1: np.ndarray,
    logits2: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare two sets of logits for numerical similarity.
    
    Args:
        logits1: First set of logits
        logits2: Second set of logits  
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        verbose: If True, print detailed comparison
    
    Returns:
        Dictionary with comparison metrics
    """
    # Convert to numpy if needed
    if hasattr(logits1, 'numpy'):
        logits1 = logits1.numpy()
    if hasattr(logits2, 'numpy'):
        logits2 = logits2.numpy()
    
    # Ensure same shape
    assert logits1.shape == logits2.shape, f"Shape mismatch: {logits1.shape} vs {logits2.shape}"
    
    # Compute differences
    abs_diff = np.abs(logits1 - logits2)
    rel_diff = abs_diff / (np.abs(logits2) + 1e-10)
    
    # Check if close
    all_close = np.allclose(logits1, logits2, rtol=rtol, atol=atol)
    
    # Get top-k accuracy (do the models predict the same top tokens?)
    top1_match = np.mean(logits1.argmax(-1) == logits2.argmax(-1))
    
    results = {
        'all_close': all_close,
        'max_abs_diff': float(abs_diff.max()),
        'mean_abs_diff': float(abs_diff.mean()),
        'max_rel_diff': float(rel_diff.max()),
        'mean_rel_diff': float(rel_diff.mean()),
        'top1_accuracy': float(top1_match),
    }
    
    if verbose:
        print("Logits Comparison:")
        print(f"  All close (rtol={rtol}, atol={atol}): {all_close}")
        print(f"  Max absolute difference: {results['max_abs_diff']:.2e}")
        print(f"  Mean absolute difference: {results['mean_abs_diff']:.2e}")
        print(f"  Top-1 token match: {results['top1_accuracy']:.1%}")
    
    return results


def get_hf_logits(input_ids_BL: np.ndarray, model_name: str = "gpt2") -> np.ndarray:
    """
    Get logits from HuggingFace model (supports GPT-2, Llama, and other causal LM models).
    
    Args:
        input_ids_BL: Input token IDs of shape (batch_size, seq_len)
        model_name: Name of the model (e.g., "gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    Returns:
        Logits array of shape (batch_size, seq_len, vocab_size)
    """
    print(f"🤗 Loading HuggingFace model: {model_name}")
    
    # Use AutoModelForCausalLM to automatically detect the correct model type
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True  # Required for some models
    )
    model.eval()
    
    # Convert to torch tensor
    input_ids_torch = torch.from_numpy(input_ids_BL).long()
    
    print(f"🔥 Running inference on {input_ids_torch.shape} tokens")
    with torch.no_grad():
        outputs = model(input_ids_torch)
    
    logits = outputs.logits.numpy()
    
    # Clean up to save memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return logits


def get_transformers_from_local_checkpoint(input_ids_BL: np.ndarray, model_name: str = "Llama-3.2-1B-Instruct") -> np.ndarray:
    """
    Load local checkpoint into official transformers LlamaForCausalLM and run inference.
    
    Args:
        input_ids_BL: Input token IDs of shape (batch_size, seq_len)  
        model_name: Name of the local llama-stack model
    
    Returns:
        Logits array of shape (batch_size, seq_len, vocab_size)
    """
    from transformers import LlamaForCausalLM, LlamaConfig
    
    print(f"🤗 Loading local checkpoint into official transformers: {model_name}")
    
    # Convert HuggingFace naming to llama-stack naming
    local_model_name = model_name.replace('meta-llama/', '').replace('Llama-', 'Llama')
    checkpoint_path = Path(f"~/.llama/checkpoints/{local_model_name}/consolidated.00.pth").expanduser()
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Local checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    print(f"📦 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Convert BFloat16 to Float32 for compatibility
    for key in checkpoint:
        if checkpoint[key].dtype == torch.bfloat16:
            checkpoint[key] = checkpoint[key].to(torch.float32)
    
    # Create LlamaConfig for Llama-3.2-1B
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048, 
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=128000,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        torch_dtype=torch.float32,
    )
    
    # Create model with config
    print("🏗️ Creating LlamaForCausalLM with local config")
    model = LlamaForCausalLM(config)
    
    # Convert entropix checkpoint format to transformers format
    print("🔄 Converting checkpoint format...")
    transformers_state_dict = {}
    
    # Token embeddings
    transformers_state_dict['model.embed_tokens.weight'] = checkpoint['tok_embeddings.weight']
    
    # Output projection
    transformers_state_dict['lm_head.weight'] = checkpoint['output.weight']
    
    # Layer norm
    transformers_state_dict['model.norm.weight'] = checkpoint['norm.weight']
    
    # Layer weights
    for i in range(16):  # 16 layers for Llama-3.2-1B
        # Attention norms
        transformers_state_dict[f'model.layers.{i}.input_layernorm.weight'] = checkpoint[f'layers.{i}.attention_norm.weight']
        transformers_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = checkpoint[f'layers.{i}.ffn_norm.weight']
        
        # Attention weights (transpose back since entropix uses transposed format)
        transformers_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = checkpoint[f'layers.{i}.attention.wq.weight']
        transformers_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = checkpoint[f'layers.{i}.attention.wk.weight'] 
        transformers_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = checkpoint[f'layers.{i}.attention.wv.weight']
        transformers_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = checkpoint[f'layers.{i}.attention.wo.weight']
        
        # FFN weights
        transformers_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = checkpoint[f'layers.{i}.feed_forward.w1.weight']
        transformers_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = checkpoint[f'layers.{i}.feed_forward.w3.weight']
        transformers_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = checkpoint[f'layers.{i}.feed_forward.w2.weight']
    
    # Load state dict into model
    print("⚡ Loading weights into transformers model...")
    model.load_state_dict(transformers_state_dict, strict=True)
    model.eval()
    
    # Run inference
    input_ids = torch.from_numpy(input_ids_BL).long()
    print(f"🔥 Running official transformers inference on {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits.detach().cpu().numpy()
    print(f"✅ Transformers inference complete, logits shape: {logits.shape}")
    
    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return logits


def get_local_pytorch_llama_logits(input_ids_BL: np.ndarray, model_name: str = "Llama-3.2-1B-Instruct") -> np.ndarray:
    """
    Get logits from local llama-stack checkpoint using minimal PyTorch implementation.
    
    Args:
        input_ids_BL: Input token IDs of shape (batch_size, seq_len)  
        model_name: Name of the local llama-stack model
    
    Returns:
        Logits array of shape (batch_size, seq_len, vocab_size)
    """
    print(f"🦙 Loading local PyTorch Llama from checkpoint: {model_name}")
    
    # Convert HuggingFace naming to llama-stack naming
    local_model_name = model_name.replace('meta-llama/', '').replace('Llama-', 'Llama')
    checkpoint_path = Path(f"~/.llama/checkpoints/{local_model_name}/consolidated.00.pth").expanduser()
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Local checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    print(f"📦 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Convert BFloat16 to Float32 for compatibility
    for key in checkpoint:
        if checkpoint[key].dtype == torch.bfloat16:
            checkpoint[key] = checkpoint[key].to(torch.float32)
    
    # Model parameters for Llama-3.2-1B
    n_layers = 16
    dim = 2048
    n_heads = 32
    n_kv_heads = 8
    vocab_size = 128256
    norm_eps = 1e-5
    rope_theta = 500000.0
    
    batch_size, seq_len = input_ids_BL.shape
    input_ids = torch.from_numpy(input_ids_BL).long()
    
    # Simple forward pass implementation
    def rms_norm(x, weight, eps=1e-5):
        variance = x.pow(2).mean(-1, keepdims=True)
        return x * torch.rsqrt(variance + eps) * weight
    
    def apply_rotary_pos_emb(q, k, seq_len):
        # Simplified RoPE - just return as-is for validation
        return q, k
    
    # Token embeddings
    h = checkpoint['tok_embeddings.weight'][input_ids]  # [batch, seq, dim]
    
    # Transformer layers
    for layer_idx in range(n_layers):
        # Pre-attention norm
        h_norm = rms_norm(h, checkpoint[f'layers.{layer_idx}.attention_norm.weight'], norm_eps)
        
        # Attention projections
        q = h_norm @ checkpoint[f'layers.{layer_idx}.attention.wq.weight'].T  # [batch, seq, dim]
        k = h_norm @ checkpoint[f'layers.{layer_idx}.attention.wk.weight'].T  # [batch, seq, kv_dim]  
        v = h_norm @ checkpoint[f'layers.{layer_idx}.attention.wv.weight'].T  # [batch, seq, kv_dim]
        
        # Reshape for multi-head attention
        head_dim = dim // n_heads
        kv_dim = n_kv_heads * head_dim
        
        q = q.view(batch_size, seq_len, n_heads, head_dim)
        k = k.view(batch_size, seq_len, n_kv_heads, head_dim) 
        v = v.view(batch_size, seq_len, n_kv_heads, head_dim)
        
        # Expand k, v for grouped query attention
        n_rep = n_heads // n_kv_heads
        k = k.repeat_interleave(n_rep, dim=2)  # [batch, seq, n_heads, head_dim]
        v = v.repeat_interleave(n_rep, dim=2)  # [batch, seq, n_heads, head_dim]
        
        # Attention computation
        q = q.transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        k = k.transpose(1, 2)  # [batch, n_heads, seq, head_dim] 
        v = v.transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # [batch, n_heads, seq, head_dim]
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        attn_out = attn_out @ checkpoint[f'layers.{layer_idx}.attention.wo.weight'].T
        
        # Residual connection
        h = h + attn_out
        
        # Pre-FFN norm
        h_norm = rms_norm(h, checkpoint[f'layers.{layer_idx}.ffn_norm.weight'], norm_eps)
        
        # SwiGLU FFN
        gate = h_norm @ checkpoint[f'layers.{layer_idx}.feed_forward.w1.weight'].T
        up = h_norm @ checkpoint[f'layers.{layer_idx}.feed_forward.w3.weight'].T
        hidden = F.silu(gate) * up
        ffn_out = hidden @ checkpoint[f'layers.{layer_idx}.feed_forward.w2.weight'].T
        
        # Residual connection
        h = h + ffn_out
    
    # Final norm and output projection
    h = rms_norm(h, checkpoint['norm.weight'], norm_eps)
    logits = h @ checkpoint['output.weight'].T  # [batch, seq, vocab]
    
    print(f"✅ PyTorch inference complete, logits shape: {logits.shape}")
    return logits.detach().cpu().numpy()


def get_llama_stack_logits(input_ids_BL: np.ndarray, model_name: str = "Llama-3.2-1B-Instruct") -> np.ndarray:
    """
    Get logits from local llama-stack model using official transformers.
    
    Args:
        input_ids_BL: Input token IDs of shape (batch_size, seq_len)  
        model_name: Name of the local llama-stack model
    
    Returns:
        Logits array of shape (batch_size, seq_len, vocab_size)
    """
    print(f"🦙 Using local llama-stack checkpoint with official transformers")
    
    # Use official transformers loaded from local checkpoint
    return get_transformers_from_local_checkpoint(input_ids_BL, model_name)


def get_reference_logits(input_ids_BL: np.ndarray, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
                        use_llama_stack: bool = True) -> np.ndarray:
    """
    Get reference logits from either llama-stack or HuggingFace.
    
    Args:
        input_ids_BL: Input token IDs of shape (batch_size, seq_len)
        model_name: Name of the model
        use_llama_stack: If True, try llama-stack first, fall back to HuggingFace
    
    Returns:
        Logits array of shape (batch_size, seq_len, vocab_size)
    """
    if use_llama_stack:
        try:
            print(f"🦙 Attempting to use local llama-stack checkpoint")
            # Extract model name for llama-stack (remove meta-llama/ prefix)
            stack_model_name = model_name.replace("meta-llama/", "")
            return get_llama_stack_logits(input_ids_BL, stack_model_name)
        except Exception as e:
            print(f"⚠️  llama-stack failed with error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"🔄 Falling back to HuggingFace...")
            return get_hf_logits(input_ids_BL, model_name)
    else:
        print(f"🤗 Using HuggingFace directly")
        return get_hf_logits(input_ids_BL, model_name)
