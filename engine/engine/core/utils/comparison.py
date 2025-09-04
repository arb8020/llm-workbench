"""
Simple logits comparison utility.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, LlamaForCausalLM, AutoModelForCausalLM
from typing import Dict, Any


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
