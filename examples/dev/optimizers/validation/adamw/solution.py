#!/usr/bin/env python3
"""
AdamW optimizer JAX implementation solution.

Complete AdamW implementation that matches PyTorch exactly.
Key insight: AdamW applies weight decay directly to parameters, not through gradients.

Usage:
    python examples/optimizers/validation/adamw/compare.py --mode solution
"""

import jax.numpy as jnp
from typing import NamedTuple, Tuple


class AdamWState(NamedTuple):
    """AdamW optimizer state following JAX/Optax pattern"""
    step: int
    m: jnp.ndarray  # First moment (momentum)
    v: jnp.ndarray  # Second moment (variance)


def adamw_init(params: jnp.ndarray, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.01) -> AdamWState:
    """Initialize AdamW optimizer state"""
    return AdamWState(
        step=0,
        m=jnp.zeros_like(params),
        v=jnp.zeros_like(params)
    )


def adamw_update(
    grads: jnp.ndarray,
    state: AdamWState,
    params: jnp.ndarray,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01
) -> Tuple[jnp.ndarray, AdamWState]:
    """
    AdamW optimizer update step - SOLUTION IMPLEMENTATION.
    
    Key difference from Adam: AdamW applies weight decay directly to parameters,
    not through gradients. This "decouples" weight decay from gradient-based optimization.
    
    Args:
        grads: Gradients with same shape as params
        state: Current AdamW state
        params: Current parameters
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient (L2 regularization strength)
        
    Returns:
        updates: Parameter updates to apply (params += updates)
        new_state: Updated AdamW state
    """
    # Step 1: Increment step count
    step = state.step + 1
    
    # Step 2: Update biased first moment estimate (same as Adam)
    m = beta1 * state.m + (1 - beta1) * grads
    
    # Step 3: Update biased second raw moment estimate (same as Adam)
    v = beta2 * state.v + (1 - beta2) * grads**2
    
    # Step 4: Compute bias-corrected first moment estimate (same as Adam)
    bias_correction1 = 1 - beta1**step
    m_hat = m / bias_correction1
    
    # Step 5: Compute bias-corrected second raw moment estimate (same as Adam)
    bias_correction2 = 1 - beta2**step
    v_hat = v / bias_correction2
    
    # Step 6: KEY DIFFERENCE - AdamW applies weight decay directly to parameters
    # This creates two separate update components:
    # 1. Weight decay: shrinks parameters toward zero
    weight_decay_update = -lr * weight_decay * params
    
    # 2. Gradient-based update: standard Adam update based on gradients
    gradient_update = -lr * m_hat / (jnp.sqrt(v_hat) + eps)
    
    # Total update combines both components
    updates = weight_decay_update + gradient_update
    
    # Create new state
    new_state = AdamWState(step=step, m=m, v=v)
    
    return updates, new_state