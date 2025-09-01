#!/usr/bin/env python3
"""
Adam optimizer JAX implementation solution.

Complete Adam implementation that matches PyTorch and Optax exactly.

Usage:
    python examples/optimizers/validation/adam/compare.py --mode solution
"""

import jax.numpy as jnp
from typing import NamedTuple, Tuple


class AdamState(NamedTuple):
    """Adam optimizer state following JAX/Optax pattern"""
    step: int
    m: jnp.ndarray  # First moment (momentum)
    v: jnp.ndarray  # Second moment (variance)


def adam_init(params: jnp.ndarray, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> AdamState:
    """Initialize Adam optimizer state"""
    return AdamState(
        step=0,
        m=jnp.zeros_like(params),
        v=jnp.zeros_like(params)
    )


def adam_update(
    grads: jnp.ndarray,
    state: AdamState,
    params: jnp.ndarray,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> Tuple[jnp.ndarray, AdamState]:
    """
    Adam optimizer update step - SOLUTION IMPLEMENTATION.
    
    Args:
        grads: Gradients with same shape as params
        state: Current Adam state
        params: Current parameters
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small constant for numerical stability
        
    Returns:
        updates: Parameter updates to apply (params += updates)
        new_state: Updated Adam state
    """
    # Step 1: Increment step count
    step = state.step + 1
    
    # Step 2: Update biased first moment estimate
    m = beta1 * state.m + (1 - beta1) * grads
    
    # Step 3: Update biased second raw moment estimate  
    v = beta2 * state.v + (1 - beta2) * grads**2
    
    # Step 4: Compute bias-corrected first moment estimate
    bias_correction1 = 1 - beta1**step
    m_hat = m / bias_correction1
    
    # Step 5: Compute bias-corrected second raw moment estimate
    bias_correction2 = 1 - beta2**step
    v_hat = v / bias_correction2
    
    # Step 6: Compute parameter updates
    updates = -lr * m_hat / (jnp.sqrt(v_hat) + eps)
    
    # Create new state
    new_state = AdamState(step=step, m=m, v=v)
    
    return updates, new_state