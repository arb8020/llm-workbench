#!/usr/bin/env python3
"""
SGD optimizer JAX implementation solution.

Complete SGD implementation that matches PyTorch exactly.
SGD is the foundation that all other optimizers build upon.

Usage:
    python examples/optimizers/validation/sgd/compare.py --mode solution
"""

import jax.numpy as jnp
from typing import NamedTuple, Tuple


class SGDState(NamedTuple):
    """SGD optimizer state following JAX/Optax pattern"""
    step: int
    momentum: jnp.ndarray  # Momentum buffer (zero for vanilla SGD)


def sgd_init(params: jnp.ndarray, lr: float = 0.01, momentum: float = 0.0) -> SGDState:
    """Initialize SGD optimizer state"""
    return SGDState(
        step=0,
        momentum=jnp.zeros_like(params)
    )


def sgd_update(
    grads: jnp.ndarray,
    state: SGDState,
    params: jnp.ndarray,
    lr: float = 0.01,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False
) -> Tuple[jnp.ndarray, SGDState]:
    """
    SGD optimizer update step - SOLUTION IMPLEMENTATION.
    
    Args:
        grads: Gradients with same shape as params
        state: Current SGD state
        params: Current parameters
        lr: Learning rate
        momentum: Momentum coefficient (0 = vanilla SGD)
        weight_decay: Weight decay coefficient (L2 regularization)
        dampening: Dampening for momentum
        nesterov: Whether to use Nesterov momentum
        
    Returns:
        updates: Parameter updates to apply (params += updates)
        new_state: Updated SGD state
    """
    step = state.step + 1
    
    # Step 1: Apply weight decay to gradients (L2 regularization)
    if weight_decay > 0:
        grads = grads + weight_decay * params
    
    # Step 2: Handle momentum
    if momentum > 0:
        if state.step == 0:
            # First step: initialize momentum buffer with gradients
            momentum_buffer = grads
        else:
            # Update momentum buffer: m_t = momentum * m_{t-1} + (1 - dampening) * g_t
            momentum_buffer = momentum * state.momentum + (1 - dampening) * grads
        
        # Apply Nesterov correction if enabled
        if nesterov:
            # Nesterov momentum: g_t = g_t + momentum * m_t
            effective_grad = grads + momentum * momentum_buffer
        else:
            # Standard momentum: use momentum buffer directly
            effective_grad = momentum_buffer
    else:
        # Vanilla SGD: use gradients directly, no momentum buffer
        effective_grad = grads
        momentum_buffer = state.momentum  # Keep existing (zero) momentum
    
    # Step 3: Compute parameter updates
    updates = -lr * effective_grad
    
    # Create new state
    new_state = SGDState(step=step, momentum=momentum_buffer)
    
    return updates, new_state