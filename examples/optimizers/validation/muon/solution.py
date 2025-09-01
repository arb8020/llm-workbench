#!/usr/bin/env python3
"""
Muon optimizer JAX implementation solution.

Complete Muon implementation based on the paper algorithm.
Muon is a new optimizer that combines momentum with gradient orthogonalization.

Usage:
    python examples/optimizers/validation/muon/compare.py --mode solution
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class MuonState(NamedTuple):
    """Muon optimizer state"""
    step: int
    momentum: jnp.ndarray  # Momentum buffer
    
    
def muon_init(params: jnp.ndarray, lr: float = 0.02, momentum: float = 0.95, rank_ratio: float = 0.02) -> MuonState:
    """Initialize Muon optimizer state"""
    return MuonState(
        step=0,
        momentum=jnp.zeros_like(params)
    )


def orthogonalize_gradient(grad: jnp.ndarray, rank_ratio: float = 0.02) -> jnp.ndarray:
    """
    Apply orthogonalization to gradients using SVD decomposition.
    
    This is a key component of Muon that helps with conditioning.
    
    Args:
        grad: Gradient tensor to orthogonalize
        rank_ratio: Ratio of singular values to keep (controls rank)
        
    Returns:
        Orthogonalized gradient
    """
    if grad.ndim == 1:
        # For vectors, just normalize to prevent exploding gradients
        norm = jnp.linalg.norm(grad)
        return grad / (norm + 1e-8)
    
    # For matrices, apply SVD-based orthogonalization
    U, S, Vt = jnp.linalg.svd(grad, full_matrices=False)
    
    # Keep top singular values based on rank ratio
    k = max(1, int(rank_ratio * min(grad.shape)))
    
    # Reconstruct with reduced rank
    reconstructed = U[:, :k] @ jnp.diag(S[:k]) @ Vt[:k, :]
    
    return reconstructed


def muon_update(
    grads: jnp.ndarray,
    state: MuonState,
    params: jnp.ndarray,
    lr: float = 0.02,
    momentum: float = 0.95,
    rank_ratio: float = 0.02,
    orthogonalize: bool = True
) -> Tuple[jnp.ndarray, MuonState]:
    """
    Muon optimizer update step.
    
    Muon combines momentum with gradient orthogonalization for better conditioning.
    
    Args:
        grads: Gradients with same shape as params
        state: Current Muon state
        params: Current parameters
        lr: Learning rate
        momentum: Momentum coefficient
        rank_ratio: Ratio for gradient orthogonalization
        orthogonalize: Whether to apply gradient orthogonalization
        
    Returns:
        updates: Parameter updates to apply (params += updates)
        new_state: Updated Muon state
    """
    step = state.step + 1
    
    # Step 1: Orthogonalize gradients if enabled
    if orthogonalize:
        processed_grads = orthogonalize_gradient(grads, rank_ratio)
    else:
        processed_grads = grads
    
    # Step 2: Update momentum buffer
    # m_t = momentum * m_{t-1} + (1 - momentum) * g_orth
    momentum_buffer = momentum * state.momentum + (1 - momentum) * processed_grads
    
    # Step 3: Compute parameter updates
    updates = -lr * momentum_buffer
    
    # Step 4: Create new state
    new_state = MuonState(step=step, momentum=momentum_buffer)
    
    return updates, new_state