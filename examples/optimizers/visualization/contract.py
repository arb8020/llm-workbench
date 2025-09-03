#!/usr/bin/env python3
"""
Optimizer-Model Contract Definitions

This module defines the standard interface between optimizers and models
for loss surface visualization and multi-optimizer comparison.
"""

import jax
import jax.numpy as jnp
from typing import Protocol, NamedTuple, Tuple, Dict, Any, Callable
from abc import ABC, abstractmethod


class OptimizerState(Protocol):
    """Protocol for optimizer state objects (typically NamedTuples)"""
    step: int


class OptimizationModel(ABC):
    """Abstract base class for models that can be optimized"""
    
    @abstractmethod
    def loss(self, params: jnp.ndarray, batch: Any) -> float:
        """Compute loss for given parameters and batch"""
        pass
    
    @abstractmethod
    def grad(self, params: jnp.ndarray, batch: Any) -> jnp.ndarray:
        """Compute gradients for given parameters and batch"""
        pass
    
    @property
    @abstractmethod
    def param_shape(self) -> Tuple[int, ...]:
        """Shape of the parameter vector"""
        pass
    
    def init_params(self, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """Initialize parameters (override for custom initialization)"""
        if key is None:
            key = jax.random.PRNGKey(42)
        return jax.random.normal(key, self.param_shape) * 0.1


class TrajectoryData(NamedTuple):
    """Container for optimization trajectory data"""
    params: list[jnp.ndarray]      # Parameter values at each step
    losses: list[float]            # Loss values at each step
    gradients: list[jnp.ndarray]   # Gradient values at each step
    updates: list[jnp.ndarray]     # Parameter updates at each step
    step_sizes: list[float]        # Effective step sizes at each step
    metadata: Dict[str, Any]       # Optimizer-specific metadata


# Type aliases for optimizer functions
OptimizerInitFn = Callable[[jnp.ndarray, ...], OptimizerState]
OptimizerUpdateFn = Callable[[jnp.ndarray, OptimizerState, jnp.ndarray, ...], 
                           Tuple[jnp.ndarray, OptimizerState]]


class OptimizerConfig(NamedTuple):
    """Configuration for a single optimizer"""
    name: str
    init_fn: OptimizerInitFn
    update_fn: OptimizerUpdateFn
    kwargs: Dict[str, Any]
    color: str = 'blue'  # For visualization


def optimize_and_track(
    model: OptimizationModel,
    optimizer_init: OptimizerInitFn,
    optimizer_update: OptimizerUpdateFn,
    initial_params: jnp.ndarray,
    batches: list[Any],
    num_steps: int,
    **optimizer_kwargs
) -> TrajectoryData:
    """
    Run optimization and track trajectory data.
    
    Args:
        model: Model to optimize
        optimizer_init: Optimizer initialization function
        optimizer_update: Optimizer update function  
        initial_params: Initial parameter values
        batches: List of training batches
        num_steps: Number of optimization steps
        **optimizer_kwargs: Optimizer-specific parameters
        
    Returns:
        TrajectoryData containing optimization trajectory
    """
    # Initialize
    params = initial_params.copy()
    state = optimizer_init(params, **optimizer_kwargs)
    
    # Track trajectory
    trajectory_params = [params.copy()]
    trajectory_losses = [model.loss(params, batches[0])]
    trajectory_gradients = []
    trajectory_updates = []
    trajectory_step_sizes = []
    
    for step in range(num_steps):
        # Cycle through batches
        batch = batches[step % len(batches)]
        
        # Core optimizer contract
        grads = model.grad(params, batch)
        updates, state = optimizer_update(grads, state, params, **optimizer_kwargs)
        
        # Update parameters
        new_params = params + updates
        
        # Calculate effective step size
        step_size = jnp.linalg.norm(updates) / (jnp.linalg.norm(grads) + 1e-8)
        
        # Record trajectory
        trajectory_params.append(new_params.copy())
        trajectory_losses.append(model.loss(new_params, batch))
        trajectory_gradients.append(grads.copy())
        trajectory_updates.append(updates.copy())
        trajectory_step_sizes.append(float(step_size))
        
        # Update for next iteration
        params = new_params
    
    # Package metadata
    metadata = {
        'num_steps': num_steps,
        'final_loss': trajectory_losses[-1],
        'convergence_rate': trajectory_losses[0] / trajectory_losses[-1],
        'total_distance': sum(jnp.linalg.norm(u) for u in trajectory_updates),
        'optimizer_kwargs': optimizer_kwargs
    }
    
    return TrajectoryData(
        params=trajectory_params,
        losses=trajectory_losses,
        gradients=trajectory_gradients,
        updates=trajectory_updates,
        step_sizes=trajectory_step_sizes,
        metadata=metadata
    )


class SimpleQuadraticModel(OptimizationModel):
    """Simple quadratic model for testing: f(x) = x^T A x + b^T x + c"""
    
    def __init__(self, dim: int = 2, condition_number: float = 1.0):
        self.dim = dim
        self.condition_number = condition_number
        
        # Create condition number matrix
        eigenvals = jnp.linspace(1.0, condition_number, dim)
        # Random orthogonal matrix
        key = jax.random.PRNGKey(42)
        Q, _ = jnp.linalg.qr(jax.random.normal(key, (dim, dim)))
        self.A = Q @ jnp.diag(eigenvals) @ Q.T
        
        # Random linear term
        self.b = jax.random.normal(jax.random.PRNGKey(43), (dim,))
        self.c = 1.0
    
    def loss(self, params: jnp.ndarray, batch: Any = None) -> float:
        """Quadratic loss function"""
        return 0.5 * params.T @ self.A @ params + self.b.T @ params + self.c
    
    def grad(self, params: jnp.ndarray, batch: Any = None) -> jnp.ndarray:
        """Analytical gradient of quadratic function"""
        return self.A @ params + self.b
    
    @property
    def param_shape(self) -> Tuple[int, ...]:
        return (self.dim,)
    
    def optimal_params(self) -> jnp.ndarray:
        """Analytical optimum"""
        return -jnp.linalg.solve(self.A, self.b)


class RosenbrockModel(OptimizationModel):
    """Rosenbrock function: challenging non-convex optimization problem"""
    
    def __init__(self, dim: int = 2):
        assert dim >= 2, "Rosenbrock function requires at least 2 dimensions"
        self.dim = dim
    
    def loss(self, params: jnp.ndarray, batch: Any = None) -> float:
        """Rosenbrock function"""
        loss_val = 0.0
        for i in range(self.dim - 1):
            loss_val += 100 * (params[i+1] - params[i]**2)**2 + (1 - params[i])**2
        return loss_val
    
    def grad(self, params: jnp.ndarray, batch: Any = None) -> jnp.ndarray:
        """Analytical gradient of Rosenbrock function"""
        grad = jnp.zeros_like(params)
        
        # Interior points
        for i in range(self.dim - 1):
            grad = grad.at[i].add(-400 * params[i] * (params[i+1] - params[i]**2) - 2 * (1 - params[i]))
            grad = grad.at[i+1].add(200 * (params[i+1] - params[i]**2))
            
        return grad
    
    @property
    def param_shape(self) -> Tuple[int, ...]:
        return (self.dim,)
    
    def optimal_params(self) -> jnp.ndarray:
        """Global optimum is at (1, 1, ..., 1)"""
        return jnp.ones(self.dim)