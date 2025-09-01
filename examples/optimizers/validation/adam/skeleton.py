#!/usr/bin/env python3
"""
Adam optimizer JAX implementation skeleton.

Students should implement adam_update() to match reference implementations exactly.
This validates that our Adam optimizer produces identical trajectories to:
- PyTorch's torch.optim.Adam
- JAX's optax.adam

Usage:
    python examples/optimizers/validation/adam/skeleton.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, NamedTuple, Tuple
import torch
import optax


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
    Adam optimizer update step.
    
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
    # TODO: Implement Adam update logic
    # This should match PyTorch's torch.optim.Adam exactly
    # 
    # Adam algorithm:
    # 1. Increment step count
    # 2. Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    # 3. Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    # 4. Compute bias-corrected first moment estimate: m_hat = m_t / (1 - beta1^t)
    # 5. Compute bias-corrected second raw moment estimate: v_hat = v_t / (1 - beta2^t)
    # 6. Compute updates: updates = -lr * m_hat / (sqrt(v_hat) + eps)
    
    # Placeholder: return zero updates and unchanged state
    updates = jnp.zeros_like(params)
    new_state = state
    
    return updates, new_state


def rosenbrock(params: jnp.ndarray) -> float:
    """Rosenbrock function: classic optimization test case"""
    x, y = params[0], params[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_grad(params: jnp.ndarray) -> jnp.ndarray:
    """Analytical gradient of Rosenbrock function"""
    x, y = params[0], params[1]
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return jnp.array([dx, dy])


def get_reference_trajectories(initial_params: jnp.ndarray, lr: float, num_steps: int = 100):
    """Get reference trajectories from PyTorch and Optax"""
    
    # PyTorch Adam reference
    torch_params = torch.tensor(np.array(initial_params), requires_grad=True)
    torch_optimizer = torch.optim.Adam([torch_params], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    torch_trajectory = []
    
    for _ in range(num_steps):
        torch_optimizer.zero_grad()
        # Convert to torch for gradient computation
        loss = torch.tensor(rosenbrock(torch_params.detach().numpy()))
        grads = torch.tensor(rosenbrock_grad(torch_params.detach().numpy()))
        torch_params.grad = grads
        torch_optimizer.step()
        torch_trajectory.append(torch_params.detach().numpy().copy())
    
    # Optax Adam reference
    optax_optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)
    optax_state = optax_optimizer.init(initial_params)
    optax_params = initial_params
    optax_trajectory = []
    
    for _ in range(num_steps):
        grads = rosenbrock_grad(optax_params)
        updates, optax_state = optax_optimizer.update(grads, optax_state, optax_params)
        optax_params = optax_params + updates
        optax_trajectory.append(optax_params.copy())
    
    return torch_trajectory, optax_trajectory


def test_adam_implementation():
    """Test our Adam implementation against references"""
    print("🧪 Testing Adam implementation...")
    
    # Test parameters
    initial_params = jnp.array([-2.0, 2.0])
    lr = 0.01
    num_steps = 100
    
    # Get reference trajectories
    print("📋 Computing reference trajectories...")
    torch_traj, optax_traj = get_reference_trajectories(initial_params, lr, num_steps)
    
    # Test our implementation
    print("⚡ Testing our Adam implementation...")
    our_state = adam_init(initial_params, lr=lr)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = rosenbrock_grad(our_params)
        updates, our_state = adam_update(grads, our_state, our_params, lr=lr)
        our_params = our_params + updates
        our_trajectory.append(our_params.copy())
    
    # Compare trajectories
    print("📊 Comparing trajectories...")
    
    # Compare with PyTorch
    torch_diff = np.mean([np.linalg.norm(np.array(our_trajectory[i]) - torch_traj[i]) 
                         for i in range(num_steps)])
    
    # Compare with Optax
    optax_diff = np.mean([np.linalg.norm(our_trajectory[i] - optax_traj[i]) 
                         for i in range(num_steps)])
    
    print(f"📈 Average trajectory difference vs PyTorch: {torch_diff:.2e}")
    print(f"📈 Average trajectory difference vs Optax: {optax_diff:.2e}")
    
    # Success criteria
    tolerance = 1e-6
    torch_match = torch_diff < tolerance
    optax_match = optax_diff < tolerance
    
    print(f"✅ PyTorch match: {torch_match} (diff: {torch_diff:.2e})")
    print(f"✅ Optax match: {optax_match} (diff: {optax_diff:.2e})")
    
    if torch_match and optax_match:
        print("🎉 SUCCESS! Adam implementation matches references!")
        return True
    else:
        print("❌ FAILED! Implementation doesn't match references.")
        print("💡 Hint: Check your bias correction and update formulas.")
        return False


if __name__ == "__main__":
    test_adam_implementation()