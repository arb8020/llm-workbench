#!/usr/bin/env python3
"""
AdamW optimizer JAX implementation skeleton.

Students should implement adamw_update() to match reference implementations exactly.
AdamW differs from Adam by applying weight decay directly to parameters rather than gradients.

Usage:
    python examples/optimizers/validation/adamw/skeleton.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, NamedTuple, Tuple
import torch
import optax


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
    AdamW optimizer update step.
    
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
    # TODO: Implement AdamW update logic
    # This should match PyTorch's torch.optim.AdamW exactly
    # 
    # AdamW algorithm (key difference from Adam):
    # 1. Increment step count
    # 2. Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    # 3. Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    # 4. Compute bias-corrected first moment estimate: m_hat = m_t / (1 - beta1^t)
    # 5. Compute bias-corrected second raw moment estimate: v_hat = v_t / (1 - beta2^t)
    # 6. Apply weight decay DIRECTLY to parameters: theta = theta * (1 - lr * weight_decay)
    # 7. Compute gradient-based updates: updates = -lr * m_hat / (sqrt(v_hat) + eps)
    # 8. IMPORTANT: Weight decay is applied separately from gradient updates
    
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


def quadratic_with_l2(params: jnp.ndarray, l2_coef: float = 0.1) -> float:
    """Quadratic function with L2 regularization to test weight decay"""
    x, y = params[0], params[1]
    return x**2 + 10 * y**2 + l2_coef * (x**2 + y**2)


def quadratic_l2_grad(params: jnp.ndarray, l2_coef: float = 0.1) -> jnp.ndarray:
    """Gradient of quadratic with L2 regularization"""
    x, y = params[0], params[1]
    return jnp.array([2 * x + 2 * l2_coef * x, 20 * y + 2 * l2_coef * y])


def get_reference_trajectories(
    initial_params: jnp.ndarray, 
    lr: float, 
    weight_decay: float,
    surface_fn, 
    grad_fn,
    num_steps: int = 100
):
    """Get reference trajectories from PyTorch and Optax"""
    
    # PyTorch AdamW reference
    torch_params = torch.tensor(np.array(initial_params), requires_grad=True)
    torch_optimizer = torch.optim.AdamW([torch_params], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    torch_trajectory = []
    
    for _ in range(num_steps):
        torch_optimizer.zero_grad()
        grads = torch.tensor(grad_fn(torch_params.detach().numpy()))
        torch_params.grad = grads
        torch_optimizer.step()
        torch_trajectory.append(torch_params.detach().numpy().copy())
    
    # Optax AdamW reference
    optax_optimizer = optax.adamw(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=weight_decay)
    optax_state = optax_optimizer.init(initial_params)
    optax_params = initial_params
    optax_trajectory = []
    
    for _ in range(num_steps):
        grads = grad_fn(optax_params)
        updates, optax_state = optax_optimizer.update(grads, optax_state, optax_params)
        optax_params = optax_params + updates
        optax_trajectory.append(optax_params.copy())
    
    return torch_trajectory, optax_trajectory


def test_adamw_implementation():
    """Test our AdamW implementation against references"""
    print("🧪 Testing AdamW implementation...")
    
    # Test parameters
    initial_params = jnp.array([2.0, -2.0])  # Start away from origin to see weight decay effect
    lr = 0.01
    weight_decay = 0.01
    num_steps = 100
    
    # Get reference trajectories
    print("📋 Computing reference trajectories...")
    torch_traj, optax_traj = get_reference_trajectories(
        initial_params, lr, weight_decay, quadratic_with_l2, quadratic_l2_grad, num_steps
    )
    
    # Test our implementation
    print("⚡ Testing our AdamW implementation...")
    our_state = adamw_init(initial_params, lr=lr, weight_decay=weight_decay)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = quadratic_l2_grad(our_params)
        updates, our_state = adamw_update(grads, our_state, our_params, lr=lr, weight_decay=weight_decay)
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
    
    # Test weight decay effect
    print("\n🎯 Testing weight decay effect...")
    
    # Compare AdamW (with weight decay) vs Adam (without weight decay) 
    # Both should converge to origin, but AdamW should get there faster due to weight decay
    final_params = our_trajectory[-1]
    final_norm = jnp.linalg.norm(final_params)
    
    print(f"Final parameter norm: {final_norm:.6f}")
    print(f"Expected: smaller norm due to weight decay shrinking parameters")
    
    # Success criteria
    tolerance = 1e-6
    torch_match = torch_diff < tolerance
    optax_match = optax_diff < tolerance
    
    print(f"✅ PyTorch match: {torch_match} (diff: {torch_diff:.2e})")
    print(f"✅ Optax match: {optax_match} (diff: {optax_diff:.2e})")
    
    if torch_match and optax_match:
        print("🎉 SUCCESS! AdamW implementation matches references!")
        print("💡 Key insight: Weight decay applied directly to parameters, not gradients")
        return True
    else:
        print("❌ FAILED! Implementation doesn't match references.")
        print("💡 Hint: Make sure weight decay is applied to parameters BEFORE gradient updates")
        print("💡 Formula: params = params * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)")
        return False


if __name__ == "__main__":
    test_adamw_implementation()