#!/usr/bin/env python3
"""
SGD optimizer JAX implementation skeleton.

Students should implement sgd_update() to match reference implementations exactly.
SGD is the fundamental optimizer that all others build upon.

Usage:
    python examples/optimizers/validation/sgd/skeleton.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, NamedTuple, Tuple
import torch
import optax


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
    SGD optimizer update step.
    
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
    # TODO: Implement SGD update logic
    # This should match PyTorch's torch.optim.SGD exactly
    # 
    # SGD algorithm:
    # 1. Apply weight decay if weight_decay > 0: g_t = g_t + weight_decay * p_t
    # 2. If momentum > 0:
    #    - If first step: buf = g_t
    #    - Else: buf = momentum * buf + (1 - dampening) * g_t  
    #    - If nesterov: g_t = g_t + momentum * buf
    #    - Else: g_t = buf
    # 3. Apply update: p_t = p_t - lr * g_t
    
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


def quadratic_bowl(params: jnp.ndarray) -> float:
    """Simple quadratic function: x^2 + 10*y^2"""
    x, y = params[0], params[1]
    return x**2 + 10 * y**2


def quadratic_grad(params: jnp.ndarray) -> jnp.ndarray:
    """Analytical gradient of quadratic bowl"""
    x, y = params[0], params[1]
    return jnp.array([2 * x, 20 * y])


def get_reference_trajectories(
    initial_params: jnp.ndarray, 
    lr: float, 
    momentum: float,
    weight_decay: float,
    surface_fn, 
    grad_fn,
    num_steps: int = 100,
    nesterov: bool = False
):
    """Get reference trajectories from PyTorch and Optax"""
    
    # PyTorch SGD reference
    torch_params = torch.tensor(np.array(initial_params), requires_grad=True)
    torch_optimizer = torch.optim.SGD(
        [torch_params], 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay,
        nesterov=nesterov
    )
    torch_trajectory = []
    
    for _ in range(num_steps):
        torch_optimizer.zero_grad()
        grads = torch.tensor(grad_fn(torch_params.detach().numpy()))
        torch_params.grad = grads
        torch_optimizer.step()
        torch_trajectory.append(torch_params.detach().numpy().copy())
    
    # Optax SGD reference
    if momentum > 0 and nesterov:
        optax_optimizer = optax.sgd(lr, momentum=momentum, nesterov=True)
    elif momentum > 0:
        optax_optimizer = optax.sgd(lr, momentum=momentum)
    else:
        optax_optimizer = optax.sgd(lr)
        
    # Add weight decay if specified
    if weight_decay > 0:
        optax_optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax_optimizer
        )
    
    optax_state = optax_optimizer.init(initial_params)
    optax_params = initial_params
    optax_trajectory = []
    
    for _ in range(num_steps):
        grads = grad_fn(optax_params)
        updates, optax_state = optax_optimizer.update(grads, optax_state, optax_params)
        optax_params = optax_params + updates
        optax_trajectory.append(optax_params.copy())
    
    return torch_trajectory, optax_trajectory


def test_vanilla_sgd():
    """Test vanilla SGD (no momentum)"""
    print("🏃 Testing Vanilla SGD (no momentum)...")
    
    initial_params = jnp.array([2.0, -1.0])
    lr = 0.01
    num_steps = 100
    
    # Get reference trajectories
    torch_traj, optax_traj = get_reference_trajectories(
        initial_params, lr, momentum=0.0, weight_decay=0.0,
        quadratic_bowl, quadratic_grad, num_steps
    )
    
    # Test our implementation
    our_state = sgd_init(initial_params, lr=lr)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = quadratic_grad(our_params)
        updates, our_state = sgd_update(grads, our_state, our_params, lr=lr)
        our_params = our_params + updates
        our_trajectory.append(our_params.copy())
    
    # Compare trajectories
    torch_diff = np.mean([np.linalg.norm(np.array(our_trajectory[i]) - torch_traj[i]) 
                         for i in range(num_steps)])
    
    optax_diff = np.mean([np.linalg.norm(our_trajectory[i] - optax_traj[i]) 
                         for i in range(num_steps)])
    
    print(f"   📈 Trajectory diff vs PyTorch: {torch_diff:.2e}")
    print(f"   📈 Trajectory diff vs Optax: {optax_diff:.2e}")
    
    # Success criteria
    tolerance = 1e-6
    torch_match = torch_diff < tolerance
    optax_match = optax_diff < tolerance
    
    if torch_match and optax_match:
        print("   ✅ PASS: Vanilla SGD matches references")
        return True
    else:
        print("   ❌ FAIL: Vanilla SGD doesn't match references")
        return False


def test_momentum_sgd():
    """Test SGD with momentum"""
    print("\n🏃‍♂️ Testing SGD with Momentum...")
    
    initial_params = jnp.array([-1.0, 2.0])
    lr = 0.01
    momentum = 0.9
    num_steps = 100
    
    # Get reference trajectories
    torch_traj, optax_traj = get_reference_trajectories(
        initial_params, lr, momentum=momentum, weight_decay=0.0,
        quadratic_bowl, quadratic_grad, num_steps
    )
    
    # Test our implementation
    our_state = sgd_init(initial_params, lr=lr, momentum=momentum)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = quadratic_grad(our_params)
        updates, our_state = sgd_update(grads, our_state, our_params, lr=lr, momentum=momentum)
        our_params = our_params + updates
        our_trajectory.append(our_params.copy())
    
    # Compare trajectories
    torch_diff = np.mean([np.linalg.norm(np.array(our_trajectory[i]) - torch_traj[i]) 
                         for i in range(num_steps)])
    
    optax_diff = np.mean([np.linalg.norm(our_trajectory[i] - optax_traj[i]) 
                         for i in range(num_steps)])
    
    print(f"   📈 Trajectory diff vs PyTorch: {torch_diff:.2e}")
    print(f"   📈 Trajectory diff vs Optax: {optax_diff:.2e}")
    
    # Success criteria
    tolerance = 1e-6
    torch_match = torch_diff < tolerance
    optax_match = optax_diff < tolerance
    
    if torch_match and optax_match:
        print("   ✅ PASS: Momentum SGD matches references")
        return True
    else:
        print("   ❌ FAIL: Momentum SGD doesn't match references")
        return False


def test_sgd_implementation():
    """Test our SGD implementation against references"""
    print("🧪 Testing SGD implementation...")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test vanilla SGD
    vanilla_success = test_vanilla_sgd()
    all_tests_passed = all_tests_passed and vanilla_success
    
    # Test momentum SGD
    momentum_success = test_momentum_sgd()
    all_tests_passed = all_tests_passed and momentum_success
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 SUCCESS! SGD implementation matches references!")
        print("✅ Vanilla SGD working correctly")
        print("✅ Momentum SGD working correctly") 
        print("\n💡 SGD is the foundation that all other optimizers build upon!")
        return True
    else:
        print("❌ FAILED! Implementation doesn't match references.")
        print("💡 Hint: Check gradient application and momentum buffer updates")
        print("💡 Formula: params = params - lr * (grads or momentum_buffer)")
        return False


if __name__ == "__main__":
    test_sgd_implementation()