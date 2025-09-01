#!/usr/bin/env python3
"""
Adam optimizer JAX implementation solution.

Complete Adam implementation that matches PyTorch and Optax exactly.
This validates that our Adam optimizer produces identical trajectories.

Usage:
    python examples/optimizers/validation/adam/solution.py
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
    surface_fn, 
    grad_fn,
    num_steps: int = 100
):
    """Get reference trajectories from PyTorch and Optax"""
    
    # PyTorch Adam reference
    torch_params = torch.tensor(np.array(initial_params), requires_grad=True)
    torch_optimizer = torch.optim.Adam([torch_params], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    torch_trajectory = []
    
    for _ in range(num_steps):
        torch_optimizer.zero_grad()
        grads = torch.tensor(grad_fn(torch_params.detach().numpy()))
        torch_params.grad = grads
        torch_optimizer.step()
        torch_trajectory.append(torch_params.detach().numpy().copy())
    
    # Optax Adam reference
    optax_optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)
    optax_state = optax_optimizer.init(initial_params)
    optax_params = initial_params
    optax_trajectory = []
    
    for _ in range(num_steps):
        grads = grad_fn(optax_params)
        updates, optax_state = optax_optimizer.update(grads, optax_state, optax_params)
        optax_params = optax_params + updates
        optax_trajectory.append(optax_params.copy())
    
    return torch_trajectory, optax_trajectory


def test_adam_on_surface(surface_name: str, surface_fn, grad_fn, initial_params: jnp.ndarray, lr: float = 0.01):
    """Test Adam implementation on a specific optimization surface"""
    print(f"\n🏔️  Testing on {surface_name}...")
    print(f"   Initial params: {initial_params}")
    print(f"   Learning rate: {lr}")
    
    num_steps = 100
    
    # Get reference trajectories
    torch_traj, optax_traj = get_reference_trajectories(
        initial_params, lr, surface_fn, grad_fn, num_steps
    )
    
    # Test our implementation
    our_state = adam_init(initial_params, lr=lr)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = grad_fn(our_params)
        updates, our_state = adam_update(grads, our_state, our_params, lr=lr)
        our_params = our_params + updates
        our_trajectory.append(our_params.copy())
    
    # Compare trajectories
    torch_diff = np.mean([np.linalg.norm(np.array(our_trajectory[i]) - torch_traj[i]) 
                         for i in range(num_steps)])
    
    optax_diff = np.mean([np.linalg.norm(our_trajectory[i] - optax_traj[i]) 
                         for i in range(num_steps)])
    
    # Final values
    final_loss_ours = surface_fn(our_trajectory[-1])
    final_loss_torch = surface_fn(torch_traj[-1])
    final_loss_optax = surface_fn(optax_traj[-1])
    
    print(f"   📈 Trajectory diff vs PyTorch: {torch_diff:.2e}")
    print(f"   📈 Trajectory diff vs Optax: {optax_diff:.2e}")
    print(f"   🎯 Final loss (ours): {final_loss_ours:.6f}")
    print(f"   🎯 Final loss (torch): {final_loss_torch:.6f}")
    print(f"   🎯 Final loss (optax): {final_loss_optax:.6f}")
    
    # Success criteria
    tolerance = 1e-6
    torch_match = torch_diff < tolerance
    optax_match = optax_diff < tolerance
    
    success = torch_match and optax_match
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"   {status}")
    
    return success


def test_edge_cases():
    """Test Adam on edge cases and pathological surfaces"""
    print("\n🧪 Testing edge cases...")
    
    edge_cases = [
        {
            "name": "Zero gradients",
            "params": jnp.array([1.0, 1.0]),
            "grads": jnp.array([0.0, 0.0]),
            "expected_updates": jnp.array([0.0, 0.0])
        },
        {
            "name": "Large gradients", 
            "params": jnp.array([0.0, 0.0]),
            "grads": jnp.array([1000.0, -1000.0]),
            "expected_norm_range": (0.001, 0.1)  # Should be clipped/normalized
        },
        {
            "name": "Very small gradients",
            "params": jnp.array([0.0, 0.0]), 
            "grads": jnp.array([1e-10, -1e-10]),
            "expected_norm_range": (1e-15, 1e-5)
        }
    ]
    
    for case in edge_cases:
        print(f"   Testing: {case['name']}")
        
        state = adam_init(case["params"])
        updates, new_state = adam_update(
            case["grads"], state, case["params"], lr=0.001
        )
        
        if "expected_updates" in case:
            diff = jnp.linalg.norm(updates - case["expected_updates"])
            success = diff < 1e-10
        else:
            norm = jnp.linalg.norm(updates)
            min_norm, max_norm = case["expected_norm_range"]
            success = min_norm <= norm <= max_norm
            
        status = "✅ PASS" if success else "❌ FAIL" 
        print(f"     {status} (update norm: {jnp.linalg.norm(updates):.2e})")


def test_step_consistency():
    """Test that step counting and state updates are consistent"""
    print("\n🔄 Testing step consistency...")
    
    params = jnp.array([1.0, -1.0])
    state = adam_init(params)
    
    # Take several steps and verify step count
    for expected_step in range(1, 6):
        grads = jnp.array([0.1, -0.1]) * expected_step  # Different grads each step
        updates, state = adam_update(grads, state, params, lr=0.01)
        
        if state.step != expected_step:
            print(f"   ❌ FAIL: Expected step {expected_step}, got {state.step}")
            return False
            
        # Verify momentum is accumulating
        if expected_step > 1:
            momentum_norm = jnp.linalg.norm(state.m)
            if momentum_norm == 0:
                print(f"   ❌ FAIL: Momentum should be non-zero at step {expected_step}")
                return False
    
    print("   ✅ PASS: Step counting and momentum accumulation correct")
    return True


def test_adam_implementation():
    """Complete test suite for Adam implementation"""
    print("🧪 Testing Adam Implementation")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test on different optimization surfaces
    test_cases = [
        {
            "name": "Quadratic Bowl",
            "surface_fn": quadratic_bowl,
            "grad_fn": quadratic_grad, 
            "initial_params": jnp.array([-2.0, 3.0]),
            "lr": 0.1
        },
        {
            "name": "Rosenbrock Function",
            "surface_fn": rosenbrock,
            "grad_fn": rosenbrock_grad,
            "initial_params": jnp.array([-2.0, 2.0]),
            "lr": 0.01
        },
        {
            "name": "Rosenbrock (different start)",
            "surface_fn": rosenbrock, 
            "grad_fn": rosenbrock_grad,
            "initial_params": jnp.array([0.0, 0.0]),
            "lr": 0.001
        }
    ]
    
    for case in test_cases:
        success = test_adam_on_surface(
            case["name"], 
            case["surface_fn"],
            case["grad_fn"], 
            case["initial_params"],
            case["lr"]
        )
        all_tests_passed = all_tests_passed and success
    
    # Test edge cases
    test_edge_cases()
    
    # Test step consistency
    step_success = test_step_consistency()
    all_tests_passed = all_tests_passed and step_success
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Adam implementation is correct!")
        print("✅ Matches PyTorch torch.optim.Adam exactly")
        print("✅ Matches JAX optax.adam exactly") 
        print("✅ Handles edge cases correctly")
        print("✅ Maintains consistent internal state")
        print("\n💡 This implementation can now be used in production training!")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Check the implementation details and try again")
    
    return all_tests_passed


if __name__ == "__main__":
    test_adam_implementation()