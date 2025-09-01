#!/usr/bin/env python3
"""
AdamW optimizer JAX implementation solution.

Complete AdamW implementation that matches PyTorch and Optax exactly.
Key insight: AdamW applies weight decay directly to parameters, not through gradients.

Usage:
    python examples/optimizers/validation/adamw/solution.py
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


def simple_quadratic(params: jnp.ndarray) -> float:
    """Simple quadratic without L2 term (to isolate weight decay effect)"""
    x, y = params[0], params[1]
    return x**2 + 10 * y**2


def simple_quadratic_grad(params: jnp.ndarray) -> jnp.ndarray:
    """Gradient of simple quadratic"""
    x, y = params[0], params[1]
    return jnp.array([2 * x, 20 * y])


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


def test_adamw_on_surface(surface_name: str, surface_fn, grad_fn, initial_params: jnp.ndarray, lr: float = 0.01, weight_decay: float = 0.01):
    """Test AdamW implementation on a specific optimization surface"""
    print(f"\n🏔️  Testing on {surface_name}...")
    print(f"   Initial params: {initial_params}")
    print(f"   Learning rate: {lr}, Weight decay: {weight_decay}")
    
    num_steps = 100
    
    # Get reference trajectories
    torch_traj, optax_traj = get_reference_trajectories(
        initial_params, lr, weight_decay, surface_fn, grad_fn, num_steps
    )
    
    # Test our implementation
    our_state = adamw_init(initial_params, lr=lr, weight_decay=weight_decay)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = grad_fn(our_params)
        updates, our_state = adamw_update(grads, our_state, our_params, lr=lr, weight_decay=weight_decay)
        our_params = our_params + updates
        our_trajectory.append(our_params.copy())
    
    # Compare trajectories
    torch_diff = np.mean([np.linalg.norm(np.array(our_trajectory[i]) - torch_traj[i]) 
                         for i in range(num_steps)])
    
    optax_diff = np.mean([np.linalg.norm(our_trajectory[i] - optax_traj[i]) 
                         for i in range(num_steps)])
    
    # Final values and weight decay effect
    final_loss_ours = surface_fn(our_trajectory[-1])
    final_loss_torch = surface_fn(torch_traj[-1])
    final_loss_optax = surface_fn(optax_traj[-1])
    final_norm = jnp.linalg.norm(our_trajectory[-1])
    
    print(f"   📈 Trajectory diff vs PyTorch: {torch_diff:.2e}")
    print(f"   📈 Trajectory diff vs Optax: {optax_diff:.2e}")
    print(f"   🎯 Final loss (ours): {final_loss_ours:.6f}")
    print(f"   🎯 Final loss (torch): {final_loss_torch:.6f}")
    print(f"   🎯 Final loss (optax): {final_loss_optax:.6f}")
    print(f"   📏 Final param norm: {final_norm:.6f} (smaller = more weight decay effect)")
    
    # Success criteria
    tolerance = 1e-6
    torch_match = torch_diff < tolerance
    optax_match = optax_diff < tolerance
    
    success = torch_match and optax_match
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"   {status}")
    
    return success


def test_weight_decay_effect():
    """Test that weight decay actually shrinks parameters as expected"""
    print("\n🎯 Testing weight decay effect...")
    
    # Compare AdamW with different weight decay values
    initial_params = jnp.array([2.0, -2.0])  # Start away from origin
    lr = 0.01
    num_steps = 50
    
    weight_decays = [0.0, 0.01, 0.1]
    final_norms = []
    
    for wd in weight_decays:
        state = adamw_init(initial_params, lr=lr, weight_decay=wd)
        params = initial_params
        
        for step in range(num_steps):
            grads = simple_quadratic_grad(params)
            updates, state = adamw_update(grads, state, params, lr=lr, weight_decay=wd)
            params = params + updates
        
        final_norm = jnp.linalg.norm(params)
        final_norms.append(final_norm)
        print(f"   Weight decay {wd}: final norm = {final_norm:.4f}")
    
    # Verify that higher weight decay leads to smaller final norms
    if final_norms[0] > final_norms[1] > final_norms[2]:
        print("   ✅ PASS: Higher weight decay → smaller parameter norms")
        return True
    else:
        print("   ❌ FAIL: Weight decay not working correctly")
        return False


def test_adamw_vs_adam_comparison():
    """Compare AdamW vs Adam to highlight the difference"""
    print("\n⚡ Comparing AdamW vs Adam...")
    
    # Both optimizers start from same point, same hyperparams
    initial_params = jnp.array([1.0, -1.0])
    lr = 0.01
    num_steps = 100
    
    # AdamW with weight decay
    adamw_state = adamw_init(initial_params, lr=lr, weight_decay=0.01)
    adamw_params = initial_params
    
    # Adam (no weight decay) - we'll implement a simple version
    adam_m = jnp.zeros_like(initial_params)
    adam_v = jnp.zeros_like(initial_params)
    adam_step = 0
    adam_params = initial_params
    
    for step in range(num_steps):
        grads = simple_quadratic_grad(adamw_params)  # Same gradients for both
        
        # AdamW update
        adamw_updates, adamw_state = adamw_update(
            grads, adamw_state, adamw_params, lr=lr, weight_decay=0.01
        )
        adamw_params = adamw_params + adamw_updates
        
        # Simple Adam update (for comparison)
        adam_step += 1
        adam_m = 0.9 * adam_m + 0.1 * grads
        adam_v = 0.999 * adam_v + 0.001 * grads**2
        m_hat = adam_m / (1 - 0.9**adam_step)
        v_hat = adam_v / (1 - 0.999**adam_step)
        adam_updates = -lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        adam_params = adam_params + adam_updates
    
    adamw_norm = jnp.linalg.norm(adamw_params)
    adam_norm = jnp.linalg.norm(adam_params)
    
    print(f"   Final AdamW param norm: {adamw_norm:.4f}")
    print(f"   Final Adam param norm:  {adam_norm:.4f}")
    print(f"   Difference: {adam_norm - adamw_norm:.4f}")
    
    if adamw_norm < adam_norm:
        print("   ✅ PASS: AdamW produces smaller parameters (weight decay working)")
        return True
    else:
        print("   ❌ FAIL: AdamW should produce smaller parameters than Adam")
        return False


def test_adamw_implementation():
    """Complete test suite for AdamW implementation"""
    print("🧪 Testing AdamW Implementation")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test on different optimization surfaces
    test_cases = [
        {
            "name": "Simple Quadratic (isolates weight decay)",
            "surface_fn": simple_quadratic,
            "grad_fn": simple_quadratic_grad,
            "initial_params": jnp.array([2.0, -2.0]),
            "lr": 0.1,
            "weight_decay": 0.01
        },
        {
            "name": "Quadratic with L2 regularization",
            "surface_fn": lambda p: quadratic_with_l2(p, 0.1),
            "grad_fn": lambda p: quadratic_l2_grad(p, 0.1),
            "initial_params": jnp.array([-1.0, 1.5]),
            "lr": 0.05,
            "weight_decay": 0.01
        },
        {
            "name": "Rosenbrock Function",
            "surface_fn": rosenbrock,
            "grad_fn": rosenbrock_grad,
            "initial_params": jnp.array([-1.0, 1.0]),
            "lr": 0.001,
            "weight_decay": 0.001  # Small weight decay for Rosenbrock
        }
    ]
    
    for case in test_cases:
        success = test_adamw_on_surface(
            case["name"], 
            case["surface_fn"],
            case["grad_fn"], 
            case["initial_params"],
            case["lr"],
            case["weight_decay"]
        )
        all_tests_passed = all_tests_passed and success
    
    # Test weight decay effect
    weight_decay_success = test_weight_decay_effect()
    all_tests_passed = all_tests_passed and weight_decay_success
    
    # Test AdamW vs Adam comparison
    comparison_success = test_adamw_vs_adam_comparison()
    all_tests_passed = all_tests_passed and comparison_success
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! AdamW implementation is correct!")
        print("✅ Matches PyTorch torch.optim.AdamW exactly")
        print("✅ Matches JAX optax.adamw exactly") 
        print("✅ Weight decay applied directly to parameters (not gradients)")
        print("✅ Produces smaller parameter norms than Adam")
        print("\n💡 Key insight: AdamW decouples weight decay from gradient-based updates")
        print("💡 This leads to better generalization in many deep learning tasks")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Check that weight decay is applied directly to parameters")
        print("🔧 Formula: updates = -lr * weight_decay * params - lr * m_hat / (sqrt(v_hat) + eps)")
    
    return all_tests_passed


if __name__ == "__main__":
    test_adamw_implementation()