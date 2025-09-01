#!/usr/bin/env python3
"""
SGD optimizer comparison script.

This script compares SGD implementations (skeleton vs solution) against
PyTorch reference implementation across multiple test surfaces.

Usage:
    python examples/optimizers/validation/sgd/compare.py --mode skeleton
    python examples/optimizers/validation/sgd/compare.py --mode solution
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import argparse
import torch


def load_sgd_implementation(mode):
    """Load SGD implementation based on mode."""
    if mode == "skeleton":
        try:
            from skeleton import sgd_update, sgd_init, SGDState
            print("✅ Successfully imported from skeleton")
            return sgd_update, sgd_init, SGDState
        except ImportError as e:
            print(f"❌ Failed to import from skeleton: {e}")
            sys.exit(1)
    
    elif mode == "solution":
        try:
            from solution import sgd_update, sgd_init, SGDState
            print("✅ Successfully imported from solution")
            return sgd_update, sgd_init, SGDState
        except ImportError as e:
            print(f"❌ Failed to import from solution: {e}")
            sys.exit(1)
    
    else:
        print(f"❌ Invalid mode: {mode}")
        print("💡 Use --mode skeleton or --mode solution")
        sys.exit(1)


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
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False,
    num_steps: int = 100
):
    """Get reference trajectory from PyTorch SGD"""
    
    # PyTorch SGD reference
    torch_params = torch.tensor(np.array(initial_params), requires_grad=True)
    torch_optimizer = torch.optim.SGD([torch_params], lr=lr, momentum=momentum, 
                                    weight_decay=weight_decay, dampening=dampening, 
                                    nesterov=nesterov)
    torch_trajectory = []
    
    for _ in range(num_steps):
        torch_optimizer.zero_grad()
        grads = torch.tensor(np.array(grad_fn(torch_params.detach().numpy())))
        torch_params.grad = grads
        torch_optimizer.step()
        torch_trajectory.append(np.array(torch_params.detach().numpy()))
    
    return torch_trajectory


def test_sgd_on_surface(sgd_update, sgd_init, surface_name: str, surface_fn, grad_fn, 
                       initial_params: jnp.ndarray, lr: float = 0.01, momentum: float = 0.0,
                       weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False):
    """Test SGD implementation on a specific optimization surface"""
    print(f"\n🏔️  Testing on {surface_name}...")
    print(f"   Initial params: {initial_params}")
    print(f"   Learning rate: {lr}")
    print(f"   Momentum: {momentum}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Dampening: {dampening}")
    print(f"   Nesterov: {nesterov}")
    
    num_steps = 100
    
    # Get reference trajectory
    torch_traj = get_reference_trajectories(
        initial_params, lr, surface_fn, grad_fn, momentum, weight_decay, 
        dampening, nesterov, num_steps
    )
    
    # Test our implementation
    our_state = sgd_init(initial_params, lr=lr, momentum=momentum)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = grad_fn(our_params)
        updates, our_state = sgd_update(grads, our_state, our_params, lr=lr, 
                                       momentum=momentum, weight_decay=weight_decay,
                                       dampening=dampening, nesterov=nesterov)
        our_params = our_params + updates
        our_trajectory.append(our_params.copy())
    
    # Compare trajectories
    torch_diff = np.mean([np.linalg.norm(np.array(our_trajectory[i]) - torch_traj[i]) 
                         for i in range(num_steps)])
    
    # Check final convergence
    our_final = our_trajectory[-1]
    torch_final = torch_traj[-1]
    
    print(f"   Final values:")
    print(f"     Our implementation: {our_final}")
    print(f"     PyTorch reference:  {torch_final}")
    print(f"   Mean trajectory difference vs PyTorch: {torch_diff:.6f}")
    
    # Success criteria: very close to PyTorch reference
    torch_success = torch_diff < 1e-5
    
    if torch_success:
        print(f"   ✅ PASS - Close match to PyTorch reference")
        return True
    else:
        print(f"   ❌ FAIL - Trajectory difference too large (threshold: 1e-5)")
        return False


def compare_sgd_implementations(sgd_update, sgd_init, SGDState, mode_name):
    """Run full SGD test suite"""
    print(f"\n🧪 Testing SGD {mode_name} implementation")
    print("="*60)
    
    test_cases = [
        {
            "name": "Vanilla SGD - Quadratic Bowl",
            "surface_fn": quadratic_bowl,
            "grad_fn": quadratic_grad,
            "initial_params": jnp.array([3.0, 4.0]),
            "lr": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "dampening": 0.0,
            "nesterov": False
        },
        {
            "name": "SGD with Momentum",
            "surface_fn": quadratic_bowl,
            "grad_fn": quadratic_grad,
            "initial_params": jnp.array([3.0, 4.0]),
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "dampening": 0.0,
            "nesterov": False
        },
        {
            "name": "SGD with Weight Decay",
            "surface_fn": quadratic_bowl,
            "grad_fn": quadratic_grad,
            "initial_params": jnp.array([3.0, 4.0]),
            "lr": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.01,
            "dampening": 0.0,
            "nesterov": False
        },
        {
            "name": "SGD with Nesterov Momentum",
            "surface_fn": quadratic_bowl,
            "grad_fn": quadratic_grad,
            "initial_params": jnp.array([3.0, 4.0]),
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "dampening": 0.0,
            "nesterov": True
        },
        {
            "name": "Rosenbrock with Momentum",
            "surface_fn": rosenbrock,
            "grad_fn": rosenbrock_grad,
            "initial_params": jnp.array([-2.0, 2.0]),
            "lr": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "dampening": 0.0,
            "nesterov": False
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        success = test_sgd_on_surface(
            sgd_update, sgd_init,
            test_case["name"],
            test_case["surface_fn"],
            test_case["grad_fn"],
            test_case["initial_params"],
            test_case["lr"],
            test_case["momentum"],
            test_case["weight_decay"],
            test_case["dampening"],
            test_case["nesterov"]
        )
        results.append((test_case["name"], success))
    
    return results


def print_summary(results, mode_name):
    """Print test summary"""
    print("\n" + "="*60)
    print("📋 SUMMARY REPORT")
    print("="*60)
    print(f"Mode: {mode_name}")
    print(f"Total tests: {len(results)}")
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    pass_rate = (passed / len(results)) * 100
    
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Pass rate: {pass_rate:.1f}%")
    
    print("\nPer-test results:")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name} - {status}")
    
    if failed > 0:
        print(f"\n💡 {failed} tests failed. Check implementation!")
    else:
        print(f"\n🎉 All tests passed! Implementation is correct.")


def main():
    parser = argparse.ArgumentParser(description="Compare SGD optimizer implementations")
    parser.add_argument("--mode", choices=["skeleton", "solution"], required=True,
                       help="Which implementation to test")
    args = parser.parse_args()
    
    print("🚀 SGD Optimizer Comparison Suite")
    print(f"Mode: {args.mode}")
    
    # Load implementation
    sgd_update, sgd_init, SGDState = load_sgd_implementation(args.mode)
    
    # Run comparison tests
    results = compare_sgd_implementations(sgd_update, sgd_init, SGDState, args.mode)
    
    # Print summary
    print_summary(results, args.mode)


if __name__ == "__main__":
    main()