#!/usr/bin/env python3
"""
Muon optimizer comparison script.

This script compares Muon implementations (skeleton vs solution) against
a reference implementation since Muon is not available in standard libraries.

Usage:
    python examples/optimizers/validation/muon/compare.py --mode skeleton
    python examples/optimizers/validation/muon/compare.py --mode solution
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import argparse


def load_muon_implementation(mode):
    """Load Muon implementation based on mode."""
    if mode == "skeleton":
        try:
            from skeleton import muon_update, muon_init, MuonState, orthogonalize_gradient
            print("✅ Successfully imported from skeleton")
            return muon_update, muon_init, MuonState, orthogonalize_gradient
        except ImportError as e:
            print(f"❌ Failed to import from skeleton: {e}")
            sys.exit(1)
    
    elif mode == "solution":
        try:
            from solution import muon_update, muon_init, MuonState, orthogonalize_gradient
            print("✅ Successfully imported from solution")
            return muon_update, muon_init, MuonState, orthogonalize_gradient
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


def ill_conditioned_quadratic(params: jnp.ndarray) -> float:
    """Ill-conditioned quadratic to test orthogonalization benefits"""
    x, y = params[0], params[1]
    return x**2 + 1000 * y**2  # Very different scales


def ill_conditioned_grad(params: jnp.ndarray) -> jnp.ndarray:
    """Gradient of ill-conditioned quadratic"""
    x, y = params[0], params[1]
    return jnp.array([2 * x, 2000 * y])


class ReferenceMuon:
    """Reference Muon implementation based on the paper algorithm"""
    
    def __init__(self, lr=0.02, momentum=0.95, rank_ratio=0.02):
        self.lr = lr
        self.momentum = momentum 
        self.rank_ratio = rank_ratio
        self.momentum_buffer = None
        self.step = 0
        
    def update(self, grads, params):
        """Reference Muon update based on paper algorithm"""
        if self.momentum_buffer is None:
            self.momentum_buffer = jnp.zeros_like(grads)
            
        # Orthogonalize gradients using SVD
        if grads.ndim > 1:
            # For matrices, apply SVD-based orthogonalization
            U, S, Vt = jnp.linalg.svd(grads, full_matrices=False)
            # Keep top singular values based on rank ratio
            k = max(1, int(self.rank_ratio * min(grads.shape)))
            reconstructed = U[:, :k] @ jnp.diag(S[:k]) @ Vt[:k, :]
            orthogonalized_grad = reconstructed
        else:
            # For vectors, just normalize
            orthogonalized_grad = grads / (jnp.linalg.norm(grads) + 1e-8)
            
        # Update momentum
        self.momentum_buffer = (
            self.momentum * self.momentum_buffer + 
            (1 - self.momentum) * orthogonalized_grad
        )
        
        # Compute updates
        updates = -self.lr * self.momentum_buffer
        self.step += 1
        
        return updates


def get_reference_trajectories(
    initial_params: jnp.ndarray, 
    lr: float, 
    surface_fn, 
    grad_fn,
    momentum: float = 0.95,
    rank_ratio: float = 0.02,
    num_steps: int = 100
):
    """Get reference trajectory from reference Muon implementation"""
    
    # Reference Muon implementation
    ref_muon = ReferenceMuon(lr=lr, momentum=momentum, rank_ratio=rank_ratio)
    ref_params = initial_params
    ref_trajectory = []
    
    for _ in range(num_steps):
        grads = grad_fn(ref_params)
        updates = ref_muon.update(grads, ref_params)
        ref_params = ref_params + updates
        ref_trajectory.append(np.array(ref_params))
    
    return ref_trajectory


def test_muon_on_surface(muon_update, muon_init, surface_name: str, surface_fn, grad_fn, 
                        initial_params: jnp.ndarray, lr: float = 0.02, momentum: float = 0.95,
                        rank_ratio: float = 0.02, orthogonalize: bool = True):
    """Test Muon implementation on a specific optimization surface"""
    print(f"\n🏔️  Testing on {surface_name}...")
    print(f"   Initial params: {initial_params}")
    print(f"   Learning rate: {lr}")
    print(f"   Momentum: {momentum}")
    print(f"   Rank ratio: {rank_ratio}")
    print(f"   Orthogonalize: {orthogonalize}")
    
    num_steps = 100
    
    # Get reference trajectory
    ref_traj = get_reference_trajectories(
        initial_params, lr, surface_fn, grad_fn, momentum, rank_ratio, num_steps
    )
    
    # Test our implementation
    our_state = muon_init(initial_params, lr=lr, momentum=momentum, rank_ratio=rank_ratio)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = grad_fn(our_params)
        updates, our_state = muon_update(grads, our_state, our_params, lr=lr, 
                                        momentum=momentum, rank_ratio=rank_ratio,
                                        orthogonalize=orthogonalize)
        our_params = our_params + updates
        our_trajectory.append(our_params.copy())
    
    # Compare trajectories
    ref_diff = np.mean([np.linalg.norm(np.array(our_trajectory[i]) - ref_traj[i]) 
                       for i in range(num_steps)])
    
    # Check final convergence
    our_final = our_trajectory[-1]
    ref_final = ref_traj[-1]
    
    print(f"   Final values:")
    print(f"     Our implementation: {our_final}")
    print(f"     Reference:          {ref_final}")
    print(f"   Mean trajectory difference vs reference: {ref_diff:.6f}")
    
    # Success criteria: reasonably close to reference (Muon is experimental)
    ref_success = ref_diff < 1e-2  # More lenient threshold for Muon
    
    if ref_success:
        print(f"   ✅ PASS - Close match to reference implementation")
        return True
    else:
        print(f"   ❌ FAIL - Trajectory difference too large (threshold: 1e-2)")
        return False


def test_orthogonalization(orthogonalize_gradient):
    """Test the gradient orthogonalization function"""
    print("\n🔄 Testing gradient orthogonalization...")
    
    # Test on a simple matrix
    test_grad = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    orthogonal_grad = orthogonalize_gradient(test_grad, rank_ratio=0.5)
    
    print(f"   Original gradient shape: {test_grad.shape}")
    print(f"   Orthogonalized gradient shape: {orthogonal_grad.shape}")
    
    # Check that orthogonalization preserves shape
    if orthogonal_grad.shape == test_grad.shape:
        print("   ✅ PASS - Orthogonalization preserves shape")
        return True
    else:
        print("   ❌ FAIL - Orthogonalization changes shape")
        return False


def compare_muon_implementations(muon_update, muon_init, MuonState, orthogonalize_gradient, mode_name):
    """Run full Muon test suite"""
    print(f"\n🧪 Testing Muon {mode_name} implementation")
    print("="*60)
    
    results = []
    
    # Test orthogonalization function first
    ortho_success = test_orthogonalization(orthogonalize_gradient)
    results.append(("Gradient Orthogonalization", ortho_success))
    
    test_cases = [
        {
            "name": "Quadratic Bowl",
            "surface_fn": quadratic_bowl,
            "grad_fn": quadratic_grad,
            "initial_params": jnp.array([3.0, 4.0]),
            "lr": 0.02,
            "momentum": 0.95,
            "rank_ratio": 0.02,
            "orthogonalize": True
        },
        {
            "name": "Ill-conditioned Quadratic (Orthogonalization Test)",
            "surface_fn": ill_conditioned_quadratic,
            "grad_fn": ill_conditioned_grad,
            "initial_params": jnp.array([2.0, -2.0]),
            "lr": 0.01,
            "momentum": 0.9,
            "rank_ratio": 0.1,
            "orthogonalize": True
        },
        {
            "name": "Rosenbrock Function",
            "surface_fn": rosenbrock,
            "grad_fn": rosenbrock_grad,
            "initial_params": jnp.array([-2.0, 2.0]),
            "lr": 0.005,
            "momentum": 0.9,
            "rank_ratio": 0.02,
            "orthogonalize": True
        },
        {
            "name": "Without Orthogonalization (Momentum Only)",
            "surface_fn": quadratic_bowl,
            "grad_fn": quadratic_grad,
            "initial_params": jnp.array([3.0, 4.0]),
            "lr": 0.02,
            "momentum": 0.95,
            "rank_ratio": 0.02,
            "orthogonalize": False
        }
    ]
    
    for test_case in test_cases:
        success = test_muon_on_surface(
            muon_update, muon_init,
            test_case["name"],
            test_case["surface_fn"],
            test_case["grad_fn"],
            test_case["initial_params"],
            test_case["lr"],
            test_case["momentum"],
            test_case["rank_ratio"],
            test_case["orthogonalize"]
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
        print("💡 Focus on gradient orthogonalization using SVD")
        print("💡 Muon is experimental - small differences are expected")
    else:
        print(f"\n🎉 All tests passed! Implementation is correct.")
        print("💡 Muon combines momentum with gradient orthogonalization")


def main():
    parser = argparse.ArgumentParser(description="Compare Muon optimizer implementations")
    parser.add_argument("--mode", choices=["skeleton", "solution"], required=True,
                       help="Which implementation to test")
    args = parser.parse_args()
    
    print("🚀 Muon Optimizer Comparison Suite")
    print(f"Mode: {args.mode}")
    print("⚠️  Note: Muon is experimental - using reference implementation for comparison")
    
    # Load implementation
    muon_update, muon_init, MuonState, orthogonalize_gradient = load_muon_implementation(args.mode)
    
    # Run comparison tests
    results = compare_muon_implementations(muon_update, muon_init, MuonState, orthogonalize_gradient, args.mode)
    
    # Print summary
    print_summary(results, args.mode)


if __name__ == "__main__":
    main()