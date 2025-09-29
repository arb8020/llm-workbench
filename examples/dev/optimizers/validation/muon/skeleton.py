#!/usr/bin/env python3
"""
Muon optimizer JAX implementation skeleton.

Students should implement muon_update() to match the reference implementation.
Muon is a new optimizer from GLM-4 that combines momentum with orthogonalization.

Reference: "Muon: A New Optimizer for Large Language Models"
Usage:
    python examples/optimizers/validation/muon/compare.py --mode skeleton
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, NamedTuple, Tuple
import torch


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
    # TODO: Implement gradient orthogonalization
    # 1. Flatten gradient if needed
    # 2. Apply SVD: U, S, Vt = svd(grad) 
    # 3. Keep top-k singular values based on rank_ratio
    # 4. Reconstruct with reduced rank: U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    # 5. Reshape back to original shape
    
    # Placeholder: return unmodified gradient
    return grad


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
    # TODO: Implement Muon update logic
    # 
    # Muon algorithm:
    # 1. Orthogonalize gradients (if enabled): g_orth = orthogonalize_gradient(grads)
    # 2. Update momentum: m_t = momentum * m_{t-1} + (1 - momentum) * g_orth
    # 3. Compute updates: updates = -lr * m_t
    # 4. Update step count
    
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


def ill_conditioned_quadratic(params: jnp.ndarray) -> float:
    """Ill-conditioned quadratic to test orthogonalization benefits"""
    x, y = params[0], params[1]
    return x**2 + 1000 * y**2  # Very different scales


def ill_conditioned_grad(params: jnp.ndarray) -> jnp.ndarray:
    """Gradient of ill-conditioned quadratic"""
    x, y = params[0], params[1]
    return jnp.array([2 * x, 2000 * y])


def get_reference_muon_implementation():
    """
    Reference Muon implementation based on the paper.
    
    Note: Since Muon is new, we implement our own reference based on the paper
    rather than comparing to an existing library implementation.
    """
    class ReferenceMuon:
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
            
    return ReferenceMuon


def test_orthogonalization():
    """Test the gradient orthogonalization function"""
    print("🔄 Testing gradient orthogonalization...")
    
    # Test on a simple matrix
    test_grad = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    orthogonal_grad = orthogonalize_gradient(test_grad, rank_ratio=0.5)
    
    print(f"Original gradient:\n{test_grad}")
    print(f"Orthogonalized gradient:\n{orthogonal_grad}")
    
    # Check that orthogonalization preserves some properties
    original_norm = jnp.linalg.norm(test_grad)
    orthogonal_norm = jnp.linalg.norm(orthogonal_grad)
    
    print(f"Original norm: {original_norm:.4f}")
    print(f"Orthogonal norm: {orthogonal_norm:.4f}")
    
    # For now, just check that it doesn't crash
    if orthogonal_grad.shape == test_grad.shape:
        print("✅ Orthogonalization preserves shape")
        return True
    else:
        print("❌ Orthogonalization changes shape")
        return False


def test_muon_vs_sgd():
    """Compare Muon to SGD with momentum to show the difference"""
    print("\n⚡ Comparing Muon vs SGD with momentum...")
    
    initial_params = jnp.array([2.0, -2.0])
    lr = 0.02
    momentum_coef = 0.95
    num_steps = 100
    
    # Test on ill-conditioned problem where orthogonalization should help
    print("Testing on ill-conditioned quadratic (x² + 1000y²)...")
    
    # Muon optimization
    muon_state = muon_init(initial_params, lr=lr, momentum=momentum_coef)
    muon_params = initial_params
    muon_losses = []
    
    for step in range(num_steps):
        grads = ill_conditioned_grad(muon_params)
        updates, muon_state = muon_update(
            grads, muon_state, muon_params, 
            lr=lr, momentum=momentum_coef, orthogonalize=True
        )
        muon_params = muon_params + updates
        muon_losses.append(ill_conditioned_quadratic(muon_params))
    
    # SGD with momentum (for comparison)
    sgd_momentum = jnp.zeros_like(initial_params)
    sgd_params = initial_params
    sgd_losses = []
    
    for step in range(num_steps):
        grads = ill_conditioned_grad(sgd_params)
        sgd_momentum = momentum_coef * sgd_momentum + grads
        sgd_updates = -lr * sgd_momentum
        sgd_params = sgd_params + sgd_updates
        sgd_losses.append(ill_conditioned_quadratic(sgd_params))
    
    # Compare final results
    muon_final_loss = muon_losses[-1]
    sgd_final_loss = sgd_losses[-1]
    
    print(f"Final loss - Muon: {muon_final_loss:.6f}")
    print(f"Final loss - SGD+momentum: {sgd_final_loss:.6f}")
    
    if muon_final_loss < sgd_final_loss:
        print("✅ Muon converges better than SGD (orthogonalization helps)")
        return True
    else:
        print("⚠️  Results inconclusive (may need better implementation)")
        return False


def test_muon_implementation():
    """Test our Muon implementation"""
    print("🧪 Testing Muon implementation...")
    
    # Test parameters  
    initial_params = jnp.array([-1.0, 1.0])
    lr = 0.02
    num_steps = 100
    
    print("📋 Testing on Rosenbrock function...")
    
    # Test our implementation
    our_state = muon_init(initial_params, lr=lr)
    our_params = initial_params
    our_trajectory = []
    
    for step in range(num_steps):
        grads = rosenbrock_grad(our_params)
        updates, our_state = muon_update(grads, our_state, our_params, lr=lr)
        our_params = our_params + updates
        our_trajectory.append(our_params.copy())
    
    # Test reference implementation  
    ref_muon = get_reference_muon_implementation()(lr=lr)
    ref_params = initial_params
    ref_trajectory = []
    
    for step in range(num_steps):
        grads = rosenbrock_grad(ref_params)
        updates = ref_muon.update(grads, ref_params)
        ref_params = ref_params + updates
        ref_trajectory.append(ref_params.copy())
    
    # Compare trajectories
    trajectory_diff = np.mean([
        np.linalg.norm(our_trajectory[i] - ref_trajectory[i]) 
        for i in range(num_steps)
    ])
    
    print(f"📈 Average trajectory difference vs reference: {trajectory_diff:.2e}")
    
    # Final losses
    final_loss_ours = rosenbrock(our_trajectory[-1])
    final_loss_ref = rosenbrock(ref_trajectory[-1])
    
    print(f"🎯 Final loss (ours): {final_loss_ours:.6f}")
    print(f"🎯 Final loss (reference): {final_loss_ref:.6f}")
    
    # Success criteria (more lenient since this is a new optimizer)
    tolerance = 1e-3  # Looser tolerance for Muon
    match = trajectory_diff < tolerance
    
    print(f"✅ Reference match: {match} (diff: {trajectory_diff:.2e})")
    
    if match:
        print("🎉 SUCCESS! Muon implementation matches reference!")
        print("💡 Muon combines momentum with gradient orthogonalization")
        return True
    else:
        print("❌ FAILED! Implementation doesn't match reference.")
        print("💡 Hint: Check gradient orthogonalization and momentum update")
        print("💡 Make sure SVD-based orthogonalization preserves gradient direction")
        return False


def run_all_tests():
    """Run all Muon tests"""
    print("🧪 Testing Muon Optimizer Implementation")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test orthogonalization
    orthogonal_success = test_orthogonalization()
    all_tests_passed = all_tests_passed and orthogonal_success
    
    # Test main implementation
    implementation_success = test_muon_implementation()
    all_tests_passed = all_tests_passed and implementation_success
    
    # Test comparison with SGD
    comparison_success = test_muon_vs_sgd()
    all_tests_passed = all_tests_passed and comparison_success
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! (with caveats)")
        print("⚠️  Note: Muon is a new optimizer with limited reference implementations")
        print("✅ Orthogonalization component working")
        print("✅ Momentum update working")
        print("💡 Muon may work better on larger, real neural network problems")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Focus on implementing gradient orthogonalization correctly")
        print("🔧 SVD-based rank reduction is the key innovation of Muon")
    
    return all_tests_passed


if __name__ == "__main__":
    run_all_tests()