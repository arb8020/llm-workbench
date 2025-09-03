#!/usr/bin/env python3
"""
Complete Example: Optimizer Visualization Pipeline

This example demonstrates the full pipeline for comparing optimizers
and visualizing their behavior on different loss surfaces.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '..', 'validation'))

# Import optimizers
from adam.solution import adam_init, adam_update
from adamw.solution import adamw_init, adamw_update
from sgd.solution import sgd_init, sgd_update
from muon.solution import muon_init, muon_update

# Import visualization tools
from contract import RosenbrockModel, SimpleQuadraticModel, OptimizerConfig
from compare import compare_optimizers, print_comparison_table, plot_loss_curves, plot_convergence_summary
from visualize import plot_2d_trajectories, plot_high_dimensional_trajectories, plot_parameter_evolution


def main():
    """Run comprehensive optimizer comparison and visualization"""
    
    print("🚀 Optimizer Visualization Pipeline Example")
    print("=" * 50)
    
    # Configure optimizers
    optimizers = {
        'Adam': OptimizerConfig(
            name='Adam',
            init_fn=adam_init,
            update_fn=adam_update,
            kwargs={'lr': 0.01},
            color='red'
        ),
        'AdamW': OptimizerConfig(
            name='AdamW',
            init_fn=adamw_init,
            update_fn=adamw_update,
            kwargs={'lr': 0.01, 'weight_decay': 0.001},
            color='blue'
        ),
        'SGD+Momentum': OptimizerConfig(
            name='SGD+Momentum',
            init_fn=sgd_init,
            update_fn=sgd_update,
            kwargs={'lr': 0.002, 'momentum': 0.9},
            color='green'
        ),
        'Muon': OptimizerConfig(
            name='Muon',
            init_fn=muon_init,
            update_fn=muon_update,
            kwargs={'lr': 0.005, 'momentum': 0.9, 'orthogonalize': True},
            color='purple'
        )
    }
    
    # =================================================================
    # EXAMPLE 1: 2D Rosenbrock Function (Non-convex, Challenging)
    # =================================================================
    print("\n📊 Example 1: Rosenbrock Function (2D)")
    print("-" * 40)
    
    model_2d = RosenbrockModel(dim=2)
    initial_params_2d = jnp.array([-2.0, 2.0])
    batches = [None]  # No batches for deterministic functions
    
    # Run optimization comparison
    results_2d = compare_optimizers(
        model=model_2d,
        optimizer_configs=optimizers,
        initial_params=initial_params_2d,
        batches=batches,
        num_steps=1000,
        verbose=True
    )
    
    # Print comparison table
    print_comparison_table(results_2d)
    
    # Create visualizations
    print("\n🎨 Creating 2D visualizations...")
    
    # Loss curves
    fig1 = plot_loss_curves(results_2d, title="Rosenbrock Function - Loss Curves")
    
    # 2D trajectory plot
    fig2 = plot_2d_trajectories(
        results_2d, model_2d, 
        x_range=(-2.5, 2.5), y_range=(-1, 3),
        title="Rosenbrock Function - Optimizer Trajectories"
    )
    
    # Convergence summary
    fig3 = plot_convergence_summary(results_2d)
    
    # Parameter evolution
    fig4 = plot_parameter_evolution(results_2d)
    
    # =================================================================
    # EXAMPLE 2: High-Dimensional Quadratic (Ill-conditioned)  
    # =================================================================
    print("\n📊 Example 2: High-Dimensional Ill-Conditioned Quadratic")
    print("-" * 55)
    
    # Create challenging high-dimensional problem
    model_hd = SimpleQuadraticModel(dim=20, condition_number=1000)
    initial_params_hd = jax.random.normal(jax.random.PRNGKey(42), (20,)) * 2.0
    
    # Run optimization (fewer steps for high-dim)
    results_hd = compare_optimizers(
        model=model_hd,
        optimizer_configs=optimizers,
        initial_params=initial_params_hd,
        batches=batches,
        num_steps=500,
        verbose=True
    )
    
    # Print comparison
    print_comparison_table(results_hd)
    
    # High-dimensional visualizations
    print("\n🎨 Creating high-dimensional visualizations...")
    
    # PCA projection
    fig5 = plot_high_dimensional_trajectories(
        results_hd, method='pca', n_components=2
    )
    fig5.suptitle("High-Dimensional Trajectories (PCA Projection)")
    
    # 3D PCA projection
    fig6 = plot_high_dimensional_trajectories(
        results_hd, method='pca', n_components=3
    )
    
    # Parameter evolution (first 6 parameters)
    fig7 = plot_parameter_evolution(results_hd, max_params=6)
    fig7.suptitle("Parameter Evolution (High-Dimensional Problem)")
    
    # Loss curves comparison
    fig8 = plot_loss_curves(results_hd, title="High-Dimensional Problem - Loss Curves")
    
    # =================================================================
    # EXAMPLE 3: Different Conditioning Examples
    # =================================================================
    print("\n📊 Example 3: Effect of Problem Conditioning")
    print("-" * 45)
    
    # Test on different condition numbers
    condition_numbers = [1, 10, 100, 1000]
    conditioning_results = {}
    
    for cond_num in condition_numbers:
        print(f"Testing condition number: {cond_num}")
        model_cond = SimpleQuadraticModel(dim=5, condition_number=cond_num)
        initial_params_cond = jax.random.normal(jax.random.PRNGKey(123), (5,)) * 1.0
        
        # Test just Adam vs Muon for conditioning comparison
        optimizers_subset = {
            'Adam': optimizers['Adam'],
            'Muon': optimizers['Muon']
        }
        
        results_cond = compare_optimizers(
            model=model_cond,
            optimizer_configs=optimizers_subset,
            initial_params=initial_params_cond,
            batches=batches,
            num_steps=200,
            verbose=False
        )
        
        conditioning_results[cond_num] = results_cond
    
    # Plot conditioning analysis
    fig9, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, cond_num in enumerate(condition_numbers):
        results = conditioning_results[cond_num]
        
        for name, trajectory in results.items():
            losses = np.array(trajectory.losses)
            steps = np.arange(len(losses))
            axes[i].semilogy(steps, losses, label=name, linewidth=2)
        
        axes[i].set_title(f'Condition Number: {cond_num}')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Loss')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    fig9.suptitle('Effect of Problem Conditioning on Optimizer Performance')
    plt.tight_layout()
    
    # =================================================================
    # Show all plots
    # =================================================================
    print("\n🎉 Visualization complete!")
    print("Close the plot windows to exit.")
    
    # Display all figures
    plt.show()
    
    print("\n💡 Key Insights:")
    print("- Adam/AdamW adapt well to different scales")
    print("- SGD+Momentum needs careful tuning but can be very stable") 
    print("- Muon's orthogonalization helps with ill-conditioned problems")
    print("- High condition numbers challenge all optimizers")
    print("- 2D visualizations reveal trajectory differences clearly")
    print("- PCA projections help understand high-dimensional behavior")


if __name__ == "__main__":
    main()