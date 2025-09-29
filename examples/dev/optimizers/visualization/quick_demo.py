#!/usr/bin/env python3
"""
Quick Optimizer Demo - Minimal Version

Just run, save plots, and show key results. No interactive prompts.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import os

# Setup
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '..', 'validation'))

from adam.solution import adam_init, adam_update
from sgd.solution import sgd_init, sgd_update
from muon.solution import muon_init, muon_update

from contract import RosenbrockModel, OptimizerConfig
from compare import compare_optimizers, analyze_convergence
from visualize import plot_2d_trajectories


def main():
    # Configure
    optimizers = {
        'Adam': OptimizerConfig('Adam', adam_init, adam_update, {'lr': 0.01}, 'red'),
        'SGD+Mom': OptimizerConfig('SGD', sgd_init, sgd_update, {'lr': 0.002, 'momentum': 0.9}, 'green'),
        'Muon': OptimizerConfig('Muon', muon_init, muon_update, {'lr': 0.005, 'momentum': 0.9}, 'purple')
    }
    
    # Setup problem
    model = RosenbrockModel(dim=2)
    initial_params = jnp.array([-2.0, 2.0])
    
    print("🚀 Running optimizer comparison on Rosenbrock function...")
    
    # Run comparison
    results = compare_optimizers(
        model=model,
        optimizer_configs=optimizers,
        initial_params=initial_params,
        batches=[None],
        num_steps=500,
        verbose=False
    )
    
    # Results summary
    analysis = analyze_convergence(results)
    print("\n📊 Results:")
    for name, metrics in analysis.items():
        print(f"  {name:10} → Final loss: {metrics['final_loss']:.2e}")
    
    # Create and save plot
    print("\n🎨 Creating visualization...")
    fig = plot_2d_trajectories(results, model, figsize=(10, 8))
    fig.suptitle("Optimizer Comparison - Rosenbrock Function")
    
    # Save plot
    output_file = "optimizer_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"💾 Plot saved: {output_file}")
    print("✅ Demo complete!")


if __name__ == "__main__":
    main()