#!/usr/bin/env python3
"""
Multi-Optimizer Comparison Utilities

Tools for comparing multiple optimizers on the same optimization problem
and analyzing their relative performance characteristics.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

from .contract import (
    OptimizationModel, TrajectoryData, OptimizerConfig, 
    optimize_and_track
)


def compare_optimizers(
    model: OptimizationModel,
    optimizer_configs: Dict[str, OptimizerConfig],
    initial_params: jnp.ndarray,
    batches: List[Any],
    num_steps: int = 1000,
    verbose: bool = True
) -> Dict[str, TrajectoryData]:
    """
    Compare multiple optimizers on the same optimization problem.
    
    Args:
        model: Model to optimize
        optimizer_configs: Dictionary of optimizer configurations
        initial_params: Initial parameter values (same for all optimizers)
        batches: List of training batches
        num_steps: Number of optimization steps
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping optimizer names to their trajectory data
    """
    results = {}
    
    for name, config in optimizer_configs.items():
        if verbose:
            print(f"🚀 Running {name}...")
            
        trajectory = optimize_and_track(
            model=model,
            optimizer_init=config.init_fn,
            optimizer_update=config.update_fn,
            initial_params=initial_params,
            batches=batches,
            num_steps=num_steps,
            **config.kwargs
        )
        
        results[name] = trajectory
        
        if verbose:
            final_loss = trajectory.losses[-1]
            convergence = trajectory.metadata['convergence_rate']
            print(f"   Final loss: {final_loss:.6f} (convergence: {convergence:.2f}x)")
    
    return results


def analyze_convergence(results: Dict[str, TrajectoryData]) -> Dict[str, Dict[str, float]]:
    """
    Analyze convergence characteristics of different optimizers.
    
    Returns:
        Dictionary of convergence metrics for each optimizer
    """
    analysis = {}
    
    for name, trajectory in results.items():
        losses = np.array(trajectory.losses)
        step_sizes = np.array(trajectory.step_sizes)
        
        # Find when optimizer reaches 90% of final improvement
        initial_loss = losses[0]
        final_loss = losses[-1]
        target_loss = initial_loss - 0.9 * (initial_loss - final_loss)
        
        convergence_step = len(losses) - 1  # Default to final step
        for i, loss in enumerate(losses):
            if loss <= target_loss:
                convergence_step = i
                break
        
        analysis[name] = {
            'final_loss': float(final_loss),
            'convergence_rate': float(initial_loss / final_loss),
            'convergence_step': convergence_step,
            'total_distance': trajectory.metadata['total_distance'],
            'avg_step_size': float(np.mean(step_sizes)),
            'step_size_variance': float(np.var(step_sizes)),
            'loss_reduction': float(initial_loss - final_loss)
        }
    
    return analysis


def plot_loss_curves(results: Dict[str, TrajectoryData], 
                    title: str = "Optimizer Comparison - Loss Curves",
                    log_scale: bool = True,
                    figsize: tuple = (10, 6)):
    """Plot loss curves for all optimizers"""
    
    plt.figure(figsize=figsize)
    
    for name, trajectory in results.items():
        losses = np.array(trajectory.losses)
        steps = np.arange(len(losses))
        
        plt.plot(steps, losses, label=name, linewidth=2)
    
    plt.xlabel('Optimization Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
        
    plt.tight_layout()
    return plt.gcf()


def plot_step_sizes(results: Dict[str, TrajectoryData],
                   title: str = "Optimizer Comparison - Step Sizes",
                   figsize: tuple = (10, 6)):
    """Plot effective step sizes over time"""
    
    plt.figure(figsize=figsize)
    
    for name, trajectory in results.items():
        step_sizes = np.array(trajectory.step_sizes)
        steps = np.arange(len(step_sizes))
        
        plt.plot(steps, step_sizes, label=name, linewidth=2, alpha=0.7)
    
    plt.xlabel('Optimization Step')
    plt.ylabel('Effective Step Size')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    return plt.gcf()


def plot_convergence_summary(results: Dict[str, TrajectoryData],
                           figsize: tuple = (15, 5)):
    """Create a summary dashboard of convergence metrics"""
    
    analysis = analyze_convergence(results)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Final loss comparison
    names = list(analysis.keys())
    final_losses = [analysis[name]['final_loss'] for name in names]
    
    axes[0].bar(names, final_losses)
    axes[0].set_title('Final Loss')
    axes[0].set_ylabel('Loss Value')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Convergence speed (steps to 90% improvement)
    convergence_steps = [analysis[name]['convergence_step'] for name in names]
    
    axes[1].bar(names, convergence_steps)
    axes[1].set_title('Convergence Speed')
    axes[1].set_ylabel('Steps to 90% Improvement')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Total distance traveled
    distances = [analysis[name]['total_distance'] for name in names]
    
    axes[2].bar(names, distances)
    axes[2].set_title('Total Distance Traveled')
    axes[2].set_ylabel('Parameter Space Distance')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def print_comparison_table(results: Dict[str, TrajectoryData]):
    """Print a formatted comparison table"""
    
    analysis = analyze_convergence(results)
    
    print("\n" + "="*80)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Optimizer':<12} {'Final Loss':<12} {'Conv. Rate':<12} {'Conv. Steps':<12} {'Avg Step':<12}")
    print("-" * 80)
    
    # Sort by final loss
    sorted_results = sorted(analysis.items(), key=lambda x: x[1]['final_loss'])
    
    for name, metrics in sorted_results:
        print(f"{name:<12} {metrics['final_loss']:<12.6f} "
              f"{metrics['convergence_rate']:<12.2f} {metrics['convergence_step']:<12} "
              f"{metrics['avg_step_size']:<12.2e}")
    
    print("="*80)
    
    # Find best performer in each category
    best_loss = min(analysis.items(), key=lambda x: x[1]['final_loss'])
    best_speed = min(analysis.items(), key=lambda x: x[1]['convergence_step'])
    best_stability = min(analysis.items(), key=lambda x: x[1]['step_size_variance'])
    
    print(f"🏆 Best Final Loss: {best_loss[0]} ({best_loss[1]['final_loss']:.6f})")
    print(f"⚡ Fastest Convergence: {best_speed[0]} ({best_speed[1]['convergence_step']} steps)")
    print(f"🎯 Most Stable: {best_stability[0]} (variance: {best_stability[1]['step_size_variance']:.2e})")
    print()


# Example usage function
def run_comparison_example():
    """Example of how to use the comparison tools"""
    
    # Import optimizers
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'validation'))
    
    from adam.solution import adam_init, adam_update
    from adamw.solution import adamw_init, adamw_update
    from sgd.solution import sgd_init, sgd_update
    from muon.solution import muon_init, muon_update
    
    from .contract import RosenbrockModel
    
    # Setup
    model = RosenbrockModel(dim=2)
    initial_params = jnp.array([-2.0, 2.0])
    batches = [None]  # No batches needed for Rosenbrock
    
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
        'SGD': OptimizerConfig(
            name='SGD',
            init_fn=sgd_init,
            update_fn=sgd_update,
            kwargs={'lr': 0.001, 'momentum': 0.9},
            color='green'
        ),
        'Muon': OptimizerConfig(
            name='Muon',
            init_fn=muon_init,
            update_fn=muon_update,
            kwargs={'lr': 0.005, 'momentum': 0.9},
            color='purple'
        )
    }
    
    # Run comparison
    results = compare_optimizers(model, optimizers, initial_params, batches, num_steps=1000)
    
    # Analyze and visualize
    print_comparison_table(results)
    plot_loss_curves(results)
    plot_convergence_summary(results)
    plt.show()


if __name__ == "__main__":
    run_comparison_example()