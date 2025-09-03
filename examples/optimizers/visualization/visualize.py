#!/usr/bin/env python3
"""
Loss Surface Visualization Tools

Comprehensive tools for visualizing optimizer trajectories on loss surfaces,
including strategies for high-dimensional parameter spaces.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .contract import OptimizationModel, TrajectoryData


def plot_2d_loss_surface(
    model: OptimizationModel,
    x_range: Tuple[float, float] = (-3, 3),
    y_range: Tuple[float, float] = (-3, 3),
    resolution: int = 100,
    levels: int = 20,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot 2D loss surface contours for 2-parameter models.
    
    Args:
        model: Model with 2-dimensional parameter space
        x_range: Range for first parameter
        y_range: Range for second parameter
        resolution: Grid resolution for contour plot
        levels: Number of contour levels
        figsize: Figure size
    """
    assert model.param_shape == (2,), "Model must have 2-dimensional parameters"
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute loss surface
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            params = jnp.array([X[i, j], Y[i, j]])
            Z[i, j] = model.loss(params)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Contour plot
    contour = ax.contour(X, Y, Z, levels=levels, alpha=0.6, colors='gray', linewidths=0.5)
    contourf = ax.contourf(X, Y, Z, levels=levels, alpha=0.3, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(contourf, ax=ax, label='Loss Value')
    
    # Mark global minimum if available
    if hasattr(model, 'optimal_params'):
        opt_params = model.optimal_params()
        if len(opt_params) == 2:
            ax.plot(opt_params[0], opt_params[1], 'r*', markersize=15, 
                   label='Global Minimum', markeredgecolor='black')
    
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_title('Loss Surface')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_2d_trajectories(
    results: Dict[str, TrajectoryData],
    model: OptimizationModel,
    x_range: Tuple[float, float] = (-3, 3),
    y_range: Tuple[float, float] = (-3, 3),
    show_surface: bool = True,
    show_arrows: bool = True,
    arrow_frequency: int = 10,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot optimizer trajectories on 2D loss surface.
    
    Args:
        results: Dictionary of trajectory data from compare_optimizers
        model: Model with 2-dimensional parameter space
        show_surface: Whether to show loss surface contours
        show_arrows: Whether to show trajectory direction arrows
        arrow_frequency: Show every Nth arrow
        figsize: Figure size
    """
    assert model.param_shape == (2,), "Model must have 2-dimensional parameters"
    
    # Create base plot with loss surface
    if show_surface:
        fig, ax = plot_2d_loss_surface(model, x_range, y_range, figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trajectories
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (name, trajectory), color in zip(results.items(), colors):
        params_array = np.array(trajectory.params)
        x_traj = params_array[:, 0]
        y_traj = params_array[:, 1]
        
        # Plot trajectory line
        ax.plot(x_traj, y_traj, color=color, linewidth=2, alpha=0.8, label=name)
        
        # Mark start and end points
        ax.plot(x_traj[0], y_traj[0], 'o', color=color, markersize=8, 
               markeredgecolor='black', markeredgewidth=1)
        ax.plot(x_traj[-1], y_traj[-1], 's', color=color, markersize=8,
               markeredgecolor='black', markeredgewidth=1)
        
        # Add direction arrows
        if show_arrows and len(x_traj) > arrow_frequency:
            for i in range(0, len(x_traj) - arrow_frequency, arrow_frequency):
                dx = x_traj[i + arrow_frequency] - x_traj[i]
                dy = y_traj[i + arrow_frequency] - y_traj[i]
                ax.arrow(x_traj[i], y_traj[i], dx, dy, 
                        head_width=0.05, head_length=0.05, 
                        fc=color, ec=color, alpha=0.6)
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Optimizer Trajectories on Loss Surface')
    
    plt.tight_layout()
    return fig


def animate_2d_trajectories(
    results: Dict[str, TrajectoryData],
    model: OptimizationModel,
    x_range: Tuple[float, float] = (-3, 3),
    y_range: Tuple[float, float] = (-3, 3),
    interval: int = 50,
    save_path: Optional[str] = None
):
    """
    Create animated visualization of optimizer trajectories.
    
    Args:
        results: Dictionary of trajectory data
        model: Model with 2-dimensional parameter space
        interval: Animation interval in milliseconds
        save_path: If provided, save animation as GIF
    """
    # Set up the figure and axis
    fig, ax = plot_2d_loss_surface(model, x_range, y_range)
    
    # Prepare trajectory data
    max_steps = max(len(traj.params) for traj in results.values())
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    # Initialize empty line objects
    lines = {}
    points = {}
    
    for (name, _), color in zip(results.items(), colors):
        line, = ax.plot([], [], color=color, linewidth=2, alpha=0.8, label=name)
        point, = ax.plot([], [], 'o', color=color, markersize=8, 
                        markeredgecolor='black', markeredgewidth=1)
        lines[name] = line
        points[name] = point
    
    ax.legend()
    
    def animate(frame):
        for name, trajectory in results.items():
            if frame < len(trajectory.params):
                params_array = np.array(trajectory.params[:frame+1])
                
                # Update trajectory line
                lines[name].set_data(params_array[:, 0], params_array[:, 1])
                
                # Update current position marker
                if frame < len(trajectory.params):
                    current_params = trajectory.params[frame]
                    points[name].set_data([current_params[0]], [current_params[1]])
        
        return list(lines.values()) + list(points.values())
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=max_steps, 
                        interval=interval, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow')
        print(f"Animation saved to {save_path}")
    
    return anim


def plot_high_dimensional_trajectories(
    results: Dict[str, TrajectoryData],
    method: str = 'pca',
    n_components: int = 2,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize high-dimensional trajectories using dimensionality reduction.
    
    Args:
        results: Dictionary of trajectory data
        method: Reduction method ('pca', 'tsne')
        n_components: Number of components to project to
        figsize: Figure size
    """
    # Collect all parameter vectors from all trajectories
    all_params = []
    trajectory_indices = []
    optimizer_names = []
    
    for name, trajectory in results.items():
        params_array = np.array(trajectory.params)
        all_params.extend(params_array)
        trajectory_indices.extend(range(len(params_array)))
        optimizer_names.extend([name] * len(params_array))
    
    all_params = np.array(all_params)
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced_params = reducer.fit_transform(all_params)
        variance_explained = reducer.explained_variance_ratio_
        title_suffix = f"(PCA: {variance_explained[:2].sum():.1%} variance explained)"
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        reduced_params = reducer.fit_transform(all_params)
        title_suffix = "(t-SNE projection)"
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Create visualization
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trajectories
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        color_map = {name: color for (name, _), color in zip(results.items(), colors)}
        
        start_idx = 0
        for name, trajectory in results.items():
            traj_length = len(trajectory.params)
            traj_reduced = reduced_params[start_idx:start_idx + traj_length]
            
            # Plot trajectory
            ax.plot(traj_reduced[:, 0], traj_reduced[:, 1], 
                   color=color_map[name], linewidth=2, alpha=0.8, label=name)
            
            # Mark start and end
            ax.plot(traj_reduced[0, 0], traj_reduced[0, 1], 'o', 
                   color=color_map[name], markersize=8, markeredgecolor='black')
            ax.plot(traj_reduced[-1, 0], traj_reduced[-1, 1], 's',
                   color=color_map[name], markersize=8, markeredgecolor='black')
            
            start_idx += traj_length
        
        ax.set_xlabel(f'Component 1')
        ax.set_ylabel(f'Component 2')
        ax.set_title(f'High-Dimensional Trajectories {title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        color_map = {name: color for (name, _), color in zip(results.items(), colors)}
        
        start_idx = 0
        for name, trajectory in results.items():
            traj_length = len(trajectory.params)
            traj_reduced = reduced_params[start_idx:start_idx + traj_length]
            
            # Plot trajectory
            ax.plot(traj_reduced[:, 0], traj_reduced[:, 1], traj_reduced[:, 2],
                   color=color_map[name], linewidth=2, alpha=0.8, label=name)
            
            # Mark start and end
            ax.scatter(traj_reduced[0, 0], traj_reduced[0, 1], traj_reduced[0, 2],
                      color=color_map[name], s=60, marker='o', edgecolor='black')
            ax.scatter(traj_reduced[-1, 0], traj_reduced[-1, 1], traj_reduced[-1, 2],
                      color=color_map[name], s=60, marker='s', edgecolor='black')
            
            start_idx += traj_length
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(f'3D High-Dimensional Trajectories {title_suffix}')
        ax.legend()
        
    plt.tight_layout()
    return fig


def plot_parameter_evolution(
    results: Dict[str, TrajectoryData],
    param_indices: Optional[List[int]] = None,
    max_params: int = 6,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot evolution of individual parameters over optimization steps.
    
    Args:
        results: Dictionary of trajectory data
        param_indices: Specific parameter indices to plot (if None, plot first few)
        max_params: Maximum number of parameters to plot
        figsize: Figure size
    """
    # Determine which parameters to plot
    sample_trajectory = next(iter(results.values()))
    param_dim = sample_trajectory.params[0].shape[0]
    
    if param_indices is None:
        param_indices = list(range(min(max_params, param_dim)))
    
    # Create subplots
    n_params = len(param_indices)
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot parameter evolution
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for idx, param_idx in enumerate(param_indices):
        ax = axes[idx] if idx < len(axes) else plt.subplot(rows, cols, idx + 1)
        
        for (name, trajectory), color in zip(results.items(), colors):
            params_array = np.array(trajectory.params)
            param_values = params_array[:, param_idx]
            steps = np.arange(len(param_values))
            
            ax.plot(steps, param_values, color=color, linewidth=2, 
                   alpha=0.8, label=name if idx == 0 else "")
        
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel(f'Parameter {param_idx}')
        ax.set_title(f'Parameter {param_idx} Evolution')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide empty subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


# Example usage function
def run_visualization_example():
    """Example of how to use visualization tools"""
    
    # Import optimizers and models
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'validation'))
    
    from adam.solution import adam_init, adam_update
    from sgd.solution import sgd_init, sgd_update
    from .contract import RosenbrockModel, SimpleQuadraticModel, OptimizerConfig
    from .compare import compare_optimizers
    
    # Setup 2D problem
    model = RosenbrockModel(dim=2)
    initial_params = jnp.array([-2.0, 2.0])
    batches = [None]
    
    optimizers = {
        'Adam': OptimizerConfig('Adam', adam_init, adam_update, {'lr': 0.01}),
        'SGD': OptimizerConfig('SGD', sgd_init, sgd_update, {'lr': 0.001, 'momentum': 0.9})
    }
    
    # Run comparison
    results = compare_optimizers(model, optimizers, initial_params, batches, num_steps=500)
    
    # Create visualizations
    print("Creating 2D trajectory plot...")
    plot_2d_trajectories(results, model, x_range=(-2.5, 2.5), y_range=(-1, 3))
    
    print("Creating animation...")
    anim = animate_2d_trajectories(results, model, x_range=(-2.5, 2.5), y_range=(-1, 3))
    
    print("Creating parameter evolution plot...")
    plot_parameter_evolution(results)
    
    plt.show()
    
    # High-dimensional example
    print("\nTesting high-dimensional visualization...")
    high_dim_model = SimpleQuadraticModel(dim=10, condition_number=100)
    high_initial = jax.random.normal(jax.random.PRNGKey(42), (10,))
    
    high_results = compare_optimizers(high_dim_model, optimizers, high_initial, batches, num_steps=200)
    
    plot_high_dimensional_trajectories(high_results, method='pca')
    plot_parameter_evolution(high_results, max_params=4)
    
    plt.show()


if __name__ == "__main__":
    run_visualization_example()