"""
Optimizer Visualization Tools

This package provides comprehensive tools for visualizing optimizer behavior
and comparing different optimization algorithms on loss surfaces.

Key Components:
- contract.py: Defines the optimizer-model interface and standard models
- compare.py: Multi-optimizer comparison utilities and analysis
- visualize.py: Loss surface visualization and trajectory plotting

Example Usage:

```python
from examples.optimizers.visualization import (
    RosenbrockModel, OptimizerConfig, 
    compare_optimizers, plot_2d_trajectories
)
from examples.optimizers.validation.adam.solution import adam_init, adam_update

# Setup
model = RosenbrockModel(dim=2)
initial_params = jnp.array([-2.0, 2.0])

optimizers = {
    'Adam': OptimizerConfig('Adam', adam_init, adam_update, {'lr': 0.01})
}

# Compare and visualize
results = compare_optimizers(model, optimizers, initial_params, [None])
plot_2d_trajectories(results, model)
```
"""

from .contract import (
    OptimizationModel,
    TrajectoryData, 
    OptimizerConfig,
    optimize_and_track,
    SimpleQuadraticModel,
    RosenbrockModel
)

from .compare import (
    compare_optimizers,
    analyze_convergence,
    plot_loss_curves,
    plot_step_sizes,
    plot_convergence_summary,
    print_comparison_table
)

from .visualize import (
    plot_2d_loss_surface,
    plot_2d_trajectories,
    animate_2d_trajectories,
    plot_high_dimensional_trajectories,
    plot_parameter_evolution
)

__all__ = [
    # Contract classes
    'OptimizationModel',
    'TrajectoryData',
    'OptimizerConfig',
    'optimize_and_track',
    'SimpleQuadraticModel', 
    'RosenbrockModel',
    
    # Comparison tools
    'compare_optimizers',
    'analyze_convergence',
    'plot_loss_curves',
    'plot_step_sizes', 
    'plot_convergence_summary',
    'print_comparison_table',
    
    # Visualization tools
    'plot_2d_loss_surface',
    'plot_2d_trajectories',
    'animate_2d_trajectories',
    'plot_high_dimensional_trajectories',
    'plot_parameter_evolution'
]