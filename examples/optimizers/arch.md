# Optimizer Visualization Examples

## Overview

Educational examples for building intuition about different optimizers by visualizing their trajectories on various loss surfaces. Uses production optimizer implementations from the trainer module to ensure the toy examples reflect real behavior.

## Goals

1. **Build Intuition**: Understand how SGD, Adam, RMSprop, etc. behave on different landscapes
2. **Validate Production Code**: Test trainer optimizers on known problems before using on LLMs
3. **Experiment Safely**: Try new optimizers/hyperparameters on toy problems first
4. **Educational Resource**: Clear visualizations for understanding optimization dynamics

## Directory Structure

```
examples/optimizers/
├── surfaces/
│   ├── convex.py            # Simple quadratic bowls
│   ├── rosenbrock.py        # Classic banana-shaped function
│   ├── rastrigin.py         # Highly multi-modal surface
│   ├── saddle_point.py      # Saddle points and plateaus
│   ├── neural_toy.py        # Simple 2-layer network loss surfaces
│   └── adversarial.py       # Pathological cases for optimizers
├── trajectories/
│   ├── visualize_2d.py      # 2D contour plots with optimizer paths
│   ├── visualize_3d.py      # Interactive 3D surface visualizations
│   ├── animate_paths.py     # Animated optimizer trajectories
│   └── compare_side_by_side.py # Multi-optimizer comparisons
├── experiments/
│   ├── learning_rate_sweep.py    # LR sensitivity analysis
│   ├── momentum_effects.py       # Momentum parameter studies
│   ├── adaptive_lr_behavior.py   # Adam vs RMSprop vs AdaGrad
│   └── convergence_analysis.py   # Convergence rate comparisons
├── notebooks/
│   ├── optimizer_intuition.ipynb     # Interactive exploration
│   ├── hyperparameter_sensitivity.ipynb
│   └── pathological_cases.ipynb     # When optimizers fail
└── utils/
    ├── plotting.py          # Common visualization utilities
    ├── metrics.py          # Convergence metrics and analysis
    └── logging.py          # Trajectory recording utilities
```

## Core Components

### Optimizer State Contract (JAX/PyTorch Agnostic)

Following the JAX/Optax pattern of pure functions with immutable state:

```python
from typing import Protocol, Union, Any, Tuple
import torch
import jax.numpy as jnp

# Universal types for cross-framework compatibility
Tensor = Union[torch.Tensor, jnp.ndarray]  
OptimizerState = Any  # PyTree for JAX, dict for PyTorch

class Optimizer(Protocol):
    def init(self, params: Tensor) -> OptimizerState:
        """Initialize optimizer state from example parameters"""
        
    def update(
        self, 
        grads: Tensor, 
        opt_state: OptimizerState, 
        params: Tensor
    ) -> Tuple[Tensor, OptimizerState]:
        """Apply gradients, return (updates, new_state)"""
```

### Complete Trajectory Recording
```python
@dataclass
class OptimizerStep:
    """Single step of optimization - everything needed to reproduce the step"""
    gradients: Tensor                    # Input gradients
    opt_state: OptimizerState           # Optimizer state before step  
    params: Tensor                      # Parameters before step
    updates: Tensor                     # Computed parameter updates
    new_opt_state: OptimizerState       # Optimizer state after step
    new_params: Tensor                  # Parameters after step
    loss: float                         # Loss value at this step
    step_size: float                    # Effective step size taken

class OptimizerTrace:
    """Complete trajectory of optimization with full state history"""
    def __init__(self, optimizer: Optimizer, surface: LossSurface, initial_params: Tensor):
        self.optimizer = optimizer
        self.surface = surface
        self.opt_state = optimizer.init(initial_params)
        self.steps: List[OptimizerStep] = []
        self.current_params = initial_params
        
    def step(self) -> Tensor:
        """Take one optimization step, record everything, return new params"""
        gradients = self.surface.gradient(self.current_params)
        loss = self.surface(self.current_params)
        
        updates, new_opt_state = self.optimizer.update(
            gradients, self.opt_state, self.current_params
        )
        new_params = self.current_params + updates
        
        # Record the complete step for analysis
        step_record = OptimizerStep(
            gradients=gradients,
            opt_state=self.opt_state,
            params=self.current_params,
            updates=updates,
            new_opt_state=new_opt_state,
            new_params=new_params,
            loss=loss,
            step_size=jnp.linalg.norm(updates)  # L2 norm of updates
        )
        
        self.steps.append(step_record)
        self.opt_state = new_opt_state
        self.current_params = new_params
        
        return new_params
    
    def run_optimization(self, num_steps: int):
        """Run full optimization and record trajectory"""
        for _ in range(num_steps):
            self.step()
            
    def get_trajectory_data(self) -> Dict[str, List]:
        """Extract trajectory data for plotting and analysis"""
        return {
            'params': [step.params for step in self.steps],
            'losses': [step.loss for step in self.steps],
            'gradients': [step.gradients for step in self.steps],
            'step_sizes': [step.step_size for step in self.steps],
            'optimizer_states': [step.opt_state for step in self.steps]
        }
```

### Loss Surface Interface
```python
@abstractmethod
class LossSurface:
    def __call__(self, params: Tensor) -> float:
        """Compute loss at given parameters"""
        
    def gradient(self, params: Tensor) -> Tensor:
        """Compute gradient at given parameters"""
        
    def hessian(self, params: Tensor) -> Tensor:
        """Compute Hessian matrix (for curvature analysis)"""
        
    def plot_contours(self, x_range: tuple, y_range: tuple):
        """Generate contour plot of the surface"""
```

### Cross-Framework Optimizer Support
```python
# JAX/Optax optimizers (already functional)
from trainer.backends.jax import JaxAdam, JaxSGD, JaxRMSprop

# PyTorch functional optimizers
from trainer.backends.pytorch import PyTorchAdam, PyTorchSGD, PyTorchRMSprop

# Both can be used interchangeably in visualizations
jax_adam = JaxAdam(lr=0.01)
torch_adam = PyTorchAdam(lr=0.01)

# Same surface, same hyperparams, different backends
surface = RosenbrockFunction()
jax_trace = OptimizerTrace(jax_adam, surface, jnp.array([-2.0, 2.0]))
torch_trace = OptimizerTrace(torch_adam, surface, torch.tensor([-2.0, 2.0]))

# Compare trajectories across frameworks
jax_trace.run_optimization(1000)
torch_trace.run_optimization(1000)
```

## Example Loss Surfaces

### 1. Convex Quadratic
```python
class QuadraticBowl(LossSurface):
    """Simple convex surface - all optimizers should converge"""
    def __call__(self, params):
        x, y = params
        return x**2 + 10*y**2  # Elongated bowl
```

### 2. Rosenbrock Function
```python
class RosenbrockFunction(LossSurface):
    """Banana-shaped valley - tests optimizer's ability to navigate narrow corridors"""
    def __call__(self, params):
        x, y = params
        return (1 - x)**2 + 100 * (y - x**2)**2
```

### 3. Multi-Modal Surface
```python
class RastriginFunction(LossSurface):
    """Many local minima - tests global optimization ability"""
    def __call__(self, params):
        x, y = params
        return 20 + x**2 - 10*jnp.cos(2*jnp.pi*x) + y**2 - 10*jnp.cos(2*jnp.pi*y)
```

## Usage Examples

### Basic Trajectory Visualization
```python
from trainer.core.optimizer import AdamOptimizer, SGDOptimizer
from surfaces.rosenbrock import RosenbrockFunction
from trajectories.visualize_2d import plot_trajectory

# Use production optimizers from trainer module
adam = AdamOptimizer(lr=0.01, beta1=0.9, beta2=0.999)
sgd = SGDOptimizer(lr=0.01, momentum=0.9)

surface = RosenbrockFunction()
initial_params = jnp.array([-2.0, 2.0])

# Compare trajectories
adam_recorder = TrajectoryRecorder(surface, adam)
sgd_recorder = TrajectoryRecorder(surface, sgd)

adam_recorder.run_optimization(initial_params, num_steps=1000)
sgd_recorder.run_optimization(initial_params, num_steps=1000)

# Side-by-side visualization
plot_comparison([adam_recorder, sgd_recorder], surface)
```

### Hyperparameter Sensitivity
```python
from experiments.learning_rate_sweep import run_lr_sweep

learning_rates = [0.001, 0.01, 0.1, 1.0]
surface = RosenbrockFunction()

for lr in learning_rates:
    optimizer = AdamOptimizer(lr=lr)
    convergence_metrics = run_lr_sweep(optimizer, surface, initial_params)
    # Analyze convergence rate vs learning rate
```

### Animation Creation
```python
from trajectories.animate_paths import create_animation

recorder = TrajectoryRecorder(surface, optimizer)
recorder.run_optimization(initial_params, num_steps=500)

# Create animated GIF of optimization trajectory
create_animation(recorder, filename="adam_rosenbrock.gif", fps=10)
```

## Integration with Trainer Module

```python
# Import production optimizers
from trainer.core.optimizer import (
    AdamOptimizer, 
    SGDOptimizer, 
    RMSpropOptimizer,
    AdaGradOptimizer
)

# Import backend computation utilities  
from trainer.backends.jax import compute_gradients
from trainer.core.interfaces import OptimizerState

# Use exact same optimizer implementations that train real LLMs
```

## Key Insights to Explore

1. **Learning Rate Sensitivity**: How do different optimizers respond to LR changes?
2. **Momentum Effects**: When does momentum help vs hurt convergence?
3. **Adaptive Learning Rates**: How do Adam/RMSprop adapt to different curvatures?
4. **Pathological Cases**: Where do optimizers fail and why?
5. **Hyperparameter Interactions**: How do β1, β2, ε affect Adam behavior?
6. **Convergence Patterns**: Oscillatory vs monotonic convergence

## Educational Value

- **Visual Understanding**: See optimizer behavior, don't just read about it
- **Parameter Intuition**: Understand what hyperparameters actually do
- **Failure Modes**: Recognize when optimizers struggle
- **Production Validation**: Test real optimizer code on known problems