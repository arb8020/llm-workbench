# Optimizer Examples

This directory contains educational examples of optimization algorithms commonly used in machine learning, implemented in JAX.

## Structure

```
examples/optimizers/
├── validation/           # Individual optimizer implementations with test suites
│   ├── adam/            # Adam optimizer (adaptive moments)
│   ├── adamw/           # AdamW optimizer (decoupled weight decay)
│   ├── sgd/             # SGD with momentum, weight decay, Nesterov
│   └── muon/            # Muon optimizer (momentum + orthogonalization)
└── visualization/       # Tools for visualizing optimizer behavior
    ├── contract.py      # Optimizer-model contract definitions
    ├── compare.py       # Multi-optimizer comparison utilities
    └── visualize.py     # Loss surface visualization tools
```

## Optimizer-Model Contract

All optimizers in this directory follow a consistent functional interface:

### Core Contract
```python
# Model provides: loss function + gradient function
def loss_fn(params: jnp.ndarray) -> float
def grad_fn(params: jnp.ndarray) -> jnp.ndarray  # same shape as params

# Optimizer provides: init + update functions
def optimizer_init(params: jnp.ndarray, **kwargs) -> OptimizerState
def optimizer_update(grads, state, params, **kwargs) -> (updates, new_state)

# Usage pattern:
state = optimizer_init(initial_params, lr=0.001)
for step in range(num_steps):
    grads = grad_fn(params)
    updates, state = optimizer_update(grads, state, params, lr=0.001)
    params = params + updates
```

### State Management
- All optimizer states use `NamedTuple` for immutability and JAX compatibility
- States contain step counters and momentum buffers as needed
- Follow JAX functional programming patterns (no in-place mutations)

## Usage Examples

### Individual Optimizer Testing
```bash
# Test your implementation against reference
python examples/optimizers/validation/adam/compare.py --mode skeleton
python examples/optimizers/validation/adam/compare.py --mode solution

# Same pattern for all optimizers
python examples/optimizers/validation/adamw/compare.py --mode solution
python examples/optimizers/validation/sgd/compare.py --mode solution
python examples/optimizers/validation/muon/compare.py --mode solution
```

### Multi-Optimizer Comparison
```python
from examples.optimizers.visualization.compare import compare_optimizers
from examples.optimizers.validation.adam.solution import adam_init, adam_update

# Define your model
class MyModel:
    def loss(self, params, batch): ...
    def grad(self, params, batch): ...

# Compare multiple optimizers
optimizers = {
    'Adam': (adam_init, adam_update, {'lr': 0.001}),
    'SGD': (sgd_init, sgd_update, {'lr': 0.01, 'momentum': 0.9})
}

results = compare_optimizers(model, optimizers, initial_params, batches)
```

### Loss Surface Visualization
```python
from examples.optimizers.visualization.visualize import visualize_trajectories

# Visualize how different optimizers traverse loss surfaces
visualize_trajectories(results, model)
```

## Educational Goals

This collection is designed to help understand:

1. **Optimizer Algorithms** - How different optimizers work mathematically
2. **JAX Patterns** - Functional programming with immutable states
3. **Comparative Analysis** - How optimizers behave on different loss surfaces
4. **Practical Implementation** - Real-world optimizer usage patterns

## Optimizer Characteristics

| Optimizer | Key Features | Best For |
|-----------|-------------|----------|
| **SGD** | Simple, momentum variants | Baseline, well-conditioned problems |
| **Adam** | Adaptive learning rates, bias correction | General purpose, sparse gradients |
| **AdamW** | Decoupled weight decay | Regularized training, transformers |
| **Muon** | Gradient orthogonalization | Experimental, ill-conditioned problems |

## Implementation Notes

- All implementations match PyTorch/Optax reference behavior
- Comprehensive test suites verify correctness
- Modular design allows easy integration into larger projects
- Educational comments explain the mathematical foundations

## Getting Started

1. **Learn the basics**: Start with SGD validation to understand the pattern
2. **Explore variants**: Try Adam/AdamW to see adaptive learning rates
3. **Experiment**: Use Muon to understand cutting-edge techniques
4. **Visualize**: Compare optimizers on your own loss surfaces

The goal is to build intuition for when and why to use different optimizers in practice.