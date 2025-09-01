# Trainer Architecture

## Overview

The trainer module provides unified training orchestration for all LLM training types (pretraining, SFT, RL). It abstracts distributed computing complexity, weight management, and optimization, exposing a clean functional interface that separates computation from data sources.

## Core Principles

1. **Pure Interface**: Core training logic as pure functions `(weights, data, optimizer) -> new_weights`
2. **Backend Agnostic**: Support JAX, PyTorch, Megatron-Core seamlessly
3. **Strategy Pattern**: Training type differences handled as pluggable strategies
4. **Distribution Transparent**: Handle single GPU to multi-node clusters uniformly

## Directory Structure

```
trainer/
├── core/
│   ├── trainer.py           # Main training orchestrator
│   ├── interfaces.py        # Core abstractions and protocols
│   ├── optimizer.py         # Optimizer state management
│   └── distributed.py      # GPU topology & communication primitives
├── backends/
│   ├── base.py             # Backend interface
│   ├── megatron.py         # Megatron-Core backend (model/tensor parallel)
│   ├── jax.py              # JAX backend (reuse engine JAX implementation)
│   ├── pytorch.py          # Native PyTorch backend
│   └── mixed.py            # Mixed precision and FSDP support
├── strategies/
│   ├── base.py             # Training strategy interface
│   ├── sft.py              # Supervised fine-tuning strategy
│   ├── ppo.py              # PPO reinforcement learning strategy  
│   ├── pretrain.py         # Pretraining strategy (next token prediction)
│   └── dpo.py              # Direct Preference Optimization strategy
├── memory/
│   ├── checkpointing.py    # Model checkpointing and recovery
│   ├── offloading.py       # CPU/disk offloading for large models
│   └── profiling.py        # Memory usage profiling and optimization
└── coordination/
    ├── scheduling.py       # Training job scheduling and resource allocation  
    ├── synchronization.py  # Multi-worker synchronization primitives
    └── monitoring.py       # Training progress monitoring and logging
```

## Core Interfaces

### Optimizer Interface (JAX/PyTorch Agnostic)
```python
from typing import Protocol, Union, Any, Tuple
import torch
import jax.numpy as jnp

# Universal types
Tensor = Union[torch.Tensor, jnp.ndarray]
OptimizerState = Any  # PyTree for JAX, state_dict for PyTorch

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

### JAX/Optax Pattern
```python
# JAX optimizers follow Optax's functional pattern
import optax

class JaxAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.optax_optimizer = optax.adam(lr, beta1, beta2)
    
    def init(self, params: jnp.ndarray) -> OptimizerState:
        return self.optax_optimizer.init(params)
    
    def update(
        self, 
        grads: jnp.ndarray, 
        opt_state: OptimizerState, 
        params: jnp.ndarray
    ) -> Tuple[jnp.ndarray, OptimizerState]:
        return self.optax_optimizer.update(grads, opt_state, params)
```

### PyTorch Functional Pattern
```python
# PyTorch optimizers wrapped in functional interface
class PyTorchAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.betas = (beta1, beta2)
        self.eps = 1e-8
    
    def init(self, params: torch.Tensor) -> OptimizerState:
        return {
            'step': 0,
            'exp_avg': torch.zeros_like(params),
            'exp_avg_sq': torch.zeros_like(params)
        }
    
    def update(
        self, 
        grads: torch.Tensor, 
        opt_state: OptimizerState, 
        params: torch.Tensor
    ) -> Tuple[torch.Tensor, OptimizerState]:
        # Functional Adam implementation
        step = opt_state['step'] + 1
        exp_avg = self.betas[0] * opt_state['exp_avg'] + (1 - self.betas[0]) * grads
        exp_avg_sq = self.betas[1] * opt_state['exp_avg_sq'] + (1 - self.betas[1]) * grads**2
        
        bias_correction1 = 1 - self.betas[0] ** step
        bias_correction2 = 1 - self.betas[1] ** step
        
        updates = -self.lr * (exp_avg / bias_correction1) / (
            (exp_avg_sq / bias_correction2).sqrt() + self.eps
        )
        
        new_state = {
            'step': step,
            'exp_avg': exp_avg,
            'exp_avg_sq': exp_avg_sq
        }
        
        return updates, new_state
```

### Trainer Interface
```python
class Trainer:
    def __init__(self, backend: Backend, strategy: TrainingStrategy):
        self.backend = backend
        self.strategy = strategy
        
    def train_step(
        self, 
        batch: TrainingBatch, 
        model_state: ModelState, 
        optimizer_state: OptimizerState
    ) -> TrainingResult:
        """Pure function: Execute single training step"""
        
    def train_epoch(
        self, 
        dataloader: DataLoader, 
        model_state: ModelState
    ) -> ModelState:
        """Train full epoch with checkpointing and monitoring"""
```

### Backend Interface  
```python
@abstractmethod
class Backend:
    def forward_backward(
        self, 
        model: Model, 
        batch: TrainingBatch
    ) -> GradientInfo:
        """Compute forward pass and gradients"""
        
    def apply_gradients(
        self, 
        model: Model, 
        gradients: GradientInfo, 
        optimizer: Optimizer
    ) -> Model:
        """Apply gradients to update model weights"""
        
    def all_reduce_gradients(self, gradients: GradientInfo) -> GradientInfo:
        """Distributed gradient reduction"""
```

### Training Strategy Interface
```python
@abstractmethod  
class TrainingStrategy:
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        batch: TrainingBatch
    ) -> torch.Tensor:
        """Strategy-specific loss computation"""
        
    def prepare_batch(self, batch: TrainingBatch) -> TrainingBatch:
        """Strategy-specific batch preprocessing"""
        
    def compute_metrics(
        self, 
        logits: torch.Tensor, 
        batch: TrainingBatch
    ) -> Dict[str, float]:
        """Strategy-specific evaluation metrics"""
```

## Data Flow

```
DataLoader → TrainingBatch → Strategy.prepare_batch() → Backend.forward_backward() 
    ↓
GradientInfo → Backend.all_reduce_gradients() → Backend.apply_gradients() → ModelState
```

## Usage Examples

### SFT Training
```python
backend = JaxBackend(model_config)
strategy = SFTStrategy(loss_fn="cross_entropy")
trainer = Trainer(backend, strategy)

for batch in sft_dataloader:
    result = trainer.train_step(batch, model_state, optimizer_state)
    model_state = result.model_state
```

### RL Training  
```python
backend = MegatronBackend(model_config)
strategy = PPOStrategy(kl_coef=0.1, clip_range=0.2)
trainer = Trainer(backend, strategy)

for rollout_batch in rl_dataloader:
    result = trainer.train_step(rollout_batch, model_state, optimizer_state)
    model_state = result.model_state
```

### Distributed Training
```python
# Automatically handles multi-GPU/multi-node
trainer = Trainer(
    backend=MegatronBackend(
        tensor_parallel_size=4,
        pipeline_parallel_size=2
    ),
    strategy=PretrainStrategy()
)
```

## Key Abstractions

### ModelState
- **Weights**: Model parameters
- **Optimizer State**: Adam momentum, variance, etc.  
- **Metadata**: Training step, learning rate schedule, etc.

### TrainingResult
- **Updated ModelState**: New weights and optimizer state
- **Metrics**: Loss, gradients norms, timing info
- **Diagnostics**: Memory usage, communication stats

## Integration Points

- **Data Module**: Consumes `TrainingBatch` from dataloaders
- **Engine Module**: Reuses model implementations and tokenizers  
- **Rollouts Module**: Receives rollout data for RL training
- **Shared Module**: Logging, distributed utilities

## Extension Points

1. **Custom Backends**: Add new training frameworks  
2. **Custom Strategies**: Implement novel training objectives
3. **Memory Strategies**: Custom offloading and checkpointing
4. **Communication**: Custom distributed communication patterns