# Optimizer Validation Refactoring Handoff

## Current Status

I was in the middle of refactoring the optimizer validation scripts to eliminate code duplication by adopting the GPT-2 comparison pattern used in `engine/scripts/dev/gpt2_jax/compare.py`.

## What Needs to Be Done

**Refactor all optimizer validation scripts from the current pattern:**
```
optimizer/
├── skeleton.py    # Implementation + full test suite
└── solution.py    # Implementation + duplicate test suite (200+ lines duplicated)
```

**To the GPT-2 comparison pattern:**
```
optimizer/
├── skeleton.py    # Just the core implementation functions
├── solution.py    # Just the core implementation functions  
└── compare.py     # Shared test harness with --mode skeleton/solution flag
```

## Reference Implementation to Study

**Key file:** `/Users/chiraagbalu/llm-workbench/engine/scripts/dev/gpt2_jax/compare.py`

This script shows the exact pattern:
1. Uses `argparse` with `--mode skeleton/solution` flag
2. Dynamically imports the right implementation: `from skeleton import gpt2_forward` vs `from solution import gpt2_forward`
3. Adapts different function signatures to common interface
4. Runs same test suite on both implementations
5. Provides consistent reporting

**Critical code sections to understand:**
- Lines 43-75: `load_gpt2_implementation(mode)` function that handles dynamic imports
- Lines 116-185: `compare_logits_across_batches()` - the shared test harness
- Lines 188-219: `print_summary()` - consistent reporting

## Current State of Optimizer Scripts

**Completed optimizers that need refactoring:**
- `/Users/chiraagbalu/llm-workbench/examples/optimizers/validation/adam/` (skeleton.py + solution.py)
- `/Users/chiraagbalu/llm-workbench/examples/optimizers/validation/adamw/` (skeleton.py + solution.py)  
- `/Users/chiraagbalu/llm-workbench/examples/optimizers/validation/sgd/` (skeleton.py + solution.py)
- `/Users/chiraagbalu/llm-workbench/examples/optimizers/validation/muon/` (skeleton.py, but solution.py not created yet)

## What I Started

I began refactoring the Adam skeleton by:
1. Removing all the test code and imports (torch, optax)
2. Keeping only the core implementation functions: `AdamState`, `adam_init()`, `adam_update()`
3. Updating usage docs to point to `compare.py --mode skeleton`

**What's been modified:** `/Users/chiraagbalu/llm-workbench/examples/optimizers/validation/adam/skeleton.py`

## Required Implementation Pattern

For each optimizer, create a `compare.py` that follows this structure:

### 1. Dynamic Import Function
```python
def load_optimizer_implementation(mode, optimizer_name):
    """Load optimizer implementation based on mode."""
    if mode == "skeleton":
        if optimizer_name == "adam":
            from skeleton import adam_update, adam_init, AdamState
            return adam_update, adam_init, AdamState
        # ... other optimizers
    elif mode == "solution":
        if optimizer_name == "adam":  
            from solution import adam_update, adam_init, AdamState
            return adam_update, adam_init, AdamState
        # ... other optimizers
```

### 2. Test Surface Functions
Extract these from the existing solutions (they're identical across skeleton/solution):
```python
def rosenbrock(params: jnp.ndarray) -> float:
    """Rosenbrock function: classic optimization test case"""
    
def rosenbrock_grad(params: jnp.ndarray) -> jnp.ndarray:  
    """Analytical gradient of Rosenbrock function"""
    
def quadratic_bowl(params: jnp.ndarray) -> float:
    """Simple quadratic function: x^2 + 10*y^2"""
    
def quadratic_grad(params: jnp.ndarray) -> jnp.ndarray:
    """Analytical gradient of quadratic bowl"""
```

### 3. Reference Implementation Function
```python
def get_reference_trajectories(initial_params, lr, surface_fn, grad_fn, num_steps=100, **optimizer_kwargs):
    """Get reference trajectories from PyTorch and Optax"""
    # PyTorch reference implementation
    # Optax reference implementation  
    return torch_trajectory, optax_trajectory
```

### 4. Comparison Test Suite
```python
def compare_optimizer_implementations(optimizer_update, optimizer_init, mode_name):
    """Run full test suite comparing our implementation vs references"""
    # Test on multiple surfaces: quadratic, Rosenbrock, ill-conditioned
    # Compare trajectories against PyTorch and Optax
    # Print detailed results
```

## Key Differences Per Optimizer

### Adam/AdamW
- **Functions:** `adam_update()`, `adam_init()`, `AdamState`
- **References:** `torch.optim.Adam`, `optax.adam`  
- **Key test:** Weight decay placement (AdamW vs Adam)

### SGD  
- **Functions:** `sgd_update()`, `sgd_init()`, `SGDState`
- **References:** `torch.optim.SGD`, `optax.sgd`
- **Key test:** Momentum variants, Nesterov momentum

### Muon
- **Functions:** `muon_update()`, `muon_init()`, `MuonState`, `orthogonalize_gradient()`
- **References:** Custom reference implementation (no standard library version)
- **Key test:** Gradient orthogonalization with Newton-Schulz iteration

## Code Duplication to Eliminate

Each current solution.py has ~200 lines of duplicate test code:
- Test surface functions (rosenbrock, quadratic_bowl, etc.)
- Reference trajectory generation
- Trajectory comparison logic
- Test reporting and success/failure logic
- Multiple test case configurations

**All of this should be moved to compare.py and shared.**

## Expected File Structure After Refactoring

```
examples/optimizers/validation/
├── adam/
│   ├── skeleton.py      # Just adam_update, adam_init, AdamState (~50 lines)
│   ├── solution.py      # Just adam_update, adam_init, AdamState (~50 lines)
│   └── compare.py       # All test logic (~200 lines)
├── adamw/
│   ├── skeleton.py      # Just adamw_update, adamw_init, AdamWState  
│   ├── solution.py      # Just adamw_update, adamw_init, AdamWState
│   └── compare.py       # All test logic
├── sgd/
│   ├── skeleton.py      # Just sgd_update, sgd_init, SGDState
│   ├── solution.py      # Just sgd_update, sgd_init, SGDState  
│   └── compare.py       # All test logic
└── muon/
    ├── skeleton.py      # Just muon_update, muon_init, MuonState, orthogonalize_gradient
    ├── solution.py      # Just muon_update, muon_init, MuonState, orthogonalize_gradient
    └── compare.py       # All test logic
```

## Usage After Refactoring

Students will run:
```bash
# Test skeleton implementation
python examples/optimizers/validation/adam/compare.py --mode skeleton

# Test solution implementation  
python examples/optimizers/validation/adam/compare.py --mode solution

# Same for all other optimizers
python examples/optimizers/validation/sgd/compare.py --mode skeleton
python examples/optimizers/validation/adamw/compare.py --mode solution
```

## Next Steps

1. **Complete Adam refactor** - Create `adam/compare.py` and strip down `adam/solution.py`
2. **Repeat for AdamW** - Same pattern
3. **Repeat for SGD** - Same pattern  
4. **Complete Muon** - Finish the solution.py first, then create compare.py
5. **Test all implementations** - Verify the refactor didn't break anything

## Success Criteria

After refactoring:
- No duplicated test code between skeleton and solution files
- Identical test results before and after refactor
- Clean, focused skeleton/solution files with just the core optimizer logic
- Shared test harnesses that provide consistent UX across all optimizers
- Students can easily compare their skeleton implementation against the working solution on identical test cases

The goal is to eliminate ~800 lines of duplicate code across the 4 optimizers while improving the educational experience.