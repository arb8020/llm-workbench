# GPU Project Template

## 1. Add your dependencies to pyproject.toml

In the root `pyproject.toml`, add a new optional dependency group:

```toml
[project.optional-dependencies]
# Add this section for your project
my-project = [
    "rollouts",
    "broker",
    "bifrost",
    "torch>=2.4.0,<=2.7.1",
    "jax[cuda12]>=0.4.0",  # or jax[cpu] for CPU-only
    # Add your other dependencies here
]
```

Then install: `uv sync --extra my-project`

## 2. Deploy and run on GPU

```bash
# 1. Create GPU instance
broker create --name "my-project" --min-vram 8 --max-price 0.40

# 2. Get SSH connection and deploy code
broker instances list --name "my-project" --ssh-only | xargs -I {} bifrost deploy {} 'uv run python examples/template/main.py'

# 3. Clean up when done
broker terminate $(broker instances list --name "my-project" --simple | cut -d, -f1)
```

## One-liner (when instance exists)

```bash
broker instances list --name "my-project" --ssh-only | xargs -I {} bifrost deploy {} 'uv run python examples/template/main.py'
```

## Files

- `main.py`: Basic GPU detection script (replace with your code)
- `README.md`: This guide

That's it! The script will show nvidia-smi output and detect JAX/PyTorch GPUs.