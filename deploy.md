# Remote GPU Deployment with Broker + Bifrost

## Quick Commands

### Deploy code and run JAX script (one command!)
```bash
broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost deploy {} 'uv run python engine/scripts/hello_jax.py'
```

### Create new GPU instance with specific name
```bash
broker create --name "jax-hello-world" --min-vram 8 --max-price 0.40
```

## Available JAX Scripts

- `engine/scripts/hello_jax.py` - Simple matrix multiplication
- `engine/scripts/hello_attn_jax.py` - Attention mechanism implementation

## Workflow

1. **Create GPU instance** (if needed):
   ```bash
   broker create --name "jax-hello-world" --min-vram 8 --max-price 0.40
   ```

2. **Deploy and run script**:
   ```bash
   broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost deploy {} 'uv run python engine/scripts/hello_jax.py'
   ```

3. **Clean up when done**:
   ```bash
   broker terminate $(broker instances list --name "jax-hello-world" --simple | cut -d, -f1)
   ```

## How it works

- **broker** provisions GPU instances on cloud providers
- **bifrost deploy** automatically syncs your local code and runs the command
- JAX automatically detects and uses the remote GPU
- Results are displayed locally while computation runs remotely

## Alternative Commands

### Run different scripts
```bash
# Run attention script
broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost deploy {} 'uv run python engine/scripts/hello_attn_jax.py'

# Run with environment variables
broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost deploy {} --env CUDA_VISIBLE_DEVICES=0 'uv run python engine/scripts/hello_jax.py'
```

### Manual deployment (if you need more control)
```bash
# Deploy code only
broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost push --ssh {}

# Execute command only  
broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost exec {} 'uv run python engine/scripts/hello_jax.py'
```

### Instance management
```bash
# List all instances
broker instances list

# Get SSH connection for specific instance
broker instances list --name "jax-hello-world" --ssh-only

# Check instance status
broker instances list --name "jax-hello-world"

# Terminate instance
broker terminate <instance-id>
```