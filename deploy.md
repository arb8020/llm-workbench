# Remote GPU Deployment with Broker + Bifrost

## 🎉 NEW: Google Colab Integration

### Create Jupyter-enabled instance for Colab
```bash
# Auto-start Jupyter with random password
broker create --jupyter --max-price 0.50

# Custom Jupyter password for easy connection
broker create --jupyter --jupyter-password "mypass123" --min-vram 16
```

### Connect Google Colab to remote GPU
1. **Wait for SSH to be ready:**
   ```bash
   broker instances status <instance-id>
   ```

2. **Create SSH tunnel (run in terminal):**
   ```bash
   ssh -p <port> root@<ip> -L 8888:localhost:8888
   ```

3. **In Google Colab:**
   - Click "Connect" → "Connect to local runtime"
   - Enter: `http://localhost:8888/?token=<your-password>`
   - ✅ Now using remote GPU with Colab interface!

### Benefits of Jupyter + Colab Integration
- 🚀 **Powerful GPUs**: Access high-end GPUs (RTX 4090, A100) at low cost
- 💻 **Familiar Interface**: Use Google Colab's notebook environment  
- 🔗 **Direct Access**: Jupyter also available via proxy: `https://<instance-id>-8888.proxy.runpod.net`
- 💰 **Cost Effective**: Pay only for actual usage, terminate when done

### Complete Example: ML Training with Colab + Remote GPU
```bash
# 1. Create GPU instance with Jupyter
broker create --jupyter --jupyter-password "ml-project" --gpu-type "RTX 4090" --name "ml-training"

# Output shows:
# ✅ Instance created: abc123def456
# 📓 Jupyter Lab:
#    🔗 Proxy URL: https://abc123def456-8888.proxy.runpod.net  
#    🔑 Token: ml-project

# 2. Wait for SSH (usually 2-5 minutes)
broker instances status abc123def456

# 3. Create SSH tunnel (keep this running)  
ssh -p 22091 root@69.30.85.197 -L 8888:localhost:8888

# 4. In Google Colab:
#    - Connect → Connect to local runtime
#    - Enter: http://localhost:8888/?token=ml-project
#    - Start coding with remote GPU power!

# 5. When done, clean up
broker instances terminate abc123def456
```

> **✅ Tested & Verified**: Both CLI and Python API Jupyter integration have been successfully tested with real instances. Jupyter starts automatically and is accessible via both proxy URL and SSH tunnel.

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

## Python API for Jupyter Integration

### Direct Python API usage
```python
from broker.client import GPUClient

# Create client
client = GPUClient()

# Create Jupyter-enabled instance
instance = client.create(
    client.price_per_hour < 0.50,
    start_jupyter=True,
    jupyter_password="mypass123", 
    exposed_ports=[8888],
    name="my-jupyter-gpu"
)

# Get connection info
print(f"Proxy URL: {instance.get_proxy_url(8888)}")
print(f"SSH tunnel: ssh -p {instance.ssh_port} {instance.ssh_username}@{instance.public_ip} -L 8888:localhost:8888")

# Clean up when done
instance.terminate()
```

## Alternative Commands

### Run different scripts
```bash
# Run attention script
broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost deploy {} 'uv run python engine/scripts/hello_attn_jax.py'

# Run with environment variables
broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost deploy {} --env CUDA_VISIBLE_DEVICES=0 'uv run python engine/scripts/hello_jax.py'
```

### Bifrost: Deploy Code to GPU

**Simple approach (direct commands):**
```bash
# 1. Get your instance SSH info
broker instances list  # Shows: root@69.30.85.226:22117

# 2. Push your entire codebase to the GPU
bifrost push "root@69.30.85.226:22117"

# 3. Run any Python script on the GPU
bifrost exec "root@69.30.85.226:22117" 'uv run python engine/scripts/dev/hello_jax.py'

# 4. Or combine push + execute in one command  
bifrost deploy "root@69.30.85.226:22117" 'uv run python engine/scripts/dev/hello_jax.py'
```

> **✅ Tested & Verified**: Bifrost successfully deploys the entire codebase with `uv sync` dependency installation, and JAX scripts run on remote GPU: `Available devices: [CudaDevice(id=0)]`

**Advanced piping (for automation):**
```bash
# Deploy code only (push your entire codebase to GPU)
broker instances list --ssh-only | head -1 | xargs -I {} bifrost push {}

# Execute command only (assumes code already deployed)
broker instances list --ssh-only | head -1 | xargs -I {} bifrost exec {} 'uv run python engine/scripts/dev/hello_jax.py'

# Or use specific instance by name
broker instances list --name "jax-hello-world" --ssh-only | xargs -I {} bifrost push {}
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