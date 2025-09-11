# Broker + Google Colab Integration TODO

## 🎉 SUCCESS! Integration Working

The `startJupyter` flag integration has been successfully implemented and tested:

✅ **Core Implementation Complete:**
- Added `start_jupyter: bool` and `jupyter_password: str` to ProvisionRequest
- Updated RunPod provider to send `startJupyter` flag to GraphQL API
- Updated main API to expose these parameters

✅ **Verification Tests Passed:**
- Jupyter accessible via RunPod proxy URL: `https://{instance_id}-8888.proxy.runpod.net`
- SSH tunnel working: `ssh -p {port} root@{ip} -L 8888:localhost:8888`
- Local access: `http://localhost:8888/?token={password}`

✅ **Ready for Google Colab Connection:**
```python
from broker.client import GPUClient
client = GPUClient()
instance = client.create(
    client.price_per_hour < 0.50,
    start_jupyter=True,
    jupyter_password="mypass123",
    exposed_ports=[8888]
)
# Proxy URL: instance.get_proxy_url(8888)
# SSH: ssh -p {instance.ssh_port} {instance.ssh_username}@{instance.public_ip} -L 8888:localhost:8888
```

## Context & Goal
Set up automated GPU provisioning with the broker system, then connect Google Colab to the remote instance for a seamless development experience.

## Key Findings

### RunPod API Endpoint & GraphQL Mutation
**File:** `broker/broker/providers/runpod.py:157-229`
**Endpoint:** `https://api.runpod.io/graphql`
**Mutation:** `podFindAndDeployOnDemand`

```python
def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None) -> Optional[GPUInstance]:
    mutation = """
    mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            machineId
            machine {
                podHostId
            }
        }
    }
    """
```

### Current ProvisionRequest Structure
**File:** `broker/broker/types.py:352-368`

```python
@dataclass
class ProvisionRequest:
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    name: Optional[str] = None
    max_price_per_hour: Optional[float] = None
    provider: Optional[str] = None
    spot_instance: bool = False
    ssh_startup_script: Optional[str] = None
    container_disk_gb: Optional[int] = None
    volume_disk_gb: Optional[int] = None
    exposed_ports: Optional[List[int]] = None
    enable_http_proxy: bool = True
    manufacturer: Optional[str] = None
```

### Current Pod Input Configuration
**File:** `broker/broker/providers/runpod.py:184-197`

```python
pod_input = {
    "gpuCount": request.gpu_count,
    "imageName": request.image,
    "cloudType": "SECURE" if not request.spot_instance else "COMMUNITY",
    "name": request.name or f"gpus-{request.gpu_type or 'auto'}-{int(time.time())}",
    "supportPublicIp": True,
    "containerDiskInGb": request.container_disk_gb or 50,
    "volumeInGb": request.volume_disk_gb or 0,
    "minVcpuCount": 1,
    "minMemoryInGb": 4,
    "ports": _build_ports_string(request.exposed_ports, request.enable_http_proxy),
    "startSsh": True,  # ← This enables SSH daemon
    "env": env_vars
}
```

## Tasks to Complete

### 1. ✅ COMPLETED: Research Phase
- [x] Found RunPod GraphQL endpoint: `https://api.runpod.io/graphql`
- [x] Located mutation: `podFindAndDeployOnDemand`
- [x] Identified missing fields for Jupyter setup
- [x] Found RunPod GraphQL spec shows `dockerArgs` and `startJupyter` fields

### 2. ✅ COMPLETED: Add Jupyter Support to ProvisionRequest

**Missing Fields to Add:**
From GraphQL spec at https://graphql-spec.runpod.io/:
```python
# Add to ProvisionRequest class:
docker_args: Optional[str] = None      # For custom startup commands
start_jupyter: bool = False            # Auto-start Jupyter flag
jupyter_password: Optional[str] = None # Jupyter password
```

**Modify pod_input in provision_instance():**
```python
pod_input = {
    # ... existing fields ...
    "startJupyter": request.start_jupyter,  # Add this
    "dockerArgs": request.docker_args,      # Add this
}

# Add JUPYTER_PASSWORD to env_vars if provided
if request.jupyter_password:
    env_vars.append({
        "key": "JUPYTER_PASSWORD", 
        "value": request.jupyter_password
    })
```

### 3. 🔲 TODO: Create Jupyter-Ready Provisioning Function

Create `broker/broker/colab.py`:
```python
def provision_for_colab(min_vram_gb: int = 8, jupyter_password: str = "colab123", **kwargs):
    """Provision instance optimized for Google Colab connection"""
    
    # Search for GPUs with minimum VRAM
    offers = search((gpus.vram_gb >= min_vram_gb), **kwargs)
    
    # Create with Jupyter pre-configured
    instance = create(
        offers[:3],
        start_jupyter=True,
        jupyter_password=jupyter_password,
        exposed_ports=[8888],
        enable_http_proxy=True,
        **kwargs
    )
    
    return instance, f"https://{instance.id}-8888.proxy.runpod.net"
```

### 4. 🔲 TODO: Create CLI Command

Add to `broker/broker/cli.py`:
```bash
# Usage: broker colab --min-vram 16 --jupyter-password mypass123
def colab_command():
    """Set up instance for Google Colab connection"""
    pass
```

### 5. ✅ COMPLETED: Verification Tests

**✅ Test 1: RunPod Proxy URL - PASSED**
- Successfully created instance with `start_jupyter=True` and `jupyter_password="test123"`
- Instance: RTX A5000 at $0.160/hr (instance ID: p9dwea7lgunsf4)
- Proxy URL: `https://p9dwea7lgunsf4-8888.proxy.runpod.net` 
- **Result: JUPYTER ACCESSIBLE VIA PROXY URL** ✅
- Status: 200 response with Jupyter content detected

**✅ Test 2: SSH Port Forward + localhost:8888 - PASSED**
- SSH connection successful: `ssh -p 22091 root@69.30.85.197 -L 8888:localhost:8888`
- Jupyter process confirmed running: `/usr/local/bin/jupyter-lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.token=test123`
- **Result: JUPYTER API RESPONDING LOCALLY** ✅
- API call `curl localhost:8888/api/status?token=test123` returned valid JSON
- Local access: `http://localhost:8888/?token=test123`

### 6. 🔲 TODO: Integration Test Script

Create `broker/test_colab_integration.py`:
```python
#!/usr/bin/env python3
"""
Test script to verify Broker + Colab integration works end-to-end
"""

def main():
    # 1. Provision instance with Jupyter
    # 2. Wait for ready
    # 3. Test proxy URL
    # 4. Test SSH tunnel setup  
    # 5. Provide connection instructions
    # 6. Clean up
    pass
```

## Implementation Priority

1. **Phase 1 (Manual Test):** Modify `ProvisionRequest` and `provision_instance()` to support `startJupyter=True`
2. **Phase 2 (Verify):** Test that Jupyter starts automatically and is accessible via proxy URL
3. **Phase 3 (SSH Tunnel):** Test SSH tunnel + localhost:8888 connection
4. **Phase 4 (CLI):** Create user-friendly CLI command `broker colab`
5. **Phase 5 (Documentation):** Document the complete workflow

## Expected Workflow

```python
# In Google Colab or terminal:
import broker as gpus

# Automatic Jupyter setup
instance, proxy_url = gpus.provision_for_colab(min_vram_gb=16)

# Wait for ready  
if instance.wait_until_ssh_ready():
    print(f"🚀 Jupyter ready at: {proxy_url}")
    print(f"📡 SSH tunnel: {instance.ssh_connection_string()}")
    
    # For Colab local runtime connection:
    ssh_cmd = f"ssh -p {instance.ssh_port} root@{instance.public_ip} -L 8888:localhost:8888"
    print(f"🔗 Run in terminal: {ssh_cmd}")
    print(f"🔗 Then connect Colab to: http://localhost:8888/?token=colab123")
```

## Files to Modify

1. `broker/broker/types.py` - Add Jupyter fields to ProvisionRequest
2. `broker/broker/providers/runpod.py` - Add startJupyter/dockerArgs to pod_input
3. `broker/broker/colab.py` - New file with colab-specific functions  
4. `broker/broker/cli.py` - Add colab command
5. `broker/test_colab_integration.py` - Integration test script

---

## 🔲 TODO: Add Stripe-Style Key/Value Metadata Support

### Context
Currently GPU instances only support basic naming (`name="experiment-worker-1"`). For better organization and filtering of instances across multiple experiments and projects, we need Stripe-style metadata support.

### Requirements
- **Key/Value Metadata**: `metadata={"experiment": "gsm8k_pilot", "project": "mats_neel", "worker_id": "worker_1"}`
- **Filtering Support**: Search/list instances by metadata keys
- **Provider Integration**: Map to provider-specific tagging systems (RunPod tags, AWS tags, etc.)

### Implementation Plan
```python
# Add to ProvisionRequest
metadata: Optional[Dict[str, str]] = None

# Add to GPUClient
def list_instances(metadata_filter: Dict[str, str] = None) -> List[GPUInstance]:
    """List instances optionally filtered by metadata"""
    pass

def find_by_metadata(key: str, value: str) -> List[GPUInstance]:
    """Find instances with specific metadata"""
    pass
```

### Use Cases
- **Experiment Organization**: Group GPUs by experiment name, researcher, project
- **Cost Tracking**: Tag instances with cost centers, departments, research groups  
- **Automated Cleanup**: Find and terminate instances by metadata filters
- **Multi-tenant Management**: Isolate resources by user/team metadata

### Provider Support
- **RunPod**: Uses custom fields or name prefixes (investigate GraphQL schema)
- **AWS EC2**: Native tag support via boto3
- **Google Cloud**: Native label support
- **Azure**: Native tag support

**Priority**: Medium (would greatly improve multi-experiment workflows)