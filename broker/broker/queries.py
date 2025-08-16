"""
Centralized GraphQL queries and mutations for RunPod API

All queries here have been tested and verified to work.
"""

# WORKING GPU SEARCH QUERY
GPU_SEARCH_QUERY = """
query {
    gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        lowestPrice(input: { gpuCount: 1 }) {
            minimumBidPrice
            uninterruptablePrice
        }
    }
}
"""

# WORKING POD PROVISIONING MUTATION
# This uses the corrected field requirements we discovered
POD_PROVISION_MUTATION = """
mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
    podFindAndDeployOnDemand(input: $input) {
        id
        machineId
    }
}
"""

# WORKING POD DETAILS QUERY - includes runtime ports for SSH
POD_DETAILS_QUERY = """
query pod($input: PodFilter!) {
    pod(input: $input) {
        id
        name
        desiredStatus
        lastStatusChange
        gpuCount
        vcpuCount
        memoryInGb
        costPerHr
        containerDiskInGb
        volumeInGb
        ports
        machine {
            podHostId
        }
        runtime {
            uptimeInSeconds
            ports {
                ip
                isIpPublic
                privatePort
                publicPort
                type
            }
            gpus {
                id
                gpuUtilPercent
                memoryUtilPercent
            }
        }
    }
}
"""

# WORKING POD TERMINATION MUTATION
POD_TERMINATE_MUTATION = """
mutation podTerminate($input: PodTerminateInput!) {
    podTerminate(input: $input) {
        id
    }
}
"""

# WORKING POD LIST QUERY
POD_LIST_QUERY = """
query {
    myself {
        pods {
            id
            name
            desiredStatus
            costPerHr
            gpuCount
            machine {
                podHostId
            }
            createdAt
        }
    }
}
"""

def get_working_pod_input(gpu_type_id: str, startup_script: Optional[str] = None) -> dict:
    """
    Get working pod input configuration with all required fields.
    
    This configuration has been tested and verified to work for provisioning.
    Based on our breakthrough discovery of hidden required fields.
    
    IMPORTANT: Do NOT include startup_script by default - it can cause provisioning failures.
    Only add startup script if explicitly provided and needed.
    """
    import time
    
    base_config = {
        "gpuCount": 1,
        "gpuTypeId": gpu_type_id,
        "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "cloudType": "SECURE",  # Use SECURE for better availability
        "name": f"gpus-{gpu_type_id.replace(' ', '-')}-{int(time.time())}",
        "supportPublicIp": True,  # Required for SSH access
        "containerDiskInGb": 50,  # Default 50GB for ML workloads (10GB causes out-of-space)
        "volumeInGb": 0,          # Explicit volume setting
        "minVcpuCount": 1,        # REQUIRED (not in schema but needed!)
        "minMemoryInGb": 4,       # REQUIRED (not in schema but needed!)
        "ports": "22/tcp",        # SSH port
        "startSsh": True,         # CRITICAL: Explicitly enable SSH service
        "globalNetwork": False,   # CRITICAL: Network configuration for SSH
    }
    
    # Only add startup script if explicitly provided
    # WARNING: startup scripts can cause provisioning failures
    if startup_script:
        base_config["env"] = [{
            "key": "RUNPOD_STARTUP_SCRIPT",
            "value": startup_script
        }]
    
    return base_config

# Known working GPU types (tested)
WORKING_GPU_TYPES = [
    "RTX A5000",
    "NVIDIA A100 80GB PCIe",
    "RTX 4090",
    "RTX 3090",
]

# GPU types that tend to have better availability
HIGH_AVAILABILITY_GPU_TYPES = [
    "RTX A5000",
    "RTX 4080",
    "RTX 3090",
]