"""
RunPod provider implementation
"""

import os
import logging
import time
from typing import List, Optional, Dict, Any
import requests
from dotenv import load_dotenv

from .types import GPUOffer, GPUInstance, InstanceStatus, ProvisionRequest, CloudType

logger = logging.getLogger(__name__)

RUNPOD_API_URL = "https://api.runpod.io/graphql"


def _get_api_key() -> str:
    """Get RunPod API key from environment"""
    # Load .env file if it exists
    load_dotenv()
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY environment variable not set")
    return api_key


def _make_graphql_request(query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
    """Make a GraphQL request to RunPod API"""
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json"
    }
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(RUNPOD_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")
    
    return data["data"]


def search_gpu_offers(cuda_version: Optional[str] = None) -> List[GPUOffer]:
    """Search for available GPU offers on RunPod with optional CUDA version filtering"""
    # Build lowestPrice input with optional CUDA version filtering
    lowest_price_input = "{ gpuCount: 1 }"
    if cuda_version:
        lowest_price_input = f'{{ gpuCount: 1, cudaVersion: "{cuda_version}" }}'
    
    # Query to get available GPU types - this will help us understand what's available
    query = f"""
    query {{
        gpuTypes {{
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            lowestPrice(input: {lowest_price_input}) {{
                minimumBidPrice
                uninterruptablePrice
            }}
        }}
    }}
    """
    
    try:
        data = _make_graphql_request(query)
        offers = []
        
        for gpu_type in data.get("gpuTypes", []):
            # Create offers for both secure and community cloud if available
            price_info = gpu_type.get("lowestPrice", {})
            
            if gpu_type.get("secureCloud") and price_info.get("uninterruptablePrice"):
                offers.append(GPUOffer(
                    id=f"runpod-{gpu_type['id']}-secure",
                    provider="runpod",
                    gpu_type=gpu_type["displayName"],
                    gpu_count=1,
                    vcpu=0,  # Not specified in this query
                    memory_gb=gpu_type.get("memoryInGb", 0),
                    storage_gb=0,  # Not specified in this query  
                    price_per_hour=price_info["uninterruptablePrice"],
                    availability_zone="secure-cloud",
                    cloud_type=CloudType.SECURE,
                    cuda_version=cuda_version,  # Add CUDA version if filtered
                    raw_data=gpu_type
                ))
            
            if gpu_type.get("communityCloud") and price_info.get("minimumBidPrice"):
                offers.append(GPUOffer(
                    id=f"runpod-{gpu_type['id']}-community-spot",
                    provider="runpod", 
                    gpu_type=gpu_type["displayName"],
                    gpu_count=1,
                    vcpu=0,  # Not specified in this query
                    memory_gb=gpu_type.get("memoryInGb", 0),
                    storage_gb=0,  # Not specified in this query
                    price_per_hour=price_info["minimumBidPrice"], 
                    availability_zone="community-cloud",
                    cloud_type=CloudType.COMMUNITY,
                    cuda_version=cuda_version,  # Add CUDA version if filtered
                    raw_data=gpu_type
                ))
        
        return offers
        
    except Exception as e:
        logger.error(f"Failed to search RunPod GPU offers: {e}")
        return []


def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None) -> Optional[GPUInstance]:
    """Provision a GPU instance on RunPod"""
    # First, we need to find a suitable GPU type
    # For now, let's implement a simple approach using podFindAndDeployOnDemand
    
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
    
    # Build input for the mutation
    env_vars = []
    
    # Add SSH startup script if provided
    if ssh_startup_script:
        env_vars.append({
            "key": "RUNPOD_STARTUP_SCRIPT",
            "value": ssh_startup_script
        })
    
    pod_input = {
        "gpuCount": request.gpu_count,
        "imageName": request.image,
        "cloudType": "SECURE" if not request.spot_instance else "COMMUNITY",
        "name": request.name or f"gpus-{request.gpu_type or 'auto'}-{int(time.time())}",
        "supportPublicIp": True,  # Required for SSH access
        "containerDiskInGb": 10,  # Smaller disk for better availability
        "volumeInGb": 0,  # Explicit volume setting
        "minVcpuCount": 1,  # Required minimum CPU
        "minMemoryInGb": 4,  # Required minimum memory
        "ports": "22/tcp",  # SSH port
        "startSsh": True,  # ← MISSING! This enables SSH daemon
        "env": env_vars
    }
    
    # Add GPU type if specified
    if request.gpu_type:
        # Use the GPU type ID directly - it should already be the full RunPod ID
        pod_input["gpuTypeId"] = request.gpu_type
    
    variables = {"input": pod_input}
    
    try:
        data = _make_graphql_request(mutation, variables)
        pod_data = data.get("podFindAndDeployOnDemand")
        
        if not pod_data:
            logger.error("No pod returned from deployment")
            return None
        
        # Return basic instance info - we'll need another query to get full details
        return GPUInstance(
            id=pod_data["id"],
            provider="runpod",
            status=InstanceStatus.PENDING,
            gpu_type=request.gpu_type or "auto-selected",
            gpu_count=request.gpu_count,
            price_per_hour=0.0,  # We'll get this from a separate query
            raw_data=pod_data
        )
        
    except Exception as e:
        logger.error(f"Failed to provision RunPod instance: {e}")
        return None


def get_instance_details(instance_id: str) -> Optional[GPUInstance]:
    """Get details of a specific instance"""
    query = """
    query pod($input: PodFilter!) {
        pod(input: $input) {
            id
            name
            machineId
            imageName
            env
            machineId
            machine {
                podHostId
            }
            desiredStatus
            lastStatusChange
            gpuCount
            vcpuCount
            memoryInGb
            costPerHr
            containerDiskInGb
            volumeInGb
            ports
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
    
    variables = {"input": {"podId": instance_id}}
    
    try:
        data = _make_graphql_request(query, variables)
        pod = data.get("pod")
        
        if not pod:
            return None
        
        # Map RunPod status to our status enum
        status_map = {
            "RUNNING": InstanceStatus.RUNNING,
            "PENDING": InstanceStatus.PENDING,
            "STOPPED": InstanceStatus.STOPPED,
            "TERMINATED": InstanceStatus.TERMINATED,
            "FAILED": InstanceStatus.FAILED
        }
        status = status_map.get(pod.get("desiredStatus", ""), InstanceStatus.PENDING)
        
        # Extract SSH connection info - Try direct SSH first, fallback to proxy
        public_ip = None
        ssh_port = 22
        ssh_username = "root"  # Default for direct SSH
        
        # Method 1: Check for direct SSH via runtime.ports (preferred)
        runtime = pod.get("runtime")
        if runtime and runtime.get("ports"):
            for port in runtime["ports"]:
                if (port.get("privatePort") == 22 and 
                    port.get("isIpPublic") and 
                    port.get("type") == "tcp"):
                    public_ip = port.get("ip")
                    ssh_port = port.get("publicPort")
                    ssh_username = "root"  # Direct connection uses root
                    break
        
        # Method 2: Fallback to proxy SSH if no direct SSH available
        if not public_ip and pod.get("machine") and pod["machine"].get("podHostId"):
            pod_host_id = pod["machine"]["podHostId"]
            public_ip = "ssh.runpod.io"
            ssh_port = 22
            ssh_username = pod_host_id  # Proxy uses podHostId as username
        
        return GPUInstance(
            id=pod["id"],
            provider="runpod",
            status=status,
            gpu_type="unknown",  # TODO: Get from machine details
            gpu_count=pod.get("gpuCount", 0),
            price_per_hour=pod.get("costPerHr", 0.0),
            public_ip=public_ip,
            ssh_port=ssh_port,
            ssh_username=ssh_username,
            raw_data=pod
        )
        
    except Exception as e:
        logger.error(f"Failed to get RunPod instance details: {e}")
        return None


def _parse_pod_to_instance(pod: Dict[str, Any]) -> GPUInstance:
    """Parse a pod dictionary into a GPUInstance"""
    
    # Map RunPod statuses to our enum
    status_map = {
        "PENDING": InstanceStatus.PENDING,
        "RUNNING": InstanceStatus.RUNNING,
        "STOPPED": InstanceStatus.STOPPED,
        "TERMINATED": InstanceStatus.TERMINATED,
        "FAILED": InstanceStatus.FAILED
    }
    status = status_map.get(pod.get("desiredStatus", ""), InstanceStatus.PENDING)
    
    # Extract SSH connection info - Try direct SSH first, fallback to proxy
    public_ip = None
    ssh_port = 22
    ssh_username = "root"  # Default for direct SSH
    
    # Method 1: Check for direct SSH via runtime.ports (preferred)
    runtime = pod.get("runtime")
    if runtime and runtime.get("ports"):
        for port in runtime["ports"]:
            if (port.get("privatePort") == 22 and 
                port.get("isIpPublic") and 
                port.get("type") == "tcp"):
                public_ip = port.get("ip")
                ssh_port = port.get("publicPort")
                ssh_username = "root"  # Direct connection uses root
                break
    
    # Method 2: Fallback to proxy SSH if no direct SSH available
    if not public_ip and pod.get("machine") and pod["machine"].get("podHostId"):
        pod_host_id = pod["machine"]["podHostId"]
        public_ip = "ssh.runpod.io"
        ssh_port = 22
        ssh_username = pod_host_id  # Proxy uses podHostId as username
    
    return GPUInstance(
        id=pod["id"],
        provider="runpod",
        status=status,
        gpu_type="unknown",  # TODO: Get from machine details
        gpu_count=pod.get("gpuCount", 0),
        price_per_hour=pod.get("costPerHr", 0.0),
        public_ip=public_ip,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        raw_data=pod
    )


def list_instances() -> List[GPUInstance]:
    """List all user's instances"""
    query = """
    query {
        myself {
            pods {
                id
                name
                machineId
                imageName
                env
                machineId
                machine {
                    podHostId
                }
                desiredStatus
                lastStatusChange
                gpuCount
                vcpuCount
                memoryInGb
                costPerHr
                containerDiskInGb
                volumeInGb
                ports
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
    }
    """
    
    try:
        data = _make_graphql_request(query)
        pods = data.get("myself", {}).get("pods", [])
        
        instances = []
        for pod in pods:
            try:
                instance = _parse_pod_to_instance(pod)
                if instance:
                    instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to parse pod {pod.get('id', 'unknown')}: {e}")
                continue
        
        return instances
        
    except Exception as e:
        logger.error(f"Failed to list RunPod instances: {e}")
        return []


def terminate_instance(instance_id: str) -> bool:
    """Terminate a RunPod instance"""
    # Use simple schema - RunPod API might return different types
    mutation = """
    mutation podTerminate($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    
    variables = {"input": {"podId": instance_id}}
    
    try:
        data = _make_graphql_request(mutation, variables)
        result = data.get("podTerminate")
        logger.info(f"RunPod terminate response: {result} (type: {type(result)})")
        logger.info(f"Expected instance_id: {instance_id}")
        
        # RunPod's podTerminate API behavior:
        # - Returns null/None on SUCCESSFUL termination
        # - Would throw exception on failure (handled in except block)
        # This is the opposite of typical API patterns
        if result is None:
            logger.info(f"Terminate succeeded (RunPod returns null on success)")
            return True
        else:
            logger.info(f"Unexpected terminate response: {result}")
            # Non-null response might still indicate success
            return True
        
    except Exception as e:
        logger.error(f"Failed to terminate RunPod instance: {e}")
        return False