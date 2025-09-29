"""
RunPod provider implementation
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from ..types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProvisionRequest

logger = logging.getLogger(__name__)

RUNPOD_API_URL = "https://api.runpod.io/graphql"


def _build_ports_string(exposed_ports: Optional[List[int]], enable_http_proxy: bool) -> str:
    """Build ports string for RunPod instance configuration.
    
    Args:
        exposed_ports: List of ports to expose (e.g., [8000] for vLLM)
        enable_http_proxy: Whether to enable HTTP proxy for exposed ports
        
    Returns:
        Ports string in RunPod format (e.g., "22/tcp,8000/http")
    """
    ports = ["22/tcp"]  # Always include SSH
    
    if exposed_ports and enable_http_proxy:
        for port in exposed_ports:
            ports.append(f"{port}/http")
    elif exposed_ports:
        # TCP only if HTTP proxy disabled
        for port in exposed_ports:
            ports.append(f"{port}/tcp")
            
    return ",".join(ports)


def _make_graphql_request(query: str, variables: Optional[Dict] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Make a GraphQL request to RunPod API

    Args:
        query: GraphQL query string
        variables: Optional GraphQL variables
        api_key: RunPod API key (required)
    """
    if not api_key:
        raise ValueError("RunPod API key is required but was not provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    logger.debug("RunPod GraphQL request %s (api key ...%s)", query.split("\n")[0][:60], api_key[-4:] if api_key else "none")
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    try:
        response = requests.post(
            RUNPOD_API_URL,
            json=payload,
            headers=headers,
            timeout=(5, 5),
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        logger.error("RunPod GraphQL request timed out")
        raise
    except requests.RequestException as exc:
        logger.error(f"RunPod GraphQL request failed: {exc}")
        raise

    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")
    
    return data["data"]


def search_gpu_offers(cuda_version: Optional[str] = None, manufacturer: Optional[str] = None,
                      memory_gb: Optional[int] = None, container_disk_gb: Optional[int] = None,
                      api_key: Optional[str] = None) -> List[GPUOffer]:
    """Search for available GPU offers on RunPod with optional CUDA version and manufacturer filtering"""
    # Build lowestPrice input - RunPod API only supports basic parameters
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
            manufacturer
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
        data = _make_graphql_request(query, api_key=api_key)
        offers = []
        
        for gpu_type in data.get("gpuTypes", []):
            # Filter by manufacturer if specified
            if manufacturer and gpu_type.get("manufacturer"):
                gpu_manufacturer = gpu_type["manufacturer"].lower()
                if manufacturer.lower() not in gpu_manufacturer:
                    continue
            
            # Create offers for both secure and community cloud if available
            price_info = gpu_type.get("lowestPrice", {})
            
            if gpu_type.get("secureCloud") and price_info.get("uninterruptablePrice"):
                gpu_vram = gpu_type.get("memoryInGb", 0)
                offers.append(GPUOffer(
                    id=f"runpod-{gpu_type['id']}-secure",
                    provider="runpod",
                    gpu_type=gpu_type["displayName"],
                    gpu_count=1,
                    vcpu=0,  # Not specified in this query
                    memory_gb=gpu_vram,  # For backward compatibility - this is actually GPU VRAM
                    vram_gb=gpu_vram,    # Properly populate VRAM field
                    storage_gb=0,  # Not specified in this query  
                    price_per_hour=price_info["uninterruptablePrice"],
                    availability_zone="secure-cloud",
                    cloud_type=CloudType.SECURE,
                    cuda_version=cuda_version,  # Add CUDA version if filtered
                    manufacturer=gpu_type.get("manufacturer"),
                    raw_data=gpu_type
                ))
            
            if gpu_type.get("communityCloud") and price_info.get("minimumBidPrice"):
                gpu_vram = gpu_type.get("memoryInGb", 0)
                offers.append(GPUOffer(
                    id=f"runpod-{gpu_type['id']}-community-spot",
                    provider="runpod", 
                    gpu_type=gpu_type["displayName"],
                    gpu_count=1,
                    vcpu=0,  # Not specified in this query
                    memory_gb=gpu_vram,  # For backward compatibility - this is actually GPU VRAM
                    vram_gb=gpu_vram,    # Properly populate VRAM field
                    storage_gb=0,  # Not specified in this query
                    price_per_hour=price_info["minimumBidPrice"], 
                    availability_zone="community-cloud",
                    cloud_type=CloudType.COMMUNITY,
                    cuda_version=cuda_version,  # Add CUDA version if filtered
                    manufacturer=gpu_type.get("manufacturer"),
                    raw_data=gpu_type
                ))
        
        return offers
        
    except Exception as e:
        logger.error(f"Failed to search RunPod GPU offers: {e}")
        return []


def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None, api_key: Optional[str] = None) -> Optional[GPUInstance]:
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
    
    # Add Jupyter password if provided
    if request.jupyter_password:
        env_vars.append({
            "key": "JUPYTER_PASSWORD",
            "value": request.jupyter_password
        })
    
    pod_input = {
        "gpuCount": request.gpu_count,
        "imageName": request.image,
        "cloudType": "SECURE" if not request.spot_instance else "COMMUNITY",
        "name": request.name or f"gpus-{request.gpu_type or 'auto'}-{int(time.time())}",
        "supportPublicIp": True,  # Required for SSH access
        "containerDiskInGb": request.container_disk_gb or 50,  # Default 50GB for ML workloads, configurable
        "volumeInGb": request.volume_disk_gb or 0,  # Default 0GB, configurable
        "minVcpuCount": 1,  # Required minimum CPU
        "minMemoryInGb": request.memory_gb or 4,  # Use requested memory or default 4GB minimum
        "ports": _build_ports_string(request.exposed_ports, request.enable_http_proxy),
        "startSsh": True,  # This enables SSH daemon
        "startJupyter": request.start_jupyter,  # Auto-start Jupyter Lab
        "env": env_vars
    }
    
    # Add GPU type if specified
    if request.gpu_type:
        # Use the GPU type ID directly - it should already be the full RunPod ID
        pod_input["gpuTypeId"] = request.gpu_type
    
    variables = {"input": pod_input}
    
    try:
        data = _make_graphql_request(mutation, variables, api_key=api_key)
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
            name=request.name,  # Fix: Use name from request
            price_per_hour=0.0,  # We'll get this from a separate query
            raw_data=pod_data,
            api_key=api_key  # Store API key for instance methods
        )
        
    except Exception as e:
        logger.error(f"Failed to provision RunPod instance: {e}")
        return None


def get_instance_details_enhanced(instance_id: str, api_key: Optional[str] = None) -> Optional[dict]:
    """Get comprehensive details of a specific instance with all available fields"""
    query = """
    query pod($input: PodFilter!) {
        pod(input: $input) {
            id
            name
            machineId
            imageName
            env
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
            machine {
                podHostId
                gpuType {
                    displayName
                    manufacturer
                    memoryInGb
                }
            }
        }
    }
    """
    
    variables = {"input": {"podId": instance_id}}
    
    try:
        response = _make_graphql_request(query, variables, api_key=api_key)
        pod_data = response.get("pod")
        
        if not pod_data:
            return None
        
        return pod_data
        
    except Exception as e:
        logger.error(f"Failed to get RunPod instance details: {e}")
        return None


def get_instance_details(instance_id: str, api_key: Optional[str] = None) -> Optional[GPUInstance]:
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
                gpuType {
                    displayName
                    manufacturer
                    memoryInGb
                }
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
        data = _make_graphql_request(query, variables, api_key=api_key)
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
        
        # Extract GPU type from machine details
        gpu_type = "unknown"
        if pod.get("machine") and pod["machine"].get("gpuType") and pod["machine"]["gpuType"].get("displayName"):
            gpu_type = pod["machine"]["gpuType"]["displayName"]
        
        return GPUInstance(
            id=pod["id"],
            provider="runpod",
            status=status,
            gpu_type=gpu_type,
            gpu_count=pod.get("gpuCount", 0),
            name=pod.get("name"),  # Fix: Properly map name field
            price_per_hour=pod.get("costPerHr", 0.0),
            public_ip=public_ip,
            ssh_port=ssh_port,
            ssh_username=ssh_username,
            raw_data=pod,
            api_key=api_key  # Store API key for instance methods
        )
        
    except Exception as e:
        logger.error(f"Failed to get RunPod instance details: {e}")
        return None


def _parse_pod_to_instance(pod: Dict[str, Any], api_key: Optional[str] = None) -> GPUInstance:
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

    # Extract GPU type from machine details
    gpu_type = "unknown"
    if pod.get("machine") and pod["machine"].get("gpuType") and pod["machine"]["gpuType"].get("displayName"):
        gpu_type = pod["machine"]["gpuType"]["displayName"]

    return GPUInstance(
        id=pod["id"],
        provider="runpod",
        status=status,
        gpu_type=gpu_type,
        gpu_count=pod.get("gpuCount", 0),
        name=pod.get("name"),  # Fix: Properly map name field
        price_per_hour=pod.get("costPerHr", 0.0),
        public_ip=public_ip,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        raw_data=pod,
        api_key=api_key  # Store API key for instance methods
    )


def list_instances(api_key: Optional[str] = None) -> List[GPUInstance]:
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
                    gpuType {
                        displayName
                        manufacturer
                        memoryInGb
                    }
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
        data = _make_graphql_request(query, api_key=api_key)
        pods = data.get("myself", {}).get("pods", [])
        
        instances = []
        for pod in pods:
            try:
                instance = _parse_pod_to_instance(pod, api_key=api_key)
                if instance:
                    instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to parse pod {pod.get('id', 'unknown')}: {e}")
                continue
        
        return instances
        
    except Exception as e:
        logger.error(f"Failed to list RunPod instances: {e}")
        return []


def get_user_balance(api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get user balance and spending information from RunPod"""
    query = """
    query {
        myself {
            clientBalance
            currentSpendPerHr
            clientLifetimeSpend
            spendLimit
            creditAlertThreshold
            referralEarnings
        }
    }
    """
    
    try:
        data = _make_graphql_request(query, api_key=api_key)
        user_data = data.get("myself")
        
        if not user_data:
            return None
        
        # Convert to more readable format
        balance_info = {
            "provider": "runpod",
            "current_balance": user_data.get("clientBalance", 0),
            "current_spend_per_hour": user_data.get("currentSpendPerHr", 0),
            "lifetime_spend": user_data.get("clientLifetimeSpend", 0),
            "spend_limit": user_data.get("spendLimit"),
            "credit_alert_threshold": user_data.get("creditAlertThreshold"),
            "referral_earnings": user_data.get("referralEarnings", 0),
            "raw_data": user_data
        }
        
        return balance_info
        
    except Exception as e:
        logger.error(f"Failed to get RunPod user balance: {e}")
        return None


def terminate_instance(instance_id: str, api_key: Optional[str] = None) -> bool:
    """Terminate a RunPod instance"""
    # Use simple schema - RunPod API might return different types
    mutation = """
    mutation podTerminate($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    
    variables = {"input": {"podId": instance_id}}
    
    try:
        data = _make_graphql_request(mutation, variables, api_key=api_key)
        result = data.get("podTerminate")
        logger.info(f"RunPod terminate response: {result} (type: {type(result)})")
        logger.info(f"Expected instance_id: {instance_id}")
        
        # RunPod's podTerminate API behavior:
        # - Returns null/None on SUCCESSFUL termination
        # - Would throw exception on failure (handled in except block)
        # This is the opposite of typical API patterns
        if result is None:
            logger.info("Terminate succeeded (RunPod returns null on success)")
            return True
        else:
            logger.info(f"Unexpected terminate response: {result}")
            # Non-null response might still indicate success
            return True
        
    except Exception as e:
        logger.error(f"Failed to terminate RunPod instance: {e}")
        return False
