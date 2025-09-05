"""
Main functional API for GPU cloud operations
"""

import logging
from typing import Any, Callable, List, Optional, Union

from .providers import runpod
from .query import QueryType
from .types import GPUInstance, GPUOffer, ProvisionRequest

logger = logging.getLogger(__name__)


def search(
    query: Optional[QueryType] = None,
    # Legacy parameters for backward compatibility
    gpu_type: Optional[str] = None,
    max_price_per_hour: Optional[float] = None,
    provider: Optional[str] = None,
    cuda_version: Optional[str] = None,
    manufacturer: Optional[str] = None,
    # New sorting parameters
    sort: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False
) -> List[GPUOffer]:
    """
    Search for available GPU offers across providers
    
    Args:
        query: Pandas-style query (e.g., gpus.gpu_type.contains("A100") & gpus.price_per_hour < 2.0)
        gpu_type: Legacy - Specific GPU type to search for
        max_price_per_hour: Legacy - Maximum price per hour filter
        provider: Specific provider to search (default: all providers)
        cuda_version: Filter by specific CUDA version (e.g., "12.0", "11.8")
        manufacturer: Filter by GPU manufacturer (e.g., "nvidia", "amd")
        sort: Callable to extract sort key (e.g., lambda x: x.memory_gb/x.price_per_hour)
        reverse: Sort in descending order (default: False)
    
    Returns:
        List of available GPU offers
    """
    offers = []
    
    # Get all offers from providers
    if provider is None or provider == "runpod":
        runpod_offers = runpod.search_gpu_offers(cuda_version=cuda_version, manufacturer=manufacturer)
        offers.extend(runpod_offers)
    
    # Apply pandas-style query if provided
    if query is not None:
        offers = [offer for offer in offers if query.evaluate(offer)]
    else:
        # Legacy filtering for backward compatibility
        if gpu_type:
            offers = [o for o in offers if gpu_type.lower() in o.gpu_type.lower()]
        
        if max_price_per_hour:
            offers = [o for o in offers if o.price_per_hour <= max_price_per_hour]
    
    # Sort by specified key or default to price
    if sort is not None:
        offers.sort(key=sort, reverse=reverse)
    else:
        # Default: sort by price (cheapest first)
        offers.sort(key=lambda x: x.price_per_hour, reverse=reverse)
    
    return offers


def get_instance(instance_id: str, provider: Optional[str] = None) -> Optional[GPUInstance]:
    """
    Get details of a specific instance
    
    Args:
        instance_id: Instance ID
        provider: Provider name (if known, for optimization)
    
    Returns:
        Instance details or None if not found
    """
    # Try RunPod first (for now, it's our only provider)
    if provider is None or provider == "runpod":
        instance = runpod.get_instance_details(instance_id)
        if instance:
            return instance
    
    return None


def terminate_instance(instance_id: str, provider: Optional[str] = None) -> bool:
    """
    Terminate a GPU instance
    
    Args:
        instance_id: Instance ID
        provider: Provider name (if known, for optimization)
    
    Returns:
        True if termination was successful
    """
    # Try RunPod first
    if provider is None or provider == "runpod":
        if runpod.terminate_instance(instance_id):
            return True
    
    return False


def create(
    query: Union[QueryType, List[GPUOffer], GPUOffer] = None,
    image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    name: Optional[str] = None,
    # Search parameters for when query is a filter
    gpu_type: Optional[str] = None,
    max_price_per_hour: Optional[float] = None,
    provider: Optional[str] = None,
    cuda_version: Optional[str] = None,
    manufacturer: Optional[str] = None,
    sort: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
    # Port exposure configuration
    exposed_ports: Optional[List[int]] = None,
    enable_http_proxy: bool = True,
    # Jupyter configuration  
    start_jupyter: bool = False,
    jupyter_password: Optional[str] = None,
    # Retry parameters
    max_attempts: int = 3,
    **kwargs
) -> Optional[GPUInstance]:
    """
    Provision GPU using pandas-style query, search results, or specific offer
    
    Args:
        query: Query object, list of offers, or single offer (if None, searches all)
        image: Docker image to use
        name: Optional instance name
        gpu_type: Filter by GPU type (used if query is None)
        max_price_per_hour: Filter by max price (used if query is None)
        provider: Filter by provider (used if query is None)
        cuda_version: Filter by CUDA version (used if query is None)
        manufacturer: Filter by GPU manufacturer (used if query is None)
        sort: Sort key function (used if query is None)
        reverse: Sort order (used if query is None)
        max_attempts: Try up to this many offers before giving up
        **kwargs: Additional provisioning parameters
    
    Returns:
        Provisioned GPU instance or None if failed
        
    Examples:
        # Provision best value GPU (memory/price ratio)
        create(sort=lambda x: x.memory_gb/x.price_per_hour, reverse=True)
        
        # Provision cheapest A100
        create(gpus.gpu_type.contains("A100"))
        
        # Provision from top 5 cheapest RTX 4090s
        create(gpu_type="RTX 4090", max_attempts=5)
    """
    # Handle different input types
    if isinstance(query, GPUOffer):
        # Single offer provided
        suitable_offers = [query]
    elif isinstance(query, list):
        # List of offers provided
        if not query:
            raise ValueError("No GPU offers provided")
        suitable_offers = query
    else:
        # Query object or None - search for suitable offers
        suitable_offers = search(
            query=query,
            gpu_type=gpu_type,
            max_price_per_hour=max_price_per_hour,
            provider=provider,
            cuda_version=cuda_version,
            manufacturer=manufacturer,
            sort=sort,
            reverse=reverse
        )
        if not suitable_offers:
            raise ValueError("No GPUs found matching criteria")
    
    # Try to provision from the top offers
    last_error = None
    attempts_made = 0
    
    for offer in suitable_offers[:max_attempts]:
        attempts_made += 1
        logger.info(f"Provisioning attempt {attempts_made}/{max_attempts}: {offer.gpu_type} at ${offer.price_per_hour:.3f}/hr")
        
        try:
            # Create provision request using this offer
            # For RunPod, use the full GPU ID from raw_data, not the display name
            gpu_type_id = offer.gpu_type
            if offer.provider == "runpod" and offer.raw_data:
                gpu_type_id = offer.raw_data.get("id", offer.gpu_type)
            
            request = ProvisionRequest(
                gpu_type=gpu_type_id,
                gpu_count=offer.gpu_count,
                image=image,
                name=name,
                provider=offer.provider,
                spot_instance=offer.spot,
                exposed_ports=exposed_ports,
                enable_http_proxy=enable_http_proxy,
                start_jupyter=start_jupyter,
                jupyter_password=jupyter_password,
                manufacturer=manufacturer,
                **kwargs
            )
            
            # Provision using the appropriate provider
            if offer.provider == "runpod":
                instance = runpod.provision_instance(request, request.ssh_startup_script)
                if instance:
                    logger.info(f"✅ Successfully provisioned GPU instance: {instance.id}")
                    logger.info(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")  
                    logger.info(f"   Provider: {offer.provider}")
                    logger.info(f"   Expected price: ${offer.price_per_hour:.3f}/hr")
                    return instance
                else:
                    logger.warning(f"Provisioning returned None for {offer.gpu_type}")
                    last_error = "Provisioning returned None"
            else:
                raise ValueError(f"Unsupported provider: {offer.provider}")
                
        except Exception as e:
            logger.warning(f"Provisioning failed for {offer.gpu_type}: {e}")
            last_error = str(e)
            continue
    
    # All attempts failed
    logger.error(f"Failed to provision after {attempts_made} attempts")
    if last_error:
        logger.error(f"Last error: {last_error}")
    return None


def list_instances(provider: Optional[str] = None) -> List[GPUInstance]:
    """
    List all user's instances across providers
    
    Args:
        provider: Specific provider to list from (default: all providers)
    
    Returns:
        List of user's GPU instances
    """
    instances = []
    
    # Get instances from providers
    if provider is None or provider == "runpod":
        runpod_instances = runpod.list_instances()
        instances.extend(runpod_instances)
    
    return instances