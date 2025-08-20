"""
Provider interface definitions for GPU cloud operations.

This module defines the protocol that all GPU cloud providers must implement
to be compatible with the broker system.
"""

from typing import List, Optional, Protocol

from ..types import GPUInstance, GPUOffer, ProvisionRequest


class GPUProvider(Protocol):
    """Protocol defining the interface that all GPU providers must implement.
    
    This protocol ensures type safety and consistency across different
    cloud providers (RunPod, AWS, GCP, etc.).
    """
    
    def search_gpu_offers(self, cuda_version: Optional[str] = None) -> List[GPUOffer]:
        """Search for available GPU offers from this provider.
        
        Args:
            cuda_version: Optional CUDA version filter (e.g., "12.0", "11.8")
            
        Returns:
            List of available GPU offers
        """
        ...
    
    def provision_instance(
        self, 
        request: ProvisionRequest, 
        ssh_startup_script: Optional[str] = None
    ) -> Optional[GPUInstance]:
        """Provision a GPU instance from this provider.
        
        Args:
            request: Provisioning request with GPU specifications
            ssh_startup_script: Optional SSH setup script
            
        Returns:
            Provisioned GPU instance or None if failed
        """
        ...
    
    def get_instance_details(self, instance_id: str) -> Optional[GPUInstance]:
        """Get details of a specific instance.
        
        Args:
            instance_id: Unique instance identifier
            
        Returns:
            Instance details or None if not found
        """
        ...
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a GPU instance.
        
        Args:
            instance_id: Unique instance identifier
            
        Returns:
            True if termination was successful
        """
        ...
    
    def list_instances(self) -> List[GPUInstance]:
        """List all user's instances from this provider.
        
        Returns:
            List of user's GPU instances
        """
        ...


__all__ = ["GPUProvider"]