"""
GPU Broker Client - Main interface for GPU operations
"""

import os
from typing import List, Optional, Union, Dict, Any
from .types import GPUOffer, GPUInstance, CloudType
from .query import GPUQuery, QueryType
from . import api


class GPUClient:
    """Main client for GPU broker operations
    
    Handles configuration for API keys, SSH keys, and provides
    all GPU search, provisioning, and management functionality.
    
    Examples:
        # Basic usage with environment variables
        client = GPUClient()
        
        # Explicit configuration
        client = GPUClient(
            ssh_key_path="~/.ssh/my_runpod_key",
            api_key="your-runpod-api-key"
        )
        
        # Use the client
        offers = client.search(client.gpu_type.contains("A100"))
        instance = client.create(offers[:3])
    """
    
    def __init__(
        self, 
        ssh_key_path: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize GPU broker client
        
        Args:
            ssh_key_path: Path to SSH private key, or None for auto-discovery
            api_key: RunPod API key, or None to use RUNPOD_API_KEY env var
        """
        self._ssh_key_path = None
        self._api_key = None
        
        # Configure SSH key
        if ssh_key_path:
            self.set_ssh_key_path(ssh_key_path)
        else:
            self._discover_ssh_key()
        
        # Configure API key
        if api_key:
            self._api_key = api_key
        else:
            self._api_key = os.environ.get('RUNPOD_API_KEY')
            if not self._api_key:
                raise ValueError(
                    "RunPod API key required. Set via:\n"
                    "  client = GPUBrokerClient(api_key='your-key')\n"
                    "  export RUNPOD_API_KEY=your-key"
                )
        
        # Set up query interface
        self._query = GPUQuery()
    
    def set_ssh_key_path(self, path: str):
        """Set SSH private key path
        
        Args:
            path: Path to SSH private key file
        """
        expanded_path = os.path.expanduser(path)
        if not os.path.exists(expanded_path):
            raise ValueError(f"SSH key not found: {expanded_path}")
        self._ssh_key_path = expanded_path
    
    def get_ssh_key_path(self) -> str:
        """Get configured SSH key path"""
        if not self._ssh_key_path:
            raise ValueError("No SSH key configured")
        return self._ssh_key_path
    
    def _discover_ssh_key(self):
        """Auto-discover SSH key from common locations"""
        # Check environment variable first
        env_key = os.environ.get('GPU_BROKER_SSH_KEY')
        if env_key:
            self.set_ssh_key_path(env_key)
            return
        
        # Try common key locations
        for key_name in ["id_ed25519", "id_rsa", "id_ecdsa"]:
            key_path = os.path.expanduser(f"~/.ssh/{key_name}")
            if os.path.exists(key_path):
                self._ssh_key_path = key_path
                return
        
        raise ValueError(
            "No SSH key found. Set one via:\n"
            "  client.set_ssh_key_path('~/.ssh/id_ed25519')\n"
            "  export GPU_BROKER_SSH_KEY=~/.ssh/id_ed25519\n"
            "  Or ensure you have a key at ~/.ssh/id_ed25519"
        )
    
    # Query interface - expose as properties
    @property
    def gpu_type(self):
        """Query by GPU type: client.gpu_type.contains('A100')"""
        return self._query.gpu_type
    
    @property 
    def price_per_hour(self):
        """Query by price: client.price_per_hour < 2.0"""
        return self._query.price_per_hour
    
    @property
    def memory_gb(self):
        """Query by memory: client.memory_gb > 24"""
        return self._query.memory_gb
    
    @property
    def cloud_type(self):
        """Query by cloud type: client.cloud_type == CloudType.SECURE"""
        return self._query.cloud_type
    
    @property
    def provider(self):
        """Query by provider: client.provider == 'runpod'"""
        return self._query.provider
    
    @property
    def cuda_version(self):
        """Query by CUDA version: client.cuda_version.contains('12.0')"""
        return self._query.cuda_version
    
    # Main API methods
    def search(
        self,
        query: Optional[QueryType] = None,
        # Legacy parameters  
        gpu_type: Optional[str] = None,
        max_price_per_hour: Optional[float] = None,
        provider: Optional[str] = None,
        cuda_version: Optional[str] = None,
        # Sorting
        sort: Optional[callable] = None,
        reverse: bool = False
    ) -> List[GPUOffer]:
        """Search for GPU offers
        
        Args:
            query: Pandas-style query (e.g., client.gpu_type.contains("A100"))
            sort: Sort function (e.g., lambda x: x.memory_gb / x.price_per_hour)
            reverse: Sort descending
            
        Returns:
            List of GPU offers
        """
        return api.search(
            query=query,
            gpu_type=gpu_type,
            max_price_per_hour=max_price_per_hour,
            provider=provider,
            cuda_version=cuda_version,
            sort=sort,
            reverse=reverse
        )
    
    def create(
        self,
        query: Union[QueryType, List[GPUOffer], GPUOffer] = None,
        image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        name: Optional[str] = None,
        # Search parameters
        gpu_type: Optional[str] = None,
        max_price_per_hour: Optional[float] = None,
        provider: Optional[str] = None,
        cuda_version: Optional[str] = None,
        sort: Optional[callable] = None,
        reverse: bool = False,
        # Retry parameters
        max_attempts: int = 3,
        **kwargs
    ) -> Optional['ClientGPUInstance']:
        """Create GPU instance
        
        Returns:
            GPU instance with client configuration
        """
        instance = api.create(
            query=query,
            image=image,
            name=name,
            gpu_type=gpu_type,
            max_price_per_hour=max_price_per_hour,
            provider=provider,
            cuda_version=cuda_version,
            sort=sort,
            reverse=reverse,
            max_attempts=max_attempts,
            **kwargs
        )
        
        if instance:
            # Return wrapped instance with client configuration
            return ClientGPUInstance(instance, self)
        return None
    
    def get_instance(self, instance_id: str, provider: Optional[str] = None) -> Optional['ClientGPUInstance']:
        """Get instance details"""
        instance = api.get_instance(instance_id, provider)
        if instance:
            return ClientGPUInstance(instance, self)
        return None
    
    def terminate_instance(self, instance_id: str, provider: Optional[str] = None) -> bool:
        """Terminate instance"""
        return api.terminate_instance(instance_id, provider)
    
    def list_instances(self, provider: Optional[str] = None) -> List['ClientGPUInstance']:
        """List all user's instances"""
        instances = api.list_instances(provider)
        return [ClientGPUInstance(instance, self) for instance in instances]


class ClientGPUInstance:
    """GPU instance with client configuration
    
    Wraps GPUInstance to use client's SSH key configuration
    """
    
    def __init__(self, instance: GPUInstance, client: GPUClient):
        self._instance = instance
        self._client = client
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped instance"""
        return getattr(self._instance, name)
    
    def exec(self, command: str, ssh_key_path: str = None, timeout: int = 30):
        """Execute command using client's SSH configuration"""
        if ssh_key_path is None:
            ssh_key_path = self._client.get_ssh_key_path()
        
        return self._instance.exec(command, ssh_key_path, timeout)
    
    def wait_until_ready(self, timeout: int = 300) -> bool:
        """Wait until instance is running"""
        return self._instance.wait_until_ready(timeout)
    
    def wait_until_ssh_ready(self, timeout: int = 300) -> bool:
        """Wait until SSH is ready"""
        return self._instance.wait_until_ssh_ready(timeout)
    
    def terminate(self) -> bool:
        """Terminate this instance"""
        return self._instance.terminate()