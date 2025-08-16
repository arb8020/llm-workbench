"""
GPU Broker Minimal - RunPod GPU provisioning with SSH output capture fix

Two usage patterns:

1. Client-based (recommended):
   client = GPUClient(ssh_key_path="~/.ssh/id_ed25519")
   offers = client.search(client.gpu_type.contains("A100"))
   instance = client.create(offers[:3])

2. Global interface (legacy):
   import broker as gpus
   offers = gpus.search(gpus.gpu_type.contains("A100"))
   instance = gpus.create(offers[:3])
"""

__version__ = "0.1.0"

# Main client interface
from .client import GPUClient

# Legacy global interface with default client
import os
_default_client = None

def _get_default_client():
    """Get or create default client instance"""
    global _default_client
    if _default_client is None:
        try:
            _default_client = GPUClient()
        except ValueError as e:
            raise ValueError(
                f"Default client setup failed: {e}\n\n"
                "Consider using the client interface:\n"
                "  client = GPUClient(ssh_key_path='~/.ssh/id_ed25519')\n"
                "  offers = client.search(...)"
            )
    return _default_client

# Legacy functions that use default client
def search(*args, **kwargs):
    """Search for GPU offers using default client"""
    return _get_default_client().search(*args, **kwargs)

def create(*args, **kwargs):
    """Create GPU instance using default client"""
    return _get_default_client().create(*args, **kwargs)

def get_instance(*args, **kwargs):
    """Get instance details using default client"""
    return _get_default_client().get_instance(*args, **kwargs)

def terminate_instance(*args, **kwargs):
    """Terminate instance using default client"""
    return _get_default_client().terminate_instance(*args, **kwargs)

def list_instances(*args, **kwargs):
    """List all instances using default client"""
    return _get_default_client().list_instances(*args, **kwargs)

def set_ssh_key_path(path: str):
    """Set SSH key path for default client"""
    _get_default_client().set_ssh_key_path(path)

def get_ssh_key_path() -> str:
    """Get SSH key path from default client"""
    return _get_default_client().get_ssh_key_path()

# Legacy query interface - simplified to avoid complex type inference
class _LegacyQueryProperty:
    def __init__(self, name: str):
        self.name = name
    
    # Explicitly define the methods that are actually used
    def contains(self, value: str):
        """For gpu_type.contains() usage"""
        return getattr(_get_default_client(), self.name).contains(value)
    
    def __lt__(self, other):
        """For price_per_hour < X usage"""  
        return getattr(_get_default_client(), self.name).__lt__(other)
    
    def __gt__(self, other):
        """For price_per_hour > X usage"""
        return getattr(_get_default_client(), self.name).__gt__(other)
    
    def __eq__(self, other):
        """For cloud_type == X usage"""
        return getattr(_get_default_client(), self.name).__eq__(other)
    
    def __le__(self, other):
        """For price_per_hour <= X usage"""
        return getattr(_get_default_client(), self.name).__le__(other)
    
    def __ge__(self, other):
        """For price_per_hour >= X usage"""
        return getattr(_get_default_client(), self.name).__ge__(other)

# Expose query fields at module level for backward compatibility  
gpu_type = _LegacyQueryProperty('gpu_type')
price_per_hour = _LegacyQueryProperty('price_per_hour')
memory_gb = _LegacyQueryProperty('memory_gb') 
cloud_type = _LegacyQueryProperty('cloud_type')
provider = _LegacyQueryProperty('provider')
cuda_version = _LegacyQueryProperty('cuda_version')

__all__ = [
    "GPUClient",  # Main client interface
    "search", "create", "get_instance", "terminate_instance", "list_instances", "set_ssh_key_path", "get_ssh_key_path",  # Legacy
    "gpu_type", "price_per_hour", "memory_gb", "cloud_type", "provider", "cuda_version"  # Legacy query
]