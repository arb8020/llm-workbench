"""
Provider registry for managing GPU cloud providers.

This module provides a centralized registry for all available GPU providers,
enabling dynamic provider discovery and management.
"""

from typing import Dict, List

from . import GPUProvider


class ProviderRegistry:
    """Central registry for GPU cloud providers."""
    
    def __init__(self):
        self._providers: Dict[str, GPUProvider] = {}
    
    def register_provider(self, name: str, provider: GPUProvider) -> None:
        """Register a new GPU provider.
        
        Args:
            name: Provider name (e.g., "runpod", "aws", "gcp")
            provider: Provider implementation
        """
        self._providers[name] = provider
    
    def get_provider(self, name: str) -> GPUProvider:
        """Get a specific provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider implementation
            
        Raises:
            ValueError: If provider is not registered
        """
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}. Available: {list(self._providers.keys())}")
        return self._providers[name]
    
    def get_all_providers(self) -> Dict[str, GPUProvider]:
        """Get all registered providers.
        
        Returns:
            Dict mapping provider names to implementations
        """
        return self._providers.copy()
    
    def list_provider_names(self) -> List[str]:
        """Get list of all registered provider names.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def is_provider_registered(self, name: str) -> bool:
        """Check if a provider is registered.
        
        Args:
            name: Provider name
            
        Returns:
            True if provider is registered
        """
        return name in self._providers


# Global registry instance
_global_registry = ProviderRegistry()


def register_provider(name: str, provider: GPUProvider) -> None:
    """Register a provider with the global registry.
    
    Args:
        name: Provider name
        provider: Provider implementation
    """
    _global_registry.register_provider(name, provider)


def get_provider(name: str) -> GPUProvider:
    """Get a provider from the global registry.
    
    Args:
        name: Provider name
        
    Returns:
        Provider implementation
    """
    return _global_registry.get_provider(name)


def get_all_providers() -> Dict[str, GPUProvider]:
    """Get all providers from the global registry.
    
    Returns:
        Dict mapping provider names to implementations
    """
    return _global_registry.get_all_providers()


def list_provider_names() -> List[str]:
    """Get list of all registered provider names.
    
    Returns:
        List of provider names
    """
    return _global_registry.list_provider_names()


def is_provider_registered(name: str) -> bool:
    """Check if a provider is registered.
    
    Args:
        name: Provider name
        
    Returns:
        True if provider is registered
    """
    return _global_registry.is_provider_registered(name)


# Auto-register RunPod provider
def _register_default_providers():
    """Register default providers on module import."""
    try:
        from . import runpod
        register_provider("runpod", runpod)
    except ImportError:
        pass  # RunPod provider not available


# Register default providers when module is imported
_register_default_providers()


__all__ = [
    "ProviderRegistry",
    "register_provider", 
    "get_provider",
    "get_all_providers",
    "list_provider_names",
    "is_provider_registered"
]