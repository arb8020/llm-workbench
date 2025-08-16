"""Inference backends package."""

from typing import Dict, Type, List, Optional
from .base import InferenceBackend, BackendConfig

# Backend registry
_BACKENDS: Dict[str, Type[InferenceBackend]] = {}


def register_backend(backend_class: Type[InferenceBackend]):
    """Register a backend implementation."""
    _BACKENDS[backend_class.backend_name] = backend_class
    return backend_class


def get_backend(name: str) -> Optional[Type[InferenceBackend]]:
    """Get backend class by name."""
    return _BACKENDS.get(name)


def list_backends() -> List[str]:
    """List available backend names."""
    return list(_BACKENDS.keys())


def create_backend(name: str, config: BackendConfig) -> InferenceBackend:
    """Create a backend instance by name."""
    backend_class = get_backend(name)
    if backend_class is None:
        raise ValueError(f"Unknown backend: {name}. Available: {list_backends()}")
    return backend_class(config)


# Auto-register available backends
try:
    from .vllm import VLLMBackend
    register_backend(VLLMBackend)
except ImportError as e:
    print(f"vLLM backend not available: {e}")

# Future backends
# try:
#     from .sglang import SGLangBackend
#     register_backend(SGLangBackend)
# except ImportError:
#     pass


# Main exports
__all__ = [
    "InferenceBackend", 
    "BackendConfig",
    "register_backend", 
    "get_backend", 
    "list_backends",
    "create_backend"
]