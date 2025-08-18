"""
GPU Broker Minimal - Clean architecture with GPUClient-first design
"""

__version__ = "0.1.0"

# Main client interface - the only public API
from .client import GPUClient

__all__ = ["GPUClient"]