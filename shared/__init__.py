"""
Shared utilities and foundations for llm-workbench components.

This module contains common functionality that is used across
broker, bifrost, and other components to reduce code duplication
and ensure consistent behavior.
"""

__version__ = "0.1.0"

# Export main SSH foundation classes
from .ssh_foundation import SSHConnectionInfo, UniversalSSHClient, secure_temp_ssh_key

__all__ = [
    "SSHConnectionInfo",
    "UniversalSSHClient", 
    "secure_temp_ssh_key"
]