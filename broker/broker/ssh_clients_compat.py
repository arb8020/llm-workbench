"""
Compatibility layer for broker SSH clients using shared foundation.

This module provides the same interface as the old ssh_clients.py but delegates
to the shared SSH foundation. This allows existing code and tests to work
without modification while using the shared implementation.

All advanced features from the deprecated ssh_clients.py are now available:
- Streaming execution with callbacks
- Interactive SSH sessions  
- Connection testing utilities
"""

# Direct re-exports from shared foundation (simplified)
import sys
from pathlib import Path

# Add llm-workbench root to Python path for shared module access
_workbench_root = Path(__file__).parent.parent.parent
if str(_workbench_root) not in sys.path:
    sys.path.insert(0, str(_workbench_root))

# Import generic functions from shared foundation
from shared.ssh_foundation import (
    execute_command_sync as _execute_command_sync,
    execute_command_async as _execute_command_async, 
    execute_command_streaming as _execute_command_streaming,
    start_interactive_ssh_session as _start_interactive_ssh_session,
    test_ssh_connection as _test_ssh_connection
)

# Re-export core utilities (shared module already imported above)
from shared.ssh_foundation import (
    SSHConnectionInfo,
    UniversalSSHClient,
    secure_temp_ssh_key
)

# Re-export enums for backward compatibility (create simple ones)
from enum import Enum
from typing import Optional, Callable, Tuple

class SSHMethod(Enum):
    """SSH connection methods - only direct SSH supported"""
    DIRECT = "direct"

class SSHClient(Enum):
    """SSH client types"""
    PARAMIKO = "paramiko"
    ASYNCSSH = "asyncssh"


# Broker-specific conversion utilities
def _create_connection_info_from_gpu_instance(instance, private_key: Optional[str] = None, timeout: int = 30) -> SSHConnectionInfo:
    """Create SSHConnectionInfo from broker GPUInstance
    
    Args:
        instance: GPUInstance with SSH connection details
        private_key: Optional SSH private key content
        timeout: Connection timeout
        
    Returns:
        SSHConnectionInfo for the instance
        
    Raises:
        ValueError: If proxy SSH detected or SSH details unavailable
    """
    # Reject proxy SSH - only direct SSH supported
    if instance.public_ip == "ssh.runpod.io":
        raise ValueError("Direct SSH not available - only proxy SSH found")
        
    if not instance.public_ip or not instance.ssh_port or not instance.ssh_username:
        raise ValueError("Instance SSH details not available")
        
    return SSHConnectionInfo(
        hostname=instance.public_ip,
        port=instance.ssh_port,
        username=instance.ssh_username,
        key_content=private_key,
        timeout=timeout
    )


# Broker-compatible wrapper functions
def execute_command_sync(instance, private_key: Optional[str], command: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute command synchronously using broker GPUInstance"""
    try:
        conn_info = _create_connection_info_from_gpu_instance(instance, private_key, timeout)
        return _execute_command_sync(conn_info, command, timeout)
    except Exception as e:
        return -1, "", f"Broker sync execution failed: {e}"


async def execute_command_async(instance, private_key: Optional[str], command: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute command asynchronously using broker GPUInstance"""
    try:
        conn_info = _create_connection_info_from_gpu_instance(instance, private_key, timeout)
        return await _execute_command_async(conn_info, command, timeout)
    except Exception as e:
        return -1, "", f"Broker async execution failed: {e}"


def execute_command_streaming(instance, command: str, private_key: Optional[str] = None, timeout: int = 30,
                             output_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[int, str, str]:
    """Execute command with streaming output using broker GPUInstance"""
    try:
        conn_info = _create_connection_info_from_gpu_instance(instance, private_key, timeout)
        return _execute_command_streaming(conn_info, command, timeout, output_callback)
    except Exception as e:
        return -1, "", f"Broker streaming execution failed: {e}"


def start_interactive_ssh_session(instance, private_key_path: Optional[str] = None):
    """Start an interactive SSH session using broker GPUInstance
    
    Args:
        instance: GPU instance to connect to
        private_key_path: Optional private key path (if None, uses SSH agent/default keys)
    """
    try:
        conn_info = _create_connection_info_from_gpu_instance(instance)
        _start_interactive_ssh_session(conn_info, private_key_path)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Interactive SSH session failed: {e}")
        raise


def test_ssh_connection(instance, private_key: Optional[str] = None, test_both_clients: bool = True) -> Tuple[bool, str]:
    """Test SSH connection using broker GPUInstance
    
    Args:
        instance: GPU instance to test
        private_key: Optional private key content
        test_both_clients: Whether to test both sync and async clients
        
    Returns:
        Tuple of (success, message)
    """
    try:
        conn_info = _create_connection_info_from_gpu_instance(instance, private_key)
        return _test_ssh_connection(conn_info, test_both_clients)
    except Exception as e:
        return False, f"❌ Test setup failed: {e}"

__all__ = [
    # Core execution functions
    'execute_command_sync',
    'execute_command_async',
    
    # Advanced features
    'execute_command_streaming', 
    'start_interactive_ssh_session',
    'test_ssh_connection',
    
    # Core utilities
    'SSHConnectionInfo',
    'UniversalSSHClient', 
    'secure_temp_ssh_key',
    
    # Enums for compatibility
    'SSHMethod',
    'SSHClient'
]