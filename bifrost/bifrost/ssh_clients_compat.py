"""
Bifrost SSH compatibility layer using shared SSH foundation.

This module provides bifrost-specific wrappers around the shared SSH foundation,
allowing bifrost to leverage the same SSH implementation as broker while
maintaining its deployment-focused interface.
"""

import sys
from pathlib import Path
from typing import Optional, Callable, Tuple

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

# Re-export enums for backward compatibility
from enum import Enum

class SSHMethod(Enum):
    """SSH connection methods - only direct SSH supported"""
    DIRECT = "direct"

class SSHClient(Enum):
    """SSH client types"""
    PARAMIKO = "paramiko"
    ASYNCSSH = "asyncssh"


# Bifrost-specific conversion utilities
def _create_connection_info_from_ssh_connection(ssh_conn, private_key: Optional[str] = None, timeout: int = 30) -> SSHConnectionInfo:
    """Create SSHConnectionInfo from bifrost SSHConnection
    
    Args:
        ssh_conn: Bifrost SSHConnection object with host, user, port
        private_key: Optional SSH private key content
        timeout: Connection timeout
        
    Returns:
        SSHConnectionInfo for the connection
    """
    return SSHConnectionInfo(
        hostname=ssh_conn.host,
        port=ssh_conn.port,
        username=ssh_conn.user,
        key_content=private_key,
        timeout=timeout
    )


def _create_connection_info_from_string(ssh_string: str, private_key: Optional[str] = None, timeout: int = 30) -> SSHConnectionInfo:
    """Create SSHConnectionInfo from SSH connection string
    
    Args:
        ssh_string: SSH string like 'user@host:port'
        private_key: Optional SSH private key content
        timeout: Connection timeout
        
    Returns:
        SSHConnectionInfo for the connection
    """
    conn_info = SSHConnectionInfo.from_string(ssh_string, timeout=timeout)
    if private_key:
        # Update with key content
        conn_info = SSHConnectionInfo(
            hostname=conn_info.hostname,
            port=conn_info.port,
            username=conn_info.username,
            key_content=private_key,
            timeout=timeout
        )
    return conn_info


# Bifrost-compatible wrapper functions
def execute_command_sync(ssh_conn_or_string, command: str, private_key: Optional[str] = None, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute command synchronously using bifrost SSH connection
    
    Args:
        ssh_conn_or_string: Either SSHConnection object or connection string like 'user@host:port'
        command: Command to execute
        private_key: Optional SSH private key content
        timeout: Command timeout
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        if isinstance(ssh_conn_or_string, str):
            conn_info = _create_connection_info_from_string(ssh_conn_or_string, private_key, timeout)
        else:
            conn_info = _create_connection_info_from_ssh_connection(ssh_conn_or_string, private_key, timeout)
        
        return _execute_command_sync(conn_info, command, timeout)
    except Exception as e:
        return -1, "", f"Bifrost sync execution failed: {e}"


async def execute_command_async(ssh_conn_or_string, command: str, private_key: Optional[str] = None, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute command asynchronously using bifrost SSH connection
    
    Args:
        ssh_conn_or_string: Either SSHConnection object or connection string like 'user@host:port'
        command: Command to execute
        private_key: Optional SSH private key content
        timeout: Command timeout
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        if isinstance(ssh_conn_or_string, str):
            conn_info = _create_connection_info_from_string(ssh_conn_or_string, private_key, timeout)
        else:
            conn_info = _create_connection_info_from_ssh_connection(ssh_conn_or_string, private_key, timeout)
        
        return await _execute_command_async(conn_info, command, timeout)
    except Exception as e:
        return -1, "", f"Bifrost async execution failed: {e}"


def execute_command_streaming(ssh_conn_or_string, command: str, private_key: Optional[str] = None, timeout: int = 30,
                             output_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[int, str, str]:
    """Execute command with streaming output using bifrost SSH connection
    
    Args:
        ssh_conn_or_string: Either SSHConnection object or connection string like 'user@host:port'
        command: Command to execute
        private_key: Optional SSH private key content
        timeout: Command timeout
        output_callback: Optional callback for real-time output
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        if isinstance(ssh_conn_or_string, str):
            conn_info = _create_connection_info_from_string(ssh_conn_or_string, private_key, timeout)
        else:
            conn_info = _create_connection_info_from_ssh_connection(ssh_conn_or_string, private_key, timeout)
        
        return _execute_command_streaming(conn_info, command, timeout, output_callback)
    except Exception as e:
        return -1, "", f"Bifrost streaming execution failed: {e}"


def start_interactive_ssh_session(ssh_conn_or_string, private_key_path: Optional[str] = None):
    """Start an interactive SSH session using bifrost SSH connection
    
    Args:
        ssh_conn_or_string: Either SSHConnection object or connection string like 'user@host:port'
        private_key_path: Optional private key path (if None, uses SSH agent/default keys)
    """
    try:
        if isinstance(ssh_conn_or_string, str):
            conn_info = _create_connection_info_from_string(ssh_conn_or_string)
        else:
            conn_info = _create_connection_info_from_ssh_connection(ssh_conn_or_string)
        
        _start_interactive_ssh_session(conn_info, private_key_path)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Interactive SSH session failed: {e}")
        raise


def test_ssh_connection(ssh_conn_or_string, private_key: Optional[str] = None, test_both_clients: bool = True) -> Tuple[bool, str]:
    """Test SSH connection using bifrost SSH connection
    
    Args:
        ssh_conn_or_string: Either SSHConnection object or connection string like 'user@host:port'
        private_key: Optional SSH private key content
        test_both_clients: Whether to test both sync and async clients
        
    Returns:
        Tuple of (success, message)
    """
    try:
        if isinstance(ssh_conn_or_string, str):
            conn_info = _create_connection_info_from_string(ssh_conn_or_string, private_key)
        else:
            conn_info = _create_connection_info_from_ssh_connection(ssh_conn_or_string, private_key)
        
        return _test_ssh_connection(conn_info, test_both_clients)
    except Exception as e:
        return False, f"❌ Test setup failed: {e}"


__all__ = [
    # Core execution functions
    'execute_command_sync',
    'execute_command_async', 
    'execute_command_streaming',
    'start_interactive_ssh_session',
    'test_ssh_connection',
    
    # Shared foundation utilities
    'SSHConnectionInfo',
    'UniversalSSHClient',
    'secure_temp_ssh_key',
    
    # Enums
    'SSHMethod',
    'SSHClient',
]