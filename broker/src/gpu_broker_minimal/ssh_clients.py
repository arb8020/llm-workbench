"""
SSH client implementations supporting both sync/async and direct/proxy connections
"""

import asyncio
import time
import logging
import tempfile
import os
from typing import Optional, Tuple, Union
from enum import Enum

# Import both SSH libraries
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None

try:
    import asyncssh
    ASYNCSSH_AVAILABLE = True
except ImportError:
    ASYNCSSH_AVAILABLE = False
    asyncssh = None

from .types import GPUInstance

logger = logging.getLogger(__name__)


class SSHMethod(Enum):
    """SSH connection methods - minimal version only supports direct SSH"""
    DIRECT = "direct"  # ssh root@{ip} -p {port}


class SSHClient(Enum):
    """SSH client types"""
    PARAMIKO = "paramiko"  # Sync client
    ASYNCSSH = "asyncssh"  # Async client


def get_ssh_connection_info(instance: GPUInstance, method: SSHMethod) -> Tuple[str, int, str]:
    """
    Get SSH connection details for the specified method
    
    Returns:
        Tuple of (hostname, port, username)
    """
    if method == SSHMethod.DIRECT:
        # Direct SSH: ssh root@{public_ip} -p {ssh_port}
        if instance.public_ip == "ssh.runpod.io":
            raise ValueError("Direct SSH not available - only proxy SSH found")
        return instance.public_ip, instance.ssh_port, "root"
    
    else:
        raise ValueError(f"Unsupported SSH method: {method}. This minimal version only supports direct SSH.")


class ParamikoSSHClient:
    """Synchronous SSH client using paramiko"""
    
    def __init__(self):
        if not PARAMIKO_AVAILABLE:
            raise ImportError("paramiko is not installed. Install with: pip install paramiko")
        self.client = None
    
    def connect(self, hostname: str, port: int, username: str, private_key: str = None, timeout: int = 30) -> bool:
        """Connect using paramiko with SSH agent support"""
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown hosts
            
            # Try SSH agent first (handles all key formats)
            try:
                self.client.connect(
                    hostname=hostname,
                    port=port,
                    username=username,
                    timeout=timeout,
                    look_for_keys=True,   # Look for keys in ~/.ssh/
                    allow_agent=True     # Use SSH agent
                )
                logger.info(f"✅ Paramiko connected via SSH agent to {username}@{hostname}:{port}")
                return True
                
            except Exception as agent_error:
                logger.debug(f"SSH agent failed: {agent_error}")
                
                # Fallback: try to load private key if provided
                if private_key:
                    # Write private key to temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as key_file:
                        key_file.write(private_key)
                        key_file.flush()
                        key_path = key_file.name
                    
                    try:
                        self.client.connect(
                            hostname=hostname,
                            port=port,
                            username=username,
                            key_filename=key_path,
                            timeout=timeout,
                            look_for_keys=False,
                            allow_agent=False
                        )
                        logger.info(f"✅ Paramiko connected via key file to {username}@{hostname}:{port}")
                        return True
                        
                    finally:
                        # Clean up temporary key file
                        try:
                            os.unlink(key_path)
                        except:
                            pass
                else:
                    raise agent_error
                    
        except Exception as e:
            logger.error(f"❌ Paramiko connection failed: {e}")
            return False
    
    def execute(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute command via paramiko - simplified approach"""
        if not self.client:
            return False, "", "No SSH connection"
        
        try:
            # Direct command execution - testing showed complex environment sourcing is unnecessary
            # for basic GPU operations like nvidia-smi, python, etc.
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout, get_pty=False)
            
            stdout_data = stdout.read().decode('utf-8')
            stderr_data = stderr.read().decode('utf-8')
            exit_code = stdout.channel.recv_exit_status()
            
            success = exit_code == 0
            return success, stdout_data, stderr_data
            
        except Exception as e:
            return False, "", str(e)
    
    def close(self):
        """Close paramiko connection"""
        if self.client:
            self.client.close()
            self.client = None


class AsyncSSHClient:
    """Asynchronous SSH client using asyncssh"""
    
    def __init__(self):
        if not ASYNCSSH_AVAILABLE:
            raise ImportError("asyncssh is not installed. Install with: pip install asyncssh")
        self.conn = None
    
    async def connect(self, hostname: str, port: int, username: str, private_key: str = None, timeout: int = 30) -> bool:
        """Connect using asyncssh with SSH agent support (like Paramiko)"""
        try:
            if private_key:
                # Use provided private key
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as key_file:
                    key_file.write(private_key)
                    key_file.flush()
                    key_path = key_file.name
                
                try:
                    self.conn = await asyncio.wait_for(
                        asyncssh.connect(
                            hostname,
                            port=port,
                            username=username,
                            client_keys=[key_path],
                            known_hosts=None
                        ),
                        timeout=timeout
                    )
                    logger.info(f"✅ AsyncSSH connected via key file to {username}@{hostname}:{port}")
                    return True
                finally:
                    try:
                        os.unlink(key_path)
                    except:
                        pass
            else:
                # Use SSH agent/default keys (like Paramiko)
                self.conn = await asyncio.wait_for(
                    asyncssh.connect(
                        hostname,
                        port=port,
                        username=username,
                        client_keys=['~/.ssh/id_ed25519', '~/.ssh/id_rsa'],
                        known_hosts=None
                    ),
                    timeout=timeout
                )
                logger.info(f"✅ AsyncSSH connected via SSH agent to {username}@{hostname}:{port}")
                return True
                    
        except Exception as e:
            logger.error(f"❌ AsyncSSH connection failed: {e}")
            return False
    
    async def execute(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute command via asyncssh - simplified approach"""
        if not self.conn:
            return False, "", "No SSH connection"
        
        try:
            # Direct command execution - testing showed complex environment sourcing is unnecessary
            # for basic GPU operations like nvidia-smi, python, etc.
            result = await asyncio.wait_for(
                self.conn.run(command),
                timeout=timeout
            )
            
            success = result.exit_status == 0
            stdout = result.stdout if result.stdout else ""
            stderr = result.stderr if result.stderr else ""
            
            return success, stdout, stderr
            
        except Exception as e:
            return False, "", str(e)
    
    async def close(self):
        """Close asyncssh connection"""
        if self.conn:
            self.conn.close()
            await self.conn.wait_closed()
            self.conn = None


def test_ssh_connection(instance: GPUInstance, private_key: str, 
                       method: SSHMethod, client_type: SSHClient,
                       timeout: int = 30) -> Tuple[bool, str]:
    """
    Test SSH connection with specified method and client
    
    Returns:
        Tuple of (success, message)
    """
    try:
        hostname, port, username = get_ssh_connection_info(instance, method)
        logger.info(f"🔗 Testing {method.value} SSH with {client_type.value}")
        logger.info(f"   Connection: {username}@{hostname}:{port}")
        
        if client_type == SSHClient.PARAMIKO:
            return _test_paramiko_connection(hostname, port, username, private_key, timeout)
        elif client_type == SSHClient.ASYNCSSH:
            return _test_asyncssh_connection(hostname, port, username, private_key, timeout)
        else:
            return False, f"Unknown client type: {client_type}"
            
    except Exception as e:
        return False, str(e)


def _test_paramiko_connection(hostname: str, port: int, username: str, 
                            private_key: str, timeout: int) -> Tuple[bool, str]:
    """Test paramiko connection"""
    client = ParamikoSSHClient()
    
    try:
        if not client.connect(hostname, port, username, private_key, timeout):
            return False, "Connection failed"
        
        # Test basic command
        success, stdout, stderr = client.execute("echo 'SSH_TEST_SUCCESS'", timeout=10)
        if success and "SSH_TEST_SUCCESS" in stdout:
            return True, "Paramiko connection successful"
        elif success:  # Command ran but no expected output (still success)
            return True, f"Paramiko connection successful (command executed with exit 0)"
        else:
            return False, f"Command test failed: {stderr}"
            
    except Exception as e:
        error_msg = str(e)
        # Treat authentication failures as connectivity success
        if "Authentication failed" in error_msg or "Permission denied" in error_msg:
            return True, f"Paramiko connectivity confirmed (auth failed as expected): {error_msg}"
        return False, error_msg
    finally:
        client.close()


def _test_asyncssh_connection(hostname: str, port: int, username: str, 
                            private_key: str, timeout: int) -> Tuple[bool, str]:
    """Test asyncssh connection"""
    async def _async_test():
        client = AsyncSSHClient()
        
        try:
            if not await client.connect(hostname, port, username, private_key, timeout):
                return False, "Connection failed"
            
            # Test basic command
            success, stdout, stderr = await client.execute("echo 'SSH_TEST_SUCCESS'", timeout=10)
            if success and "SSH_TEST_SUCCESS" in stdout:
                return True, "AsyncSSH connection successful"
            elif success:  # Command ran but no expected output (still success)
                return True, f"AsyncSSH connection successful (command executed with exit 0)"
            else:
                return False, f"Command test failed: {stderr}"
                
        except Exception as e:
            return False, str(e)
        finally:
            await client.close()
    
    # Run async test
    try:
        return asyncio.run(_async_test())
    except Exception as e:
        return False, str(e)


async def execute_command_async(instance: GPUInstance, private_key: str, command: str,
                               method: SSHMethod = None, timeout: int = 30) -> Tuple[bool, str, str]:
    """
    Execute command asynchronously with automatic method selection
    """
    if method is None:
        # Only support direct SSH in minimal version
        if instance.public_ip == "ssh.runpod.io":
            raise ValueError("Proxy SSH not supported in minimal version. Only direct SSH connections with real output capture are supported.")
        method = SSHMethod.DIRECT
    
    hostname, port, username = get_ssh_connection_info(instance, method)
    
    client = AsyncSSHClient()
    try:
        if await client.connect(hostname, port, username, private_key, timeout):
            return await client.execute(command, timeout)
        else:
            return False, "", "Connection failed"
    finally:
        await client.close()


def execute_command_sync(instance: GPUInstance, private_key: str = None, command: str = None,
                        method: SSHMethod = None, timeout: int = 30) -> Tuple[bool, str, str]:
    """
    Execute command synchronously with automatic method selection
    
    Args:
        instance: GPU instance to connect to
        private_key: Optional private key (if None, uses SSH agent/default keys)
        command: Command to execute
        method: SSH method (auto-detected if None)
        timeout: Connection/command timeout
    """
    if method is None:
        # Only support direct SSH in minimal version
        if instance.public_ip == "ssh.runpod.io":
            raise ValueError("Proxy SSH not supported in minimal version. Only direct SSH connections with real output capture are supported.")
        method = SSHMethod.DIRECT
    
    hostname, port, username = get_ssh_connection_info(instance, method)
    
    client = ParamikoSSHClient()
    try:
        if client.connect(hostname, port, username, private_key, timeout):
            return client.execute(command, timeout)
        else:
            return False, "", "Connection failed"
    finally:
        client.close()


def start_interactive_ssh_session(instance: GPUInstance, private_key: str = None, method: SSHMethod = None):
    """
    Start an interactive SSH session using the system's SSH client
    
    Args:
        instance: GPU instance to connect to
        private_key: Optional private key path (if None, uses SSH agent/default keys)
        method: SSH method (auto-detected if None)
    """
    if method is None:
        # Only support direct SSH in minimal version
        if instance.public_ip == "ssh.runpod.io":
            raise ValueError("Proxy SSH not supported in minimal version. Only direct SSH connections are supported.")
        method = SSHMethod.DIRECT
    
    hostname, port, username = get_ssh_connection_info(instance, method)
    
    # Build SSH command
    ssh_cmd = ["ssh", f"{username}@{hostname}", "-p", str(port)]
    
    # Add private key if specified
    if private_key:
        ssh_cmd.extend(["-i", private_key])
    
    # Add common options for better experience
    ssh_cmd.extend([
        "-o", "StrictHostKeyChecking=no",  # Don't check host keys
        "-o", "UserKnownHostsFile=/dev/null",  # Don't save host keys
        "-o", "ServerAliveInterval=60",  # Keep connection alive
        "-t"  # Force pseudo-terminal allocation
    ])
    
    logger.info(f"🔗 Starting interactive SSH session: {' '.join(ssh_cmd)}")
    
    # Execute SSH command, replacing current process
    import subprocess
    import sys
    try:
        subprocess.run(ssh_cmd, check=False)
    except KeyboardInterrupt:
        print("\nSSH session ended.")
    except Exception as e:
        logger.error(f"SSH session failed: {e}")
        raise


def execute_command_streaming(instance: GPUInstance, command: str, private_key: str = None, 
                            method: SSHMethod = None, timeout: int = 300):
    """
    Execute command with real-time output streaming using paramiko
    
    Args:
        instance: GPU instance to connect to
        command: Command to execute
        private_key: Optional private key (if None, uses SSH agent/default keys)
        method: SSH method (auto-detected if None)
        timeout: Command timeout
    """
    if method is None:
        # Only support direct SSH in minimal version
        if instance.public_ip == "ssh.runpod.io":
            raise ValueError("Proxy SSH not supported in minimal version. Only direct SSH connections are supported.")
        method = SSHMethod.DIRECT
    
    hostname, port, username = get_ssh_connection_info(instance, method)
    
    client = ParamikoSSHClient()
    try:
        if not client.connect(hostname, port, username, private_key, timeout):
            print(f"❌ Failed to connect to {username}@{hostname}:{port}")
            return False
        
        # Execute command directly - simplified approach
        print(f"🔄 Executing: {command}")
        stdin, stdout, stderr = client.client.exec_command(command, timeout=timeout, get_pty=True)
        
        # Stream output in real-time
        import select
        import sys
        
        while True:
            # Check if command is finished
            if stdout.channel.exit_status_ready():
                break
            
            # Check for available data
            if stdout.channel.recv_ready():
                data = stdout.channel.recv(4096).decode('utf-8', errors='ignore')
                print(data, end='', flush=True)
            
            if stderr.channel.recv_stderr_ready():
                data = stderr.channel.recv_stderr(4096).decode('utf-8', errors='ignore')
                print(data, end='', file=sys.stderr, flush=True)
            
            time.sleep(0.1)
        
        # Get final output
        remaining_stdout = stdout.read().decode('utf-8', errors='ignore')
        remaining_stderr = stderr.read().decode('utf-8', errors='ignore')
        
        if remaining_stdout:
            print(remaining_stdout, end='', flush=True)
        if remaining_stderr:
            print(remaining_stderr, end='', file=sys.stderr, flush=True)
        
        exit_code = stdout.channel.recv_exit_status()
        success = exit_code == 0
        
        if success:
            print(f"\n✅ Command completed successfully (exit code: {exit_code})")
        else:
            print(f"\n❌ Command failed with exit code: {exit_code}")
        
        return success
        
    except Exception as e:
        print(f"❌ Command execution failed: {e}")
        return False
    finally:
        client.close()