"""
Universal SSH foundation for llm-workbench components.

Provides secure, unified SSH connectivity that both broker and bifrost
can build upon, eliminating code duplication and ensuring consistent
security practices.
"""

import asyncio
import logging
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


@contextmanager
def secure_temp_ssh_key(private_key_content: str) -> Generator[str, None, None]:
    """
    Secure context manager for temporary SSH private key files.
    
    Creates a temporary file with secure permissions (600) and ensures
    cleanup even if exceptions occur.
    
    Args:
        private_key_content: SSH private key content as string
        
    Yields:
        str: Path to the temporary key file
        
    Security features:
    - File created with 600 permissions (owner read/write only)
    - Guaranteed cleanup via context manager
    - Exception-safe cleanup
    """
    temp_key_path = None
    try:
        # Create temporary file with secure permissions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as key_file:
            key_file.write(private_key_content)
            key_file.flush()
            temp_key_path = key_file.name
        
        # Set secure permissions (owner read/write only)
        os.chmod(temp_key_path, stat.S_IRUSR | stat.S_IWUSR)  # 600
        
        logger.debug(f"Created secure temporary SSH key: {temp_key_path}")
        yield temp_key_path
        
    except Exception as e:
        logger.error(f"Error in secure_temp_ssh_key: {e}")
        raise
    finally:
        # Guaranteed cleanup
        if temp_key_path and os.path.exists(temp_key_path):
            try:
                os.unlink(temp_key_path)
                logger.debug(f"Cleaned up temporary SSH key: {temp_key_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary SSH key {temp_key_path}: {cleanup_error}")


@dataclass
class SSHConnectionInfo:
    """Universal SSH connection information that works for both broker and bifrost"""
    hostname: str
    port: int
    username: str
    key_content: Optional[str] = None
    key_path: Optional[str] = None
    timeout: int = 30
    
    @classmethod
    def from_string(cls, conn_str: str, key_path: Optional[str] = None, 
                   timeout: int = 30) -> 'SSHConnectionInfo':
        """Parse SSH connection string in multiple formats
        
        Supports:
        - 'user@host:port' format
        - 'ssh -p port user@host' format (standard SSH command)
        
        Args:
            conn_str: Connection string in supported format
            key_path: Optional path to SSH private key file
            timeout: Connection timeout in seconds
            
        Returns:
            SSHConnectionInfo instance
            
        Raises:
            ValueError: If connection string format is invalid
        """
        # Handle standard SSH command format: "ssh -p port user@host"
        if conn_str.startswith('ssh '):
            return cls._parse_ssh_command(conn_str, key_path, timeout)
        
        # Handle user@host:port format
        if '@' not in conn_str or ':' not in conn_str:
            raise ValueError(f"Invalid SSH format: {conn_str}. Expected: user@host:port or ssh -p port user@host")
        
        user_host, port_str = conn_str.rsplit(':', 1)
        username, hostname = user_host.split('@', 1)
        
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port number: {port_str}")
        
        return cls(
            hostname=hostname,
            port=port,
            username=username,
            key_path=key_path,
            timeout=timeout
        )
    
    @classmethod
    def _parse_ssh_command(cls, ssh_cmd: str, key_path: Optional[str] = None, 
                          timeout: int = 30) -> 'SSHConnectionInfo':
        """Parse SSH command format: 'ssh -p port user@host'"""
        import shlex
        
        try:
            parts = shlex.split(ssh_cmd)
        except ValueError as e:
            raise ValueError(f"Failed to parse SSH command: {e}")
        
        if not parts or parts[0] != 'ssh':
            raise ValueError(f"Invalid SSH command: {ssh_cmd}")
        
        port = 22  # default SSH port
        user_host = None
        
        # Parse SSH command arguments
        i = 1
        while i < len(parts):
            if parts[i] == '-p' and i + 1 < len(parts):
                try:
                    port = int(parts[i + 1])
                    i += 2
                except ValueError:
                    raise ValueError(f"Invalid port in SSH command: {parts[i + 1]}")
            elif '@' in parts[i]:
                user_host = parts[i]
                break
            else:
                i += 1
        
        if not user_host or '@' not in user_host:
            raise ValueError(f"No user@host found in SSH command: {ssh_cmd}")
        
        username, hostname = user_host.split('@', 1)
        
        return cls(
            hostname=hostname,
            port=port,
            username=username,
            key_path=key_path,
            timeout=timeout
        )
    
    
    def load_key_content(self) -> Optional[str]:
        """Load SSH private key content from file path if specified"""
        if not self.key_path:
            return self.key_content
            
        key_path = os.path.expanduser(self.key_path)
        try:
            with open(key_path) as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load SSH key from {key_path}: {e}") from e
    
    def connection_string(self) -> str:
        """Get SSH connection string in user@host:port format"""
        return f"{self.username}@{self.hostname}:{self.port}"


class SSHClientProtocol(Protocol):
    """Protocol for SSH client implementations"""
    
    def connect(self, conn_info: SSHConnectionInfo) -> bool:
        """Establish SSH connection"""
        ...
    
    def exec_command(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute command and return (success, stdout, stderr)"""
        ...
    
    def close(self) -> None:
        """Close connection"""
        ...


class AsyncSSHClientProtocol(Protocol):
    """Protocol for async SSH client implementations"""
    
    async def connect(self, conn_info: SSHConnectionInfo) -> bool:
        """Establish SSH connection"""
        ...
    
    async def exec_command(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute command and return (success, stdout, stderr)"""
        ...
    
    async def close(self) -> None:
        """Close connection"""
        ...


class UniversalSSHClient:
    """
    Universal SSH client that provides both sync and async execution.
    
    Uses Paramiko for synchronous operations and AsyncSSH for asynchronous
    operations, with shared secure key handling and connection management.
    """
    
    def __init__(self):
        self._paramiko_client: Optional[Any] = None
        self._asyncssh_client: Optional[Any] = None
    
    def connect(self, conn_info: SSHConnectionInfo) -> bool:
        """Establish synchronous SSH connection using Paramiko"""
        try:
            import paramiko
        except ImportError as e:
            raise ImportError("paramiko is required for synchronous SSH. Install with: pip install paramiko") from e
        
        try:
            self._paramiko_client = paramiko.SSHClient()
            self._paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            key_content = conn_info.load_key_content()
            
            # Try SSH agent first (handles all key formats)
            try:
                self._paramiko_client.connect(
                    hostname=conn_info.hostname,
                    port=conn_info.port,
                    username=conn_info.username,
                    timeout=conn_info.timeout,
                    look_for_keys=True,
                    allow_agent=True
                )
                logger.info(f"✅ Connected via SSH agent to {conn_info.connection_string()}")
                return True
                
            except Exception as agent_error:
                logger.debug(f"SSH agent failed: {agent_error}")
                
                # Fallback: try private key if provided
                if key_content:
                    with secure_temp_ssh_key(key_content) as key_path:
                        self._paramiko_client.connect(
                            hostname=conn_info.hostname,
                            port=conn_info.port,
                            username=conn_info.username,
                            key_filename=key_path,
                            timeout=conn_info.timeout,
                            look_for_keys=False,
                            allow_agent=False
                        )
                        logger.info(f"✅ Connected via key file to {conn_info.connection_string()}")
                        return True
                else:
                    raise agent_error
                    
        except Exception as e:
            logger.error(f"❌ SSH connection failed: {e}")
            return False
    
    async def aconnect(self, conn_info: SSHConnectionInfo) -> bool:
        """Establish asynchronous SSH connection using AsyncSSH"""
        try:
            import asyncssh
        except ImportError as e:
            raise ImportError("asyncssh is required for asynchronous SSH. Install with: pip install asyncssh") from e
        
        try:
            key_content = conn_info.load_key_content()
            
            # Try SSH agent first
            try:
                self._asyncssh_client = await asyncio.wait_for(
                    asyncssh.connect(
                        conn_info.hostname,
                        port=conn_info.port,
                        username=conn_info.username,
                        known_hosts=None
                    ),
                    timeout=conn_info.timeout
                )
                logger.info(f"✅ AsyncSSH connected via SSH agent to {conn_info.connection_string()}")
                return True
                
            except Exception as agent_error:
                logger.debug(f"SSH agent failed: {agent_error}")
                
                # Fallback: try private key if provided
                if key_content:
                    with secure_temp_ssh_key(key_content) as key_path:
                        self._asyncssh_client = await asyncio.wait_for(
                            asyncssh.connect(
                                conn_info.hostname,
                                port=conn_info.port,
                                username=conn_info.username,
                                client_keys=[key_path],
                                known_hosts=None
                            ),
                            timeout=conn_info.timeout
                        )
                        logger.info(f"✅ AsyncSSH connected via key file to {conn_info.connection_string()}")
                        return True
                else:
                    raise agent_error
                    
        except Exception as e:
            logger.error(f"❌ AsyncSSH connection failed: {e}")
            return False
    
    def exec_command(self, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        """Execute command synchronously using Paramiko"""
        if not self._paramiko_client:
            return -1, "", "No SSH connection established"
        
        try:
            stdin, stdout, stderr = self._paramiko_client.exec_command(command, timeout=timeout, get_pty=False)
            
            stdout_data = stdout.read().decode('utf-8')
            stderr_data = stderr.read().decode('utf-8')
            exit_code = stdout.channel.recv_exit_status()
            
            return exit_code, stdout_data, stderr_data
            
        except Exception as e:
            return -1, "", str(e)
    
    def exec_command_streaming(self, command: str, timeout: int = 30, 
                              output_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[int, str, str]:
        """Execute command with real-time output streaming using Paramiko"""
        if not self._paramiko_client:
            return -1, "", "No SSH connection established"
        
        try:
            stdin, stdout, stderr = self._paramiko_client.exec_command(command, timeout=timeout, get_pty=True)
            
            stdout_lines = []
            stderr_lines = []
            
            # Read output line by line for streaming
            while not stdout.channel.exit_status_ready() or stdout.channel.recv_ready() or stderr.channel.recv_stderr_ready():
                if stdout.channel.recv_ready():
                    line = stdout.readline().rstrip('\n\r')
                    if line:
                        stdout_lines.append(line)
                        if output_callback:
                            output_callback(line, False)  # False = stdout
                
                if stderr.channel.recv_stderr_ready():
                    line = stderr.readline().rstrip('\n\r')
                    if line:
                        stderr_lines.append(line)
                        if output_callback:
                            output_callback(line, True)   # True = stderr
                
                import time
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            # Get final exit status
            exit_code = stdout.channel.recv_exit_status()
            
            return exit_code, '\n'.join(stdout_lines), '\n'.join(stderr_lines)
            
        except Exception as e:
            return -1, "", str(e)
    
    async def aexec_command(self, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        """Execute command asynchronously using AsyncSSH"""
        if not self._asyncssh_client:
            return -1, "", "No async SSH connection established"
        
        try:
            result = await asyncio.wait_for(
                self._asyncssh_client.run(command),
                timeout=timeout
            )
            
            return result.exit_status, result.stdout, result.stderr
            
        except Exception as e:
            return -1, "", str(e)
    
    def close(self) -> None:
        """Close synchronous SSH connection"""
        if self._paramiko_client:
            self._paramiko_client.close()
            self._paramiko_client = None
    
    async def aclose(self) -> None:
        """Close asynchronous SSH connection"""
        if self._asyncssh_client:
            self._asyncssh_client.close()
            await self._asyncssh_client.wait_closed()
            self._asyncssh_client = None


# Convenience functions for backward compatibility
def execute_command_sync(conn_info: SSHConnectionInfo, command: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute SSH command synchronously (convenience function)"""
    client = UniversalSSHClient()
    try:
        if client.connect(conn_info):
            return client.exec_command(command, timeout)
        else:
            return -1, "", "SSH connection failed"
    finally:
        client.close()


async def execute_command_async(conn_info: SSHConnectionInfo, command: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute SSH command asynchronously (convenience function)"""
    client = UniversalSSHClient()
    try:
        if await client.aconnect(conn_info):
            return await client.aexec_command(command, timeout)
        else:
            return -1, "", "SSH connection failed"
    finally:
        await client.aclose()




def execute_command_streaming(conn_info: SSHConnectionInfo, command: str, timeout: int = 30,
                             output_callback: Optional[Callable[[str, bool], None]] = None) -> Tuple[int, str, str]:
    """Execute SSH command with real-time streaming output (convenience function)"""
    client = UniversalSSHClient()
    try:
        if client.connect(conn_info):
            return client.exec_command_streaming(command, timeout, output_callback)
        else:
            return -1, "", "SSH connection failed"
    finally:
        client.close()




def start_interactive_ssh_session(conn_info: SSHConnectionInfo, private_key_path: Optional[str] = None):
    """
    Start an interactive SSH session using the system's SSH client
    
    Args:
        conn_info: SSH connection information
        private_key_path: Optional private key path (if None, uses SSH agent/default keys)
    """
    import subprocess
    
    try:
        # Build SSH command
        ssh_cmd = [
            "ssh", 
            f"{conn_info.username}@{conn_info.hostname}", 
            "-p", str(conn_info.port)
        ]
        
        # Add private key if specified
        if private_key_path:
            ssh_cmd.extend(["-i", private_key_path])
        elif conn_info.key_content:
            # Use temporary key file for key content
            with secure_temp_ssh_key(conn_info.key_content) as temp_key_path:
                ssh_cmd.extend(["-i", temp_key_path])
                # Add common options
                ssh_cmd.extend([
                    "-o", "StrictHostKeyChecking=no",  # Accept unknown hosts
                    "-o", "UserKnownHostsFile=/dev/null"  # Don't save to known_hosts
                ])
                
                logger.info(f"🔗 Starting interactive SSH: {' '.join(ssh_cmd[:-4])} [key]")
                
                # Execute SSH command - this will take over the terminal
                subprocess.run(ssh_cmd, check=False)
                return
        
        # Add common options (for cases without key content)
        ssh_cmd.extend([
            "-o", "StrictHostKeyChecking=no",  # Accept unknown hosts
            "-o", "UserKnownHostsFile=/dev/null"  # Don't save to known_hosts
        ])
        
        logger.info(f"🔗 Starting interactive SSH: {' '.join(ssh_cmd)}")
        
        # Execute SSH command - this will take over the terminal
        subprocess.run(ssh_cmd, check=False)
        
    except Exception as e:
        logger.error(f"❌ Interactive SSH failed: {e}")
        raise




def test_ssh_connection(conn_info: SSHConnectionInfo, test_both_clients: bool = True) -> Tuple[bool, str]:
    """
    Test SSH connection with both paramiko and asyncssh (if available)
    
    Args:
        conn_info: SSH connection information
        test_both_clients: Whether to test both sync and async clients
        
    Returns:
        Tuple of (success, message)
    """
    results = []
    
    # Test Paramiko (sync)
    try:
        client = UniversalSSHClient()
        if client.connect(conn_info):
            exit_code, stdout, stderr = client.exec_command("echo 'SSH_TEST_SUCCESS'", timeout=10)
            client.close()
            
            if exit_code == 0 and "SSH_TEST_SUCCESS" in stdout:
                results.append("✅ Paramiko: Connection successful with output")
            elif exit_code == 0:
                results.append("✅ Paramiko: Connection successful (limited output)")
            else:
                results.append(f"❌ Paramiko: Command failed - exit code {exit_code}")
        else:
            results.append("❌ Paramiko: Connection failed")
    except ImportError:
        results.append("⚠️ Paramiko: Not available")
    except Exception as e:
        results.append(f"❌ Paramiko: Error - {e}")
    
    # Test AsyncSSH (async) if requested
    if test_both_clients:
        try:
            import asyncio
            
            async def test_async():
                client = UniversalSSHClient()
                try:
                    if await client.aconnect(conn_info):
                        exit_code, stdout, stderr = await client.aexec_command("echo 'SSH_TEST_SUCCESS'", timeout=10)
                        await client.aclose()
                        
                        if exit_code == 0 and "SSH_TEST_SUCCESS" in stdout:
                            return "✅ AsyncSSH: Connection successful with output"
                        elif exit_code == 0:
                            return "✅ AsyncSSH: Connection successful (limited output)"
                        else:
                            return f"❌ AsyncSSH: Command failed - exit code {exit_code}"
                    else:
                        return "❌ AsyncSSH: Connection failed"
                except Exception as e:
                    return f"❌ AsyncSSH: Error - {e}"
            
            async_result = asyncio.run(test_async())
            results.append(async_result)
            
        except ImportError:
            results.append("⚠️ AsyncSSH: Not available")
        except Exception as e:
            results.append(f"❌ AsyncSSH: Error - {e}")
    
    # Determine overall success
    success = any("✅" in result for result in results)
    message = "\n".join(results)
    
    return success, message


