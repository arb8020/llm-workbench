"""Bifrost SDK data types and structures."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    STARTING = "starting"


@dataclass
class SSHConnection:
    """SSH connection information."""
    user: str
    host: str 
    port: int
    
    @classmethod
    def from_string(cls, ssh_string: str) -> 'SSHConnection':
        """Parse SSH string in multiple formats.
        
        Supports:
        - 'user@host:port' format
        - 'ssh -p port user@host' format (standard SSH command)
        """
        # Handle standard SSH command format: "ssh -p port user@host"
        if ssh_string.startswith('ssh '):
            return cls._parse_ssh_command(ssh_string)
        
        # Handle user@host:port format
        if '@' not in ssh_string or ':' not in ssh_string:
            raise ValueError(f"Invalid SSH format: {ssh_string}. Expected: user@host:port or ssh -p port user@host")
        
        user_host, port_str = ssh_string.rsplit(':', 1)
        user, host = user_host.split('@', 1)
        
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port: {port_str}")
        
        return cls(user=user, host=host, port=port)
    
    @classmethod
    def _parse_ssh_command(cls, ssh_cmd: str) -> 'SSHConnection':
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
        
        user, host = user_host.split('@', 1)
        
        return cls(user=user, host=host, port=port)
    
    def __str__(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"


@dataclass 
class JobInfo:
    """Information about a detached job."""
    job_id: str
    status: JobStatus
    command: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    runtime_seconds: Optional[float] = None
    
    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status in [JobStatus.RUNNING, JobStatus.STARTING]
    
    @property  
    def is_complete(self) -> bool:
        """Check if job has finished (successfully or failed)."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED]


@dataclass
class CopyResult:
    """Result of a file copy operation."""
    success: bool
    files_copied: int
    total_bytes: int
    duration_seconds: float
    error_message: Optional[str] = None
    
    @property
    def throughput_mbps(self) -> float:
        """Calculate transfer throughput in MB/s."""
        if self.duration_seconds > 0:
            return (self.total_bytes / (1024 * 1024)) / self.duration_seconds
        return 0.0


@dataclass
class RemotePath:
    """Represents a remote file path."""
    path: str
    
    def __post_init__(self):
        # Ensure path is absolute for consistency
        if not self.path.startswith('/') and not self.path.startswith('~'):
            self.path = f"./{self.path}"


class BifrostError(Exception):
    """Base exception for Bifrost SDK errors."""
    pass


class ConnectionError(BifrostError):
    """SSH connection related errors."""
    pass


class JobError(BifrostError):
    """Job execution related errors."""
    pass


class TransferError(BifrostError):
    """File transfer related errors."""
    pass