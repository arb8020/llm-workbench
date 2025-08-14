"""Bifrost SDK data types and structures."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path


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
        """Parse SSH string like 'user@host:port' into connection info."""
        if '@' not in ssh_string or ':' not in ssh_string:
            raise ValueError(f"Invalid SSH format: {ssh_string}. Expected: user@host:port")
        
        user_host, port_str = ssh_string.rsplit(':', 1)
        user, host = user_host.split('@', 1)
        
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port: {port_str}")
        
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