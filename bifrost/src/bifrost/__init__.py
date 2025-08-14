"""Bifrost - Remote GPU execution tool."""

__version__ = "0.1.0"

# SDK interface
from .client import BifrostClient
from .types import JobInfo, JobStatus, CopyResult, BifrostError, ConnectionError, JobError, TransferError

__all__ = [
    'BifrostClient',
    'JobInfo', 'JobStatus', 'CopyResult',
    'BifrostError', 'ConnectionError', 'JobError', 'TransferError'
]