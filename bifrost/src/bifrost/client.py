"""Bifrost SDK - Python client for remote GPU execution and job management."""

import paramiko
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Iterator, List, Dict, Any
import logging

from .types import (
    SSHConnection, JobInfo, JobStatus, CopyResult, RemotePath,
    BifrostError, ConnectionError, JobError, TransferError
)
from .deploy import GitDeployment


logger = logging.getLogger(__name__)


class BifrostClient:
    """
    Bifrost SDK client for remote GPU execution and job management.
    
    Provides programmatic access to all Bifrost functionality:
    - Remote code execution (detached and synchronous)
    - Job monitoring and log streaming
    - File transfer operations
    - Git-based code deployment
    
    Example:
        client = BifrostClient("root@gpu.example.com:22")
        job = client.run_detached("python train_model.py")
        client.wait_for_completion(job.job_id)
        client.copy_files("/remote/outputs/", "./results/", recursive=True)
    """
    
    def __init__(
        self, 
        ssh_connection: str,
        timeout: int = 30,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Bifrost client.
        
        Args:
            ssh_connection: SSH connection string like 'user@host:port'
            timeout: SSH connection timeout in seconds
            progress_callback: Optional callback for file transfer progress
            logger: Optional logger for SDK operations
        """
        self.ssh = SSHConnection.from_string(ssh_connection)
        self.timeout = timeout
        self.progress_callback = progress_callback
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection will be established on-demand
        self._ssh_client: Optional[paramiko.SSHClient] = None
    
    def _get_ssh_client(self) -> paramiko.SSHClient:
        """Get or create SSH client connection."""
        if self._ssh_client is None or self._ssh_client.get_transport() is None:
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            try:
                self._ssh_client.connect(
                    hostname=self.ssh.host,
                    port=self.ssh.port, 
                    username=self.ssh.user,
                    timeout=self.timeout
                )
                self.logger.info(f"Connected to {self.ssh}")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to {self.ssh}: {e}")
        
        return self._ssh_client
    
    def run(self, command: str, env: Optional[Dict[str, str]] = None, no_deploy: bool = False) -> str:
        """
        Execute command synchronously on remote instance.
        
        Args:
            command: Command to execute
            env: Optional environment variables
            no_deploy: Skip git deployment (legacy mode)
            
        Returns:
            Command output as string
            
        Raises:
            ConnectionError: SSH connection failed
            JobError: Command execution failed
        """
        try:
            if not no_deploy:
                # Use GitDeployment for code deployment
                deployment = GitDeployment(self.ssh.user, self.ssh.host, self.ssh.port)
                env_list = [f"{k}={v}" for k, v in (env or {}).items()]
                exit_code = deployment.deploy_and_execute(command, env_list)
                
                if exit_code != 0:
                    raise JobError(f"Command failed with exit code {exit_code}")
                
                return "Command completed successfully"  # GitDeployment streams output directly
            else:
                # Legacy mode - direct execution
                ssh_client = self._get_ssh_client()
                exec_command = command
                
                # Add environment variables if provided
                if env:
                    env_str = " ".join([f"{k}={v}" for k, v in env.items()])
                    exec_command = f"env {env_str} {exec_command}"
                
                # Execute command
                stdin, stdout, stderr = ssh_client.exec_command(exec_command)
                
                # Wait for completion and get results
                exit_code = stdout.channel.recv_exit_status()
                output = stdout.read().decode('utf-8')
                error = stderr.read().decode('utf-8')
                
                if exit_code != 0:
                    raise JobError(f"Command failed with exit code {exit_code}: {error}")
                
                return output
            
        except Exception as e:
            if isinstance(e, (ConnectionError, JobError)):
                raise
            raise JobError(f"Execution failed: {e}")
    
    def run_detached(
        self, 
        command: str, 
        env: Optional[Dict[str, str]] = None,
        no_deploy: bool = False
    ) -> JobInfo:
        """
        Execute command as detached background job.
        
        Args:
            command: Command to execute
            env: Optional environment variables  
            no_deploy: Skip git deployment
            
        Returns:
            JobInfo object with job details
            
        Raises:
            ConnectionError: SSH connection failed
            JobError: Job creation failed
        """
        try:
            if not no_deploy:
                # Use GitDeployment for detached execution
                deployment = GitDeployment(self.ssh.user, self.ssh.host, self.ssh.port)
                env_list = [f"{k}={v}" for k, v in (env or {}).items()]
                job_id = deployment.deploy_and_execute_detached(command, env_list)
            else:
                # TODO: Implement legacy detached mode
                raise JobError("Legacy detached mode not yet implemented in SDK")
            
            # Return job info
            return JobInfo(
                job_id=job_id,
                status=JobStatus.STARTING,
                command=command,
                start_time=datetime.now()
            )
            
        except Exception as e:
            raise JobError(f"Failed to create detached job: {e}")
    
    def get_job_status(self, job_id: str) -> JobInfo:
        """
        Get current status of a detached job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobInfo with current status
            
        Raises:
            ConnectionError: SSH connection failed
            JobError: Job not found or status check failed
        """
        try:
            ssh_client = self._get_ssh_client()
            
            # Get job metadata
            metadata_cmd = f"cat ~/.bifrost/jobs/{job_id}/metadata.json 2>/dev/null"
            stdin, stdout, stderr = ssh_client.exec_command(metadata_cmd)
            
            if stdout.channel.recv_exit_status() != 0:
                raise JobError(f"Job {job_id} not found")
            
            metadata = json.loads(stdout.read().decode())
            
            # Get current status
            status_cmd = f"cat ~/.bifrost/jobs/{job_id}/status 2>/dev/null"
            stdin, stdout, stderr = ssh_client.exec_command(status_cmd) 
            status_str = stdout.read().decode().strip()
            
            # Get end time if available
            end_time = None
            end_time_cmd = f"cat ~/.bifrost/jobs/{job_id}/end_time 2>/dev/null"
            stdin, stdout, stderr = ssh_client.exec_command(end_time_cmd)
            if stdout.channel.recv_exit_status() == 0:
                end_time_str = stdout.read().decode().strip()
                if end_time_str:
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            
            # Calculate runtime
            start_time = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00'))
            runtime_seconds = None
            if end_time:
                runtime_seconds = (end_time - start_time).total_seconds()
            else:
                runtime_seconds = (datetime.now().astimezone() - start_time).total_seconds()
            
            return JobInfo(
                job_id=job_id,
                status=JobStatus(status_str) if status_str else JobStatus.PENDING,
                command=metadata.get('command', ''),
                start_time=start_time,
                end_time=end_time,
                runtime_seconds=runtime_seconds
            )
            
        except json.JSONDecodeError as e:
            raise JobError(f"Invalid job metadata for {job_id}: {e}")
        except Exception as e:
            if isinstance(e, (ConnectionError, JobError)):
                raise
            raise JobError(f"Failed to get job status: {e}")
    
    def get_all_jobs(self) -> List[JobInfo]:
        """
        Get status of all jobs on the remote instance.
        
        Returns:
            List of JobInfo objects for all jobs
            
        Raises:
            ConnectionError: SSH connection failed
        """
        try:
            ssh_client = self._get_ssh_client()
            
            # Get list of job directories
            stdin, stdout, stderr = ssh_client.exec_command("ls -1 ~/.bifrost/jobs/ 2>/dev/null || echo ''")
            job_dirs = [d.strip() for d in stdout.read().decode().split('\n') if d.strip()]
            
            jobs = []
            for job_id in job_dirs:
                try:
                    job_info = self.get_job_status(job_id)
                    jobs.append(job_info)
                except JobError:
                    # Skip jobs that can't be read
                    continue
            
            return jobs
            
        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"Failed to list jobs: {e}")
    
    def get_job_logs(self, job_id: str, lines: int = 100) -> str:
        """
        Get recent logs from a job.
        
        Args:
            job_id: Job identifier
            lines: Number of lines to retrieve (default: 100)
            
        Returns:
            Log content as string
            
        Raises:
            ConnectionError: SSH connection failed
            JobError: Job not found or logs unavailable
        """
        try:
            ssh_client = self._get_ssh_client()
            
            log_file = f"~/.bifrost/jobs/{job_id}/job.log"
            
            # Check if log file exists
            stdin, stdout, stderr = ssh_client.exec_command(f"test -f {log_file}")
            if stdout.channel.recv_exit_status() != 0:
                raise JobError(f"No log file found for job {job_id}")
            
            # Get last N lines
            tail_cmd = f"tail -n {lines} {log_file}"
            stdin, stdout, stderr = ssh_client.exec_command(tail_cmd)
            
            if stdout.channel.recv_exit_status() != 0:
                error = stderr.read().decode()
                raise JobError(f"Failed to read logs: {error}")
            
            return stdout.read().decode()
            
        except Exception as e:
            if isinstance(e, (ConnectionError, JobError)):
                raise
            raise JobError(f"Failed to get job logs: {e}")
    
    def follow_job_logs(self, job_id: str) -> Iterator[str]:
        """
        Stream job logs in real-time (like tail -f).
        
        Args:
            job_id: Job identifier
            
        Yields:
            Log lines as they are written
            
        Raises:
            ConnectionError: SSH connection failed
            JobError: Job not found or logs unavailable
        """
        try:
            ssh_client = self._get_ssh_client()
            
            log_file = f"~/.bifrost/jobs/{job_id}/job.log"
            
            # Use tail -f to follow the log file
            tail_cmd = f"tail -f {log_file}"
            stdin, stdout, stderr = ssh_client.exec_command(tail_cmd)
            
            # Stream output line by line
            for line in iter(stdout.readline, ""):
                yield line.rstrip('\n')
                
        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise JobError(f"Failed to follow job logs: {e}")
    
    def wait_for_completion(self, job_id: str, poll_interval: float = 5.0, timeout: Optional[float] = None) -> JobInfo:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            timeout: Optional timeout in seconds
            
        Returns:
            Final JobInfo when job completes
            
        Raises:
            JobError: Job failed or timeout exceeded
            ConnectionError: SSH connection failed
        """
        start_time = time.time()
        
        while True:
            job_info = self.get_job_status(job_id)
            
            if job_info.is_complete:
                return job_info
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise JobError(f"Timeout waiting for job {job_id} to complete")
            
            time.sleep(poll_interval)
    
    def copy_files(
        self, 
        remote_path: str, 
        local_path: str, 
        recursive: bool = False
    ) -> CopyResult:
        """
        Copy files from remote to local machine.
        
        Args:
            remote_path: Remote file or directory path
            local_path: Local destination path
            recursive: Copy directories recursively
            
        Returns:
            CopyResult with transfer statistics
            
        Raises:
            ConnectionError: SSH connection failed
            TransferError: File transfer failed
        """
        start_time = time.time()
        
        try:
            ssh_client = self._get_ssh_client()
            
            # Check if remote path exists (expand tilde if present)
            if remote_path.startswith('~/'):
                # For tilde paths, don't use quotes so bash can expand ~
                stdin, stdout, stderr = ssh_client.exec_command(f"test -e {remote_path}")
            else:
                # For other paths, use quotes for safety
                stdin, stdout, stderr = ssh_client.exec_command(f"test -e '{remote_path}'")
            if stdout.channel.recv_exit_status() != 0:
                raise TransferError(f"Remote path not found: {remote_path}")
            
            # Check if remote path is directory (expand tilde if present)
            if remote_path.startswith('~/'):
                stdin, stdout, stderr = ssh_client.exec_command(f"test -d {remote_path}")
            else:
                stdin, stdout, stderr = ssh_client.exec_command(f"test -d '{remote_path}'")
            is_directory = stdout.channel.recv_exit_status() == 0
            
            if is_directory and not recursive:
                raise TransferError(f"{remote_path} is a directory. Use recursive=True")
            
            # Create SFTP client
            sftp = ssh_client.open_sftp()
            
            try:
                files_copied = 0
                total_bytes = 0
                
                if is_directory:
                    files_copied, total_bytes = self._copy_directory(sftp, ssh_client, remote_path, local_path)
                else:
                    total_bytes = self._copy_file(sftp, remote_path, local_path)
                    files_copied = 1
                
                duration = time.time() - start_time
                
                return CopyResult(
                    success=True,
                    files_copied=files_copied,
                    total_bytes=total_bytes,
                    duration_seconds=duration
                )
                
            finally:
                sftp.close()
                
        except Exception as e:
            if isinstance(e, (ConnectionError, TransferError)):
                raise
            duration = time.time() - start_time
            return CopyResult(
                success=False,
                files_copied=0,
                total_bytes=0,
                duration_seconds=duration,
                error_message=str(e)
            )
    
    def _copy_file(self, sftp, remote_path: str, local_path: str) -> int:
        """Copy single file and return bytes transferred."""
        # Ensure local directory exists
        local_dir = Path(local_path).parent
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file size
        file_size = sftp.stat(remote_path).st_size
        
        # Define progress callback wrapper
        def progress_wrapper(transferred, total):
            if self.progress_callback:
                self.progress_callback("file", transferred, total)
        
        # Copy file with optional progress reporting
        if file_size > 1024 * 1024 and self.progress_callback:  # Files >1MB
            sftp.get(remote_path, local_path, callback=progress_wrapper)
        else:
            sftp.get(remote_path, local_path)
        
        return file_size
    
    def _copy_directory(self, sftp, ssh_client, remote_path: str, local_path: str) -> tuple[int, int]:
        """Copy directory recursively and return (files_copied, total_bytes)."""
        # Get directory listing (expand tilde if present)
        if remote_path.startswith('~/'):
            stdin, stdout, stderr = ssh_client.exec_command(f"find {remote_path} -type f")
        else:
            stdin, stdout, stderr = ssh_client.exec_command(f"find '{remote_path}' -type f")
        if stdout.channel.recv_exit_status() != 0:
            error = stderr.read().decode()
            raise TransferError(f"Failed to list directory contents: {error}")
        
        file_list = [f.strip() for f in stdout.read().decode().split('\n') if f.strip()]
        
        files_copied = 0
        total_bytes = 0
        
        for remote_file in file_list:
            # Calculate relative path and local destination
            rel_path = os.path.relpath(remote_file, remote_path)
            local_file = os.path.join(local_path, rel_path)
            
            # Copy file
            try:
                file_bytes = self._copy_file(sftp, remote_file, local_file)
                files_copied += 1
                total_bytes += file_bytes
            except Exception as e:
                self.logger.warning(f"Failed to copy {rel_path}: {e}")
        
        return files_copied, total_bytes
    
    def close(self):
        """Close SSH connection."""
        if self._ssh_client:
            self._ssh_client.close()
            self._ssh_client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()