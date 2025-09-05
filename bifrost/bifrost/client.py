"""Bifrost SDK - Python client for remote GPU execution and job management."""

import paramiko
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Iterator, List, Dict
import logging

from .types import (
    SSHConnection, JobInfo, JobStatus, CopyResult, ConnectionError, JobError, TransferError
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
        ssh_key_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Bifrost client.
        
        Args:
            ssh_connection: SSH connection string like 'user@host:port'
            timeout: SSH connection timeout in seconds
            ssh_key_path: Optional path to SSH private key file
            progress_callback: Optional callback for file transfer progress
            logger: Optional logger for SDK operations
        """
        self.ssh = SSHConnection.from_string(ssh_connection)
        self.timeout = timeout
        self.ssh_key_path = ssh_key_path
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
                # Load SSH key if provided
                key_content = self._load_ssh_key()
                
                if key_content:
                    # Use provided key
                    import io
                    from paramiko import RSAKey, Ed25519Key, ECDSAKey
                    
                    key_file = io.StringIO(key_content)
                    # Try different key types
                    private_key = None
                    for key_class in [RSAKey, Ed25519Key, ECDSAKey]:
                        try:
                            key_file.seek(0)
                            private_key = key_class.from_private_key(key_file)
                            break
                        except Exception:
                            continue
                    
                    if not private_key:
                        raise ConnectionError(f"Could not parse SSH key at {self.ssh_key_path}")
                    
                    self._ssh_client.connect(
                        hostname=self.ssh.host,
                        port=self.ssh.port, 
                        username=self.ssh.user,
                        pkey=private_key,
                        timeout=self.timeout
                    )
                else:
                    # Use SSH agent or default keys
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
    
    def _load_ssh_key(self) -> Optional[str]:
        """Load SSH private key content from file path."""
        if not self.ssh_key_path:
            return None
        
        import os
        key_path = os.path.expanduser(self.ssh_key_path)
        try:
            with open(key_path, 'r') as f:
                return f.read()
        except Exception as e:
            raise ConnectionError(f"Failed to load SSH key from {key_path}: {e}")
    
    def push(self, isolated: bool = False, target_dir: Optional[str] = None, uv_extra: Optional[str] = None) -> str:
        """
        Push/sync local code to remote instance without execution.
        
        This method:
        1. Creates or updates Git worktree on remote instance
        2. Copies current directory to remote via Git
        3. Auto-detects and installs Python dependencies
        4. Returns path to deployed code
        
        Mental model: Like `git push` - send code to remote
        
        Args:
            isolated: Create isolated worktree with job_id (default: False)
            target_dir: Custom directory name (only used with isolated=True)
            uv_extra: Optional extra group for uv sync (e.g., 'interp')
            
        Returns:
            Path to deployed code directory
            
        Raises:
            ConnectionError: SSH connection failed
            JobError: Code deployment failed
        """
        try:
            deployment = GitDeployment(self.ssh.user, self.ssh.host, self.ssh.port)
            
            if isolated:
                # Use existing isolated worktree deployment
                worktree_path = deployment.deploy_code_only(target_dir, uv_extra)
                self.logger.info(f"Code deployed to isolated worktree: {worktree_path}")
                return worktree_path
            else:
                # Use workspace deployment (shared directory)
                workspace_path = target_dir or "~/.bifrost/workspace"
                deployment.deploy_to_workspace(workspace_path, uv_extra)
                self.logger.info(f"Code deployed to workspace: {workspace_path}")
                return workspace_path
        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise JobError(f"Code deployment failed: {e}")
    
    def exec(self, command: str, env: Optional[Dict[str, str]] = None, working_dir: Optional[str] = None, worktree: Optional[str] = None) -> str:
        """
        Execute command in remote environment.
        
        This method:
        1. Executes command directly on remote instance
        2. Runs in context of working directory (defaults to ~/.bifrost/workspace/)
        3. Applies environment variables if provided
        4. Returns command output
        
        Mental model: Like `docker exec` - run command in remote environment
        
        Args:
            command: Command to execute
            env: Environment variables to set
            working_dir: Working directory (defaults to ~/.bifrost/workspace/ if deployed)
            worktree: DEPRECATED - use working_dir instead
            
        Returns:
            Command output as string
            
        Raises:
            ConnectionError: SSH connection failed
            JobError: Command execution failed
        """
        try:
            ssh_client = self._get_ssh_client()
            
            # Handle deprecated worktree parameter
            if worktree is not None:
                import warnings
                warnings.warn(
                    "The 'worktree' parameter is deprecated. Use 'working_dir' instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                if working_dir is None:
                    working_dir = worktree
            
            # Build command with working directory context
            if working_dir:
                # Validate directory exists for better error messages
                stdin, stdout, stderr = ssh_client.exec_command(f"test -d {working_dir}")
                if stdout.channel.recv_exit_status() != 0:
                    raise JobError(f"Directory not found: {working_dir}")
                full_command = f"cd {working_dir} && {command}"
            else:
                # Default to workspace if it exists, otherwise home with warning
                default_dir = "~/.bifrost/workspace"
                stdin, stdout, stderr = ssh_client.exec_command(f"test -d {default_dir}")
                if stdout.channel.recv_exit_status() == 0:
                    full_command = f"cd {default_dir} && {command}"
                    self.logger.info(f"Using default working directory: {default_dir}")
                else:
                    full_command = command
                    self.logger.warning("No code deployed yet. Running from home directory. Consider running push() first.")
            
            # Always use streaming execution for consistent real-time output
            from .deploy import execute_with_env_injection
            exit_code, stdout_content, stderr_content = execute_with_env_injection(
                ssh_client, full_command, env or {}
            )
            if exit_code != 0:
                raise JobError(f"Command failed with exit code {exit_code}: {stderr_content}")
            return stdout_content
            
        except Exception as e:
            if isinstance(e, (ConnectionError, JobError)):
                raise
            raise JobError(f"Execution failed: {e}")
    
    def deploy(self, command: str, env: Optional[Dict[str, str]] = None, isolated: bool = False, uv_extra: Optional[str] = None, working_dir: Optional[str] = None) -> str:
        """
        Deploy local code and execute command (convenience method).
        
        This method:
        1. Calls push() to deploy code
        2. Calls exec() to run command in specified working directory
        3. Equivalent to: push() followed by exec(command, working_dir=working_dir or code_path)
        
        Mental model: Like deployment tools - deploy and start application
        
        Args:
            command: Command to execute after deployment
            env: Environment variables to set
            isolated: Create isolated worktree with job_id (default: False)
            uv_extra: Optional extra group for uv sync (e.g., 'interp')
            working_dir: Working directory to run command in (defaults to deployed code path)
            
        Returns:
            Command output as string
            
        Raises:
            ConnectionError: SSH connection failed
            JobError: Deployment or execution failed
        """
        code_path = self.push(isolated=isolated, uv_extra=uv_extra)
        # Use specified working_dir or default to deployed code path
        exec_working_dir = working_dir or code_path
        return self.exec(command, env, working_dir=exec_working_dir)
    
    def run(self, command: str, env: Optional[Dict[str, str]] = None, no_deploy: bool = False) -> str:
        """
        DEPRECATED: Use deploy() instead. Will be removed in v0.2.0
        
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
        import warnings
        warnings.warn(
            "BifrostClient.run() is deprecated. Use deploy() for deployment+execution or exec() for command-only execution.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if no_deploy:
            # Legacy mode - use exec() for command execution
            return self.exec(command, env)
        else:
            # Use new deploy() method
            return self.deploy(command, env)
    
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
        
        # Convert remote_path to absolute form for proper relative path calculation
        if remote_path.startswith('~/'):
            abs_remote_path = remote_path.replace('~', '/root', 1)
        else:
            abs_remote_path = remote_path
            
        for remote_file in file_list:
            # Calculate relative path and local destination
            rel_path = os.path.relpath(remote_file, abs_remote_path)
            local_file = os.path.join(local_path, rel_path)
            
            # Copy file
            try:
                file_bytes = self._copy_file(sftp, remote_file, local_file)
                files_copied += 1
                total_bytes += file_bytes
            except Exception as e:
                self.logger.warning(f"Failed to copy {rel_path}: {e}")
        
        return files_copied, total_bytes
    
    def upload_files(
        self, 
        local_path: str, 
        remote_path: str, 
        recursive: bool = False
    ) -> CopyResult:
        """
        Upload files from local to remote machine.
        
        Args:
            local_path: Local file or directory path
            remote_path: Remote destination path
            recursive: Upload directories recursively
            
        Returns:
            CopyResult with transfer statistics
            
        Raises:
            ConnectionError: SSH connection failed
            TransferError: File transfer failed
        """
        start_time = time.time()
        
        try:
            ssh_client = self._get_ssh_client()
            
            # Check if local path exists
            local_path_obj = Path(local_path)
            if not local_path_obj.exists():
                raise TransferError(f"Local path not found: {local_path}")
            
            is_directory = local_path_obj.is_dir()
            
            if is_directory and not recursive:
                raise TransferError(f"{local_path} is a directory. Use recursive=True")
            
            # Create SFTP client
            sftp = ssh_client.open_sftp()
            
            try:
                files_uploaded = 0
                total_bytes = 0
                
                if is_directory:
                    files_uploaded, total_bytes = self._upload_directory(sftp, ssh_client, local_path, remote_path)
                else:
                    total_bytes = self._upload_file(sftp, local_path, remote_path)
                    files_uploaded = 1
                
                duration = time.time() - start_time
                
                return CopyResult(
                    success=True,
                    files_copied=files_uploaded,
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
    
    def _create_remote_dir(self, sftp, remote_dir: str):
        """Create remote directory recursively."""
        try:
            sftp.stat(remote_dir)  # Check if directory exists
        except FileNotFoundError:
            # Directory doesn't exist, create it
            parent_dir = os.path.dirname(remote_dir)
            if parent_dir and parent_dir != remote_dir:  # Avoid infinite recursion
                self._create_remote_dir(sftp, parent_dir)
            try:
                sftp.mkdir(remote_dir)
            except OSError:
                # Directory might have been created by another process
                pass
    
    def _upload_file(self, sftp, local_path: str, remote_path: str) -> int:
        """Upload single file and return bytes transferred."""
        # Create remote directory if needed
        remote_dir = os.path.dirname(remote_path)
        if remote_dir and remote_dir != '.':
            # Create directory structure recursively
            self._create_remote_dir(sftp, remote_dir)
        
        # Get file size
        file_size = os.path.getsize(local_path)
        
        # Define progress callback wrapper
        def progress_wrapper(transferred, total):
            if self.progress_callback:
                self.progress_callback("file", transferred, total)
        
        # Upload file with optional progress reporting
        if file_size > 1024 * 1024 and self.progress_callback:  # Files >1MB
            sftp.put(local_path, remote_path, callback=progress_wrapper)
        else:
            sftp.put(local_path, remote_path)
        
        return file_size
    
    def _upload_directory(self, sftp, ssh_client, local_path: str, remote_path: str) -> tuple[int, int]:
        """Upload directory recursively and return (files_uploaded, total_bytes)."""
        local_path_obj = Path(local_path)
        
        files_uploaded = 0
        total_bytes = 0
        
        # Walk through local directory
        for local_file in local_path_obj.rglob('*'):
            if local_file.is_file():
                # Calculate relative path and remote destination
                rel_path = local_file.relative_to(local_path_obj)
                remote_file = f"{remote_path}/{rel_path}".replace('\\', '/')
                
                # Upload file
                try:
                    file_bytes = self._upload_file(sftp, str(local_file), remote_file)
                    files_uploaded += 1
                    total_bytes += file_bytes
                except Exception as e:
                    self.logger.warning(f"Failed to upload {rel_path}: {e}")
        
        return files_uploaded, total_bytes
    
    def download_files(
        self, 
        remote_path: str, 
        local_path: str, 
        recursive: bool = False
    ) -> CopyResult:
        """
        Download files from remote to local machine.
        
        This is an alias for copy_files() with clearer naming.
        
        Args:
            remote_path: Remote file or directory path
            local_path: Local destination path
            recursive: Download directories recursively
            
        Returns:
            CopyResult with transfer statistics
        """
        return self.copy_files(remote_path, local_path, recursive)
    
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