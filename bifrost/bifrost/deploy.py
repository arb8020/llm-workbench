"""Git-based code deployment for Bifrost."""

import os
import subprocess
import uuid
import logging
import shlex
import re
from pathlib import Path
from typing import Tuple, Optional, Dict
import paramiko
from rich.console import Console
from .job_manager import JobManager, generate_job_id

logger = logging.getLogger(__name__)
console = Console()

# Environment variable validation and payload creation
VALID_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def make_env_payload(env_dict: Dict[str, str]) -> bytes:
    """Create secure environment variable payload for stdin injection."""
    lines = []
    for k, v in env_dict.items():
        if not VALID_ENV_NAME.match(k):
            raise ValueError(f"Invalid environment variable name: {k}")
        # Use shlex.quote to safely handle special characters
        lines.append(f"{k}={shlex.quote(v)}")
    return ("\n".join(lines) + "\n").encode()

def wrap_with_env_loader(user_command: str) -> str:
    """Wrap user command to load environment variables from stdin."""
    # set -a: automatically export all subsequently defined variables
    # . /dev/stdin: source environment variables from stdin
    # set +a: turn off automatic export
    # Use bash -c instead of exec to handle shell builtins like cd
    return f"set -a; . /dev/stdin; set +a; bash -c {shlex.quote(user_command)}"

def execute_with_env_injection(
    client: paramiko.SSHClient, 
    command: str, 
    env_dict: Optional[Dict[str, str]] = None
) -> Tuple[int, str, str]:
    """Execute command with secure environment variable injection via stdin."""
    
    if env_dict:
        # Create environment payload
        env_payload = make_env_payload(env_dict)
        
        # Wrap command to load environment from stdin
        wrapped_command = wrap_with_env_loader(command)
        
        console.print(f"🔐 Injecting {len(env_dict)} environment variables securely")
        
        # Execute with environment injection
        stdin, stdout, stderr = client.exec_command(f"bash -lc {shlex.quote(wrapped_command)}")
        
        # Send environment variables over stdin
        stdin.write(env_payload)
        stdin.channel.shutdown_write()  # Signal end of input
        
    else:
        # No environment variables, execute normally
        stdin, stdout, stderr = client.exec_command(command)
    
    # Get outputs
    stdout_content = stdout.read().decode()
    stderr_content = stderr.read().decode()
    exit_code = stdout.channel.recv_exit_status()
    
    return exit_code, stdout_content, stderr_content


class GitDeployment:
    """Handles git-based code deployment to remote instances."""
    
    def __init__(self, ssh_user: str, ssh_host: str, ssh_port: int, job_id: Optional[str] = None):
        self.ssh_user = ssh_user
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.job_id = job_id or str(uuid.uuid4())[:8]  # Use provided job_id or generate one
    
    def detect_bootstrap_command(self, client: paramiko.SSHClient, worktree_path: str) -> str:
        """Detect Python dependency files and return appropriate bootstrap command."""
        
        # Check for dependency files in order of preference
        dep_files = [
            ("uv.lock", "pip install uv && uv sync"),
            ("pyproject.toml", "pip install uv && uv sync"), 
            ("requirements.txt", "pip install -r requirements.txt")
        ]
        
        for dep_file, bootstrap_cmd in dep_files:
            # Check if file exists in worktree
            stdin, stdout, stderr = client.exec_command(f"test -f {worktree_path}/{dep_file}")
            if stdout.channel.recv_exit_status() == 0:
                console.print(f"📦 Detected {dep_file}, adding bootstrap: {bootstrap_cmd}")
                return f"{bootstrap_cmd} && "
        
        # No dependency files found
        console.print("📦 No Python dependency files detected, skipping bootstrap")
        return ""
        
    def detect_git_repo(self) -> Tuple[str, str]:
        """Detect git repository and get repo name and current commit."""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"], 
                capture_output=True, text=True, check=True
            )
            
            # Get repo name from current directory
            repo_name = os.path.basename(os.getcwd())
            
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, text=True, check=True
            )
            commit_hash = result.stdout.strip()
            
            console.print(f"📦 Detected git repo: {repo_name} @ {commit_hash[:8]}")
            return repo_name, commit_hash
            
        except subprocess.CalledProcessError:
            raise ValueError("Not in a git repository. Please run bifrost from a git repository.")
    
    def setup_remote_structure(self, client: paramiko.SSHClient, repo_name: str) -> str:
        """Set up ~/.bifrost directory structure on remote."""
        
        # Ensure tmux is installed for detached job functionality
        console.print("🔧 Ensuring tmux is installed for detached jobs...")
        tmux_check_cmd = "which tmux || (apt-get update && apt-get install -y tmux)"
        stdin, stdout, stderr = client.exec_command(tmux_check_cmd)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            console.print("⚠️  Warning: tmux installation may have failed, but continuing...")
        else:
            console.print("✅ tmux is available")
        
        # Create directory structure
        commands = [
            "mkdir -p ~/.bifrost/repos ~/.bifrost/worktrees ~/.bifrost/jobs",
            f"mkdir -p ~/.bifrost/jobs/{self.job_id}"
        ]
        
        for cmd in commands:
            stdin, stdout, stderr = client.exec_command(cmd)
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                error = stderr.read().decode()
                raise RuntimeError(f"Failed to create remote directories: {error}")
        
        # Set up bare repo if it doesn't exist
        bare_repo_path = f"~/.bifrost/repos/{repo_name}.git"
        
        # Check if bare repo exists
        stdin, stdout, stderr = client.exec_command(f"test -d {bare_repo_path}")
        repo_exists = stdout.channel.recv_exit_status() == 0
        
        if not repo_exists:
            console.print(f"🔧 Initializing bare repo: {bare_repo_path}")
            stdin, stdout, stderr = client.exec_command(f"git init --bare {bare_repo_path}")
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                error = stderr.read().decode()
                raise RuntimeError(f"Failed to create bare repo: {error}")
        
        return bare_repo_path
    
    def push_code(self, repo_name: str, commit_hash: str, bare_repo_path: str) -> None:
        """Push current code to remote bare repository."""
        
        # Build SSH command for git push
        ssh_cmd = f"ssh -p {self.ssh_port} -o StrictHostKeyChecking=no"
        remote_url = f"{self.ssh_user}@{self.ssh_host}:{bare_repo_path}"
        
        console.print(f"📤 Pushing code to remote...")
        
        # Push current HEAD to a job-specific branch
        job_branch = f"job/{self.job_id}"
        
        try:
            # Set git SSH command
            env = os.environ.copy()
            env['GIT_SSH_COMMAND'] = ssh_cmd
            
            # Push to remote
            result = subprocess.run([
                "git", "push", remote_url, f"HEAD:refs/heads/{job_branch}"
            ], env=env, capture_output=True, text=True, check=True)
            
            console.print(f"✅ Code pushed to branch: {job_branch}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to push code: {e.stderr}")
    
    def create_worktree(self, client: paramiko.SSHClient, repo_name: str) -> str:
        """Create git worktree for this job."""
        
        bare_repo_path = f"~/.bifrost/repos/{repo_name}.git"
        worktree_path = f"~/.bifrost/worktrees/{self.job_id}"
        job_branch = f"job/{self.job_id}"
        
        console.print(f"🌳 Creating worktree: {worktree_path}")
        
        # Create worktree
        cmd = f"cd {bare_repo_path} && git worktree add {worktree_path} {job_branch}"
        stdin, stdout, stderr = client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to create worktree: {error}")
        
        console.print(f"✅ Worktree ready at: {worktree_path}")
        return worktree_path
    
    def cleanup_job(self, client: paramiko.SSHClient, repo_name: str, worktree_path: str) -> None:
        """Clean up job-specific resources."""
        
        # Remove worktree
        bare_repo_path = f"~/.bifrost/repos/{repo_name}.git"
        cmd = f"cd {bare_repo_path} && git worktree remove {worktree_path} --force"
        client.exec_command(cmd)
        
        # Remove job branch
        job_branch = f"job/{self.job_id}"
        cmd = f"cd {bare_repo_path} && git branch -D {job_branch}"
        client.exec_command(cmd)
        
        # Remove job directory
        cmd = f"rm -rf ~/.bifrost/jobs/{self.job_id}"
        client.exec_command(cmd)
    
    def deploy_and_execute(self, command: str, env_vars: Optional[Dict[str, str]] = None) -> int:
        """Deploy code and execute command with proper cleanup."""
        
        # Detect git repo
        repo_name, commit_hash = self.detect_git_repo()
        
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        worktree_path = None
        
        try:
            # Connect to remote
            console.print(f"🔗 Connecting to {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")
            client.connect(hostname=self.ssh_host, port=self.ssh_port, username=self.ssh_user)
            
            # Set up remote structure
            bare_repo_path = self.setup_remote_structure(client, repo_name)
            
            # Push code
            self.push_code(repo_name, commit_hash, bare_repo_path)
            
            # Create worktree
            worktree_path = self.create_worktree(client, repo_name)
            
            # Detect and add bootstrap command
            bootstrap_cmd = self.detect_bootstrap_command(client, worktree_path)
            
            # Build full command with working directory and bootstrap
            full_command = f"cd {worktree_path} && {bootstrap_cmd}{command}"
            
            # Execute command with secure environment injection
            console.print(f"🔄 Executing in worktree: {command}")
            exit_code, stdout_content, stderr_content = execute_with_env_injection(
                client, full_command, env_vars
            )
            
            # Stream output
            console.print("\n--- Remote Output ---")
            if stdout_content:
                print(stdout_content.rstrip())
            
            # Only show errors if command failed (non-zero exit code)
            if stderr_content and exit_code != 0:
                console.print(f"\n--- Remote Errors ---", style="red")
                console.print(stderr_content, style="red")
            
            return exit_code
            
        finally:
            # Always cleanup
            if worktree_path:
                try:
                    self.cleanup_job(client, repo_name, worktree_path)
                    console.print("🧹 Cleaned up remote resources")
                except Exception as e:
                    logger.warning(f"Failed to cleanup: {e}")
            
            client.close()
    
    def deploy_code_only(self, target_dir: Optional[str] = None) -> str:
        """Deploy code without executing commands. Returns worktree path.
        
        This method:
        1. Detects git repository and current commit
        2. Sets up remote .bifrost directory structure
        3. Pushes code to remote bare repository
        4. Creates git worktree for this deployment
        5. Installs Python dependencies if detected
        
        Args:
            target_dir: Optional specific directory name for worktree
            
        Returns:
            Path to deployed worktree on remote instance
            
        Raises:
            ValueError: If not in a git repository
            RuntimeError: If deployment fails
        """
        # Detect git repo
        repo_name, commit_hash = self.detect_git_repo()
        
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Connect to remote
            console.print(f"🔗 Connecting to {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")
            client.connect(hostname=self.ssh_host, port=self.ssh_port, username=self.ssh_user)
            
            # Set up remote structure
            bare_repo_path = self.setup_remote_structure(client, repo_name)
            
            # Push code
            self.push_code(repo_name, commit_hash, bare_repo_path)
            
            # Create worktree with optional custom directory name
            if target_dir:
                # Override job_id for custom directory name
                original_job_id = self.job_id
                self.job_id = target_dir
                worktree_path = self.create_worktree(client, repo_name)
                self.job_id = original_job_id  # Restore original
            else:
                worktree_path = self.create_worktree(client, repo_name)
            
            # Install dependencies
            bootstrap_cmd = self.detect_bootstrap_command(client, worktree_path)
            if bootstrap_cmd:
                # Remove the trailing " && " from bootstrap command for standalone execution
                bootstrap_only = bootstrap_cmd.rstrip(" && ")
                console.print(f"🔄 Installing dependencies: {bootstrap_only}")
                
                # Execute bootstrap command in worktree
                full_bootstrap = f"cd {worktree_path} && {bootstrap_only}"
                stdin, stdout, stderr = client.exec_command(full_bootstrap)
                exit_code = stdout.channel.recv_exit_status()
                
                if exit_code != 0:
                    error = stderr.read().decode()
                    console.print(f"⚠️  Dependency installation failed: {error}", style="yellow")
                    console.print("Continuing deployment without dependencies...")
                else:
                    console.print("✅ Dependencies installed successfully")
            
            console.print(f"✅ Code deployed to: {worktree_path}")
            return worktree_path
            
        finally:
            client.close()
    
    def deploy_and_execute_detached(self, command: str, env_vars: Optional[Dict[str, str]] = None) -> str:
        """Deploy code and execute command in detached mode, return job ID."""
        
        job_id = self._prepare_detached_job()
        repo_name, commit_hash = self.detect_git_repo()
        
        client = self._create_ssh_client()
        job_manager = JobManager(self.ssh_user, self.ssh_host, self.ssh_port)
        
        try:
            return self._execute_detached_deployment(
                client, job_manager, job_id, repo_name, commit_hash, command, env_vars
            )
        except Exception as e:
            console.print(f"❌ Failed to start detached job: {e}")
            console.print(f"🔍 Job data preserved for debugging: ~/.bifrost/jobs/{job_id}")
            raise
        finally:
            client.close()
    
    def _prepare_detached_job(self) -> str:
        """Generate job ID and update instance state."""
        job_id = generate_job_id()
        self.job_id = job_id
        console.print(f"🆔 Generated job ID: {job_id}")
        return job_id
    
    def _create_ssh_client(self) -> paramiko.SSHClient:
        """Create and configure SSH client."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return client
    
    def _execute_detached_deployment(
        self, 
        client: paramiko.SSHClient, 
        job_manager: JobManager, 
        job_id: str, 
        repo_name: str, 
        commit_hash: str, 
        command: str, 
        env_vars: Optional[Dict[str, str]]
    ) -> str:
        """Execute the main deployment steps for detached job."""
        
        # Connect to remote
        console.print(f"🔗 Connecting to {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")
        client.connect(hostname=self.ssh_host, port=self.ssh_port, username=self.ssh_user)
        
        # Set up remote environment
        bare_repo_path = self.setup_remote_structure(client, repo_name)
        self.push_code(repo_name, commit_hash, bare_repo_path)
        worktree_path = self.create_worktree(client, repo_name)
        
        # Prepare command with bootstrap
        bootstrap_cmd = self.detect_bootstrap_command(client, worktree_path)
        full_command = f"{bootstrap_cmd}{command}"
        
        # Set up job execution
        job_manager.create_job_metadata(
            client, job_id, full_command, worktree_path, commit_hash, repo_name
        )
        job_manager.upload_job_wrapper_script(client)
        
        # Start detached execution
        tmux_session = job_manager.start_tmux_session(client, job_id, full_command, env_vars)
        
        console.print(f"🚀 Job {job_id} started in session {tmux_session}")
        console.print("💡 Use 'bifrost logs {job_id}' to monitor progress (coming in Phase 2)")
        
        return job_id