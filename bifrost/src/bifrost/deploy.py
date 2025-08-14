"""Git-based code deployment for Bifrost."""

import os
import subprocess
import uuid
import logging
from pathlib import Path
from typing import Tuple, Optional
import paramiko
from rich.console import Console
from .job_manager import JobManager, generate_job_id

logger = logging.getLogger(__name__)
console = Console()


class GitDeployment:
    """Handles git-based code deployment to remote instances."""
    
    def __init__(self, ssh_user: str, ssh_host: str, ssh_port: int, job_id: Optional[str] = None):
        self.ssh_user = ssh_user
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.job_id = job_id or str(uuid.uuid4())[:8]  # Use provided job_id or generate one
        
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
    
    def deploy_and_execute(self, command: str, env_vars: Optional[list] = None) -> int:
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
            
            # Build command with environment and working directory
            full_command = f"cd {worktree_path}"
            
            if env_vars:
                env_setup = " && ".join(f"export {var}" for var in env_vars)
                full_command += f" && {env_setup}"
            
            full_command += f" && {command}"
            
            # Execute command
            console.print(f"🔄 Executing in worktree: {command}")
            stdin, stdout, stderr = client.exec_command(full_command)
            
            # Stream output
            console.print("\n--- Remote Output ---")
            for line in stdout:
                print(line.rstrip())
            
            # Check for errors
            stderr_output = stderr.read().decode()
            if stderr_output:
                console.print(f"\n--- Remote Errors ---", style="red")
                console.print(stderr_output, style="red")
            
            exit_code = stdout.channel.recv_exit_status()
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
    
    def deploy_and_execute_detached(self, command: str, env_vars: Optional[list] = None) -> str:
        """Deploy code and execute command in detached mode, return job ID."""
        
        # Generate unique job ID
        job_id = generate_job_id()
        self.job_id = job_id  # Update instance job_id to match
        console.print(f"🆔 Generated job ID: {job_id}")
        
        # Detect git repo
        repo_name, commit_hash = self.detect_git_repo()
        
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Create job manager
        job_manager = JobManager(self.ssh_user, self.ssh_host, self.ssh_port)
        
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
            
            # Create job metadata
            job_manager.create_job_metadata(
                client, job_id, command, worktree_path, commit_hash, repo_name
            )
            
            # Upload job wrapper script (if not already present)
            job_manager.upload_job_wrapper_script(client)
            
            # Start tmux session for detached execution
            tmux_session = job_manager.start_tmux_session(client, job_id, command, env_vars)
            
            console.print(f"🚀 Job {job_id} started in session {tmux_session}")
            console.print("💡 Use 'bifrost logs {job_id}' to monitor progress (coming in Phase 2)")
            
            return job_id
            
        except Exception as e:
            console.print(f"❌ Failed to start detached job: {e}")
            console.print(f"🔍 Job data preserved for debugging: ~/.bifrost/jobs/{job_id}")
            raise
        finally:
            client.close()