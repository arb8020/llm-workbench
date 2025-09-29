"""Git-based code deployment for Bifrost."""

import os
import subprocess
import uuid
import logging
import shlex
import re
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
    
    # Always print the command being executed
    console.print(f"🔄 Executing: {command}")
    
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
    
    # Stream output in real-time
    console.print("\n--- Remote Output ---")
    stdout_buffer = []
    stderr_buffer = []
    
    # Set channels to non-blocking mode for real-time streaming
    stdout.channel.settimeout(0.1)
    stderr.channel.settimeout(0.1)
    
    while not stdout.channel.exit_status_ready():
        # Read from stdout
        try:
            chunk = stdout.read(1024).decode()
            if chunk:
                print(chunk, end='', flush=True)
                stdout_buffer.append(chunk)
        except Exception:
            pass  # No data available
            
        # Read from stderr  
        try:
            chunk = stderr.read(1024).decode()
            if chunk:
                console.print(chunk, style="red", end='')
                stderr_buffer.append(chunk)
        except Exception:
            pass  # No data available
    
    # Read any remaining output
    try:
        remaining_stdout = stdout.read().decode()
        if remaining_stdout:
            print(remaining_stdout, end='', flush=True)
            stdout_buffer.append(remaining_stdout)
    except Exception:
        pass
        
    try:
        remaining_stderr = stderr.read().decode()
        if remaining_stderr:
            console.print(remaining_stderr, style="red", end='')
            stderr_buffer.append(remaining_stderr)
    except Exception:
        pass
    
    exit_code = stdout.channel.recv_exit_status()
    stdout_content = ''.join(stdout_buffer)
    stderr_content = ''.join(stderr_buffer)
    
    return exit_code, stdout_content, stderr_content


class GitDeployment:
    """Handles git-based code deployment to remote instances."""
    
    def __init__(self, ssh_user: str, ssh_host: str, ssh_port: int, job_id: Optional[str] = None):
        self.ssh_user = ssh_user
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.job_id = job_id or str(uuid.uuid4())[:8]  # Use provided job_id or generate one
    
    def detect_bootstrap_command(self, client: paramiko.SSHClient, worktree_path: str, uv_extra: Optional[str] = None) -> str:
        """Detect Python dependency files and return appropriate bootstrap command."""
        # Allow callers to skip or freeze dependency bootstrap for faster reuse
        skip_bootstrap = os.environ.get("BIFROST_SKIP_BOOTSTRAP") == "1"
        frozen = os.environ.get("BIFROST_BOOTSTRAP_FROZEN") == "1"
        if skip_bootstrap:
            console.print("📦 Skipping dependency bootstrap due to BIFROST_SKIP_BOOTSTRAP=1")
            return ""

        # Check for dependency files in order of preference
        uv_sync_cmd = "pip install uv && uv sync"
        if frozen:
            uv_sync_cmd += " --frozen"
        if uv_extra:
            uv_sync_cmd += f" --extra {uv_extra}"
            
        dep_files = [
            ("uv.lock", uv_sync_cmd),
            ("pyproject.toml", uv_sync_cmd), 
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
        
        console.print("📤 Pushing code to remote...")
        
        # Push current HEAD to a job-specific branch
        job_branch = f"job/{self.job_id}"
        
        try:
            # Set git SSH command
            env = os.environ.copy()
            env['GIT_SSH_COMMAND'] = ssh_cmd
            
            # Push to remote
            subprocess.run([
                "git", "push", remote_url, f"HEAD:refs/heads/{job_branch}"
            ], env=env, capture_output=True, text=True, check=True)
            
            console.print(f"✅ Code pushed to branch: {job_branch}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to push code: {e.stderr}")
    
    def push_code_to_main(self, repo_name: str, commit_hash: str, bare_repo_path: str) -> None:
        """Push current code to remote bare repository main branch."""
        
        # Build SSH command for git push
        ssh_cmd = f"ssh -p {self.ssh_port} -o StrictHostKeyChecking=no"
        remote_url = f"{self.ssh_user}@{self.ssh_host}:{bare_repo_path}"
        
        console.print("📤 Pushing code to remote main branch...")
        
        try:
            # Set git SSH command
            env = os.environ.copy()
            env['GIT_SSH_COMMAND'] = ssh_cmd
            
            # Push to remote main branch
            subprocess.run([
                "git", "push", remote_url, "HEAD:refs/heads/main"
            ], env=env, capture_output=True, text=True, check=True)
            
            console.print("✅ Code pushed to main branch")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to push code to main: {e.stderr}")
    
    def create_or_update_workspace(self, client: paramiko.SSHClient, bare_repo_path: str, workspace_path: str) -> None:
        """Create or update shared workspace directory."""
        
        console.print(f"🌳 Setting up workspace: {workspace_path}")
        
        # Check if workspace already exists
        stdin, stdout, stderr = client.exec_command(f"test -d {workspace_path}")
        workspace_exists = stdout.channel.recv_exit_status() == 0
        
        if workspace_exists:
            # Update existing workspace
            console.print("📝 Updating existing workspace...")
            
            # Ensure origin remote is configured (worktrees don't have it by default)
            remote_url = f"{bare_repo_path}"
            config_cmd = f"cd {workspace_path} && git remote get-url origin 2>/dev/null || git remote add origin {remote_url}"
            stdin, stdout, stderr = client.exec_command(config_cmd)
            
            # Fetch latest changes from bare repo
            fetch_cmd = f"cd {workspace_path} && git fetch origin main"
            stdin, stdout, stderr = client.exec_command(fetch_cmd)
            fetch_exit = stdout.channel.recv_exit_status()
            
            if fetch_exit != 0:
                fetch_error = stderr.read().decode()
                console.print(f"⚠️  Fetch failed: {fetch_error}")
                # Try alternative: reset to match bare repo
                reset_cmd = f"cd {workspace_path} && git reset --hard main"
                stdin, stdout, stderr = client.exec_command(reset_cmd)
                if stdout.channel.recv_exit_status() != 0:
                    console.print("⚠️  Reset failed, workspace may be out of sync")
            else:
                # Pull or reset to latest
                pull_cmd = f"cd {workspace_path} && git reset --hard origin/main"
                stdin, stdout, stderr = client.exec_command(pull_cmd)
                if stdout.channel.recv_exit_status() != 0:
                    console.print("⚠️  Reset to origin/main failed")
            
            console.print("✅ Workspace updated successfully")
        else:
            # Create new workspace
            console.print("🆕 Creating new workspace...")
            
            # Create workspace as git worktree from main branch
            cmd = f"cd {bare_repo_path} && git worktree add {workspace_path} main"
            stdin, stdout, stderr = client.exec_command(cmd)
            exit_code = stdout.channel.recv_exit_status()
            
            if exit_code != 0:
                error = stderr.read().decode()
                raise RuntimeError(f"Failed to create workspace: {error}")
            
            console.print(f"✅ Workspace created at: {workspace_path}")
    
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
    
    def deploy_and_execute(self, client: paramiko.SSHClient, command: str, env_vars: Optional[Dict[str, str]] = None, uv_extra: Optional[str] = None) -> int:
        """Deploy code and execute command using shared workspace for better Python imports.

        Args:
            client: Connected Paramiko SSH client (managed by caller)
            command: Command to execute
            env_vars: Optional environment variables
            uv_extra: Optional extra group for uv sync
        """

        # Deploy to shared workspace instead of job-specific worktree
        workspace_path = self.deploy_to_workspace(client, uv_extra=uv_extra)

        # Detect and add bootstrap command
        bootstrap_cmd = self.detect_bootstrap_command(client, workspace_path, uv_extra)

        # Build full command with working directory and bootstrap
        full_command = f"cd {workspace_path} && {bootstrap_cmd}{command}"

        # Execute command with secure environment injection
        exit_code, stdout_content, stderr_content = execute_with_env_injection(
            client, full_command, env_vars
        )

        return exit_code
    
    def deploy_to_workspace(self, client: paramiko.SSHClient, workspace_path: str = "~/.bifrost/workspace", uv_extra: Optional[str] = None) -> str:
        """Deploy code to shared workspace directory.

        This method:
        1. Detects git repository and current commit
        2. Sets up remote .bifrost directory structure
        3. Pushes code to remote bare repository
        4. Creates or updates shared workspace directory
        5. Installs Python dependencies if detected

        Args:
            client: Connected Paramiko SSH client (managed by caller)
            workspace_path: Path to workspace directory (default: ~/.bifrost/workspace)
            uv_extra: Optional extra group for uv sync (e.g., 'interp')

        Returns:
            Path to workspace directory

        Raises:
            ValueError: If not in a git repository
            RuntimeError: If deployment fails
        """
        # Detect git repo
        repo_name, commit_hash = self.detect_git_repo()

        # Set up remote structure
        bare_repo_path = self.setup_remote_structure(client, repo_name)

        # Push code to main branch instead of job-specific branch
        self.push_code_to_main(repo_name, commit_hash, bare_repo_path)

        # Create or update workspace
        self.create_or_update_workspace(client, bare_repo_path, workspace_path)

        # Install dependencies
        bootstrap_cmd = self.detect_bootstrap_command(client, workspace_path, uv_extra)
        if bootstrap_cmd:
            bootstrap_only = bootstrap_cmd.rstrip(" && ")
            console.print(f"🔄 Installing dependencies: {bootstrap_only}")

            stdin, stdout, stderr = client.exec_command(f"cd {workspace_path} && {bootstrap_only}")
            exit_code = stdout.channel.recv_exit_status()

            if exit_code != 0:
                error = stderr.read().decode()
                console.print(f"⚠️  Dependency installation warning: {error}")
            else:
                console.print("✅ Dependencies installed successfully")

        console.print(f"🎉 Code deployed successfully to workspace: {workspace_path}")
        return workspace_path
    
    def deploy_code_only(self, client: paramiko.SSHClient, target_dir: Optional[str] = None, uv_extra: Optional[str] = None) -> str:
        """Deploy code without executing commands. Returns worktree path.

        This method:
        1. Detects git repository and current commit
        2. Sets up remote .bifrost directory structure
        3. Pushes code to remote bare repository
        4. Creates git worktree for this deployment
        5. Installs Python dependencies if detected

        Args:
            client: Connected Paramiko SSH client (managed by caller)
            target_dir: Optional specific directory name for worktree
            uv_extra: Optional extra group for uv sync (e.g., 'interp')

        Returns:
            Path to deployed worktree on remote instance

        Raises:
            ValueError: If not in a git repository
            RuntimeError: If deployment fails
        """
        # Detect git repo
        repo_name, commit_hash = self.detect_git_repo()

        # Set up remote structure
        bare_repo_path = self.setup_remote_structure(client, repo_name)

        # Push code
        self.push_code(repo_name, commit_hash, bare_repo_path)

        # Create worktree with optional custom directory name
        if target_dir:
            # Create custom worktree path but keep using original job_id for git branch
            worktree_path = f"~/.bifrost/worktrees/{target_dir}"
            job_branch = f"job/{self.job_id}"

            console.print(f"🌳 Creating custom worktree: {worktree_path}")

            # Create worktree manually with custom path
            bare_repo_path = f"~/.bifrost/repos/{repo_name}.git"
            cmd = f"cd {bare_repo_path} && git worktree add {worktree_path} {job_branch}"
            stdin, stdout, stderr = client.exec_command(cmd)
            exit_code = stdout.channel.recv_exit_status()

            if exit_code != 0:
                error = stderr.read().decode()
                raise RuntimeError(f"Failed to create custom worktree: {error}")

            console.print(f"✅ Custom worktree ready at: {worktree_path}")
        else:
            worktree_path = self.create_worktree(client, repo_name)

        # Install dependencies
        bootstrap_cmd = self.detect_bootstrap_command(client, worktree_path, uv_extra)
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
    
    def deploy_and_execute_detached(self, client: paramiko.SSHClient, command: str, env_vars: Optional[Dict[str, str]] = None) -> str:
        """Deploy code and execute command in detached mode, return job ID.

        Args:
            client: Connected Paramiko SSH client (managed by caller)
            command: Command to execute
            env_vars: Optional environment variables
        """
        job_id = self._prepare_detached_job()
        repo_name, commit_hash = self.detect_git_repo()

        job_manager = JobManager(self.ssh_user, self.ssh_host, self.ssh_port)

        try:
            return self._execute_detached_deployment(
                client, job_manager, job_id, repo_name, commit_hash, command, env_vars
            )
        except Exception as e:
            console.print(f"❌ Failed to start detached job: {e}")
            console.print(f"🔍 Job data preserved for debugging: ~/.bifrost/jobs/{job_id}")
            raise
    
    def _prepare_detached_job(self) -> str:
        """Generate job ID and update instance state."""
        job_id = generate_job_id()
        self.job_id = job_id
        console.print(f"🆔 Generated job ID: {job_id}")
        return job_id
    
    def _execute_detached_deployment(
        self,
        client: paramiko.SSHClient,
        job_manager: JobManager,
        job_id: str,
        repo_name: str,
        commit_hash: str,
        command: str,
        env_vars: Optional[Dict[str, str]],
        uv_extra: Optional[str] = None
    ) -> str:
        """Execute the main deployment steps for detached job."""

        # Set up remote environment
        bare_repo_path = self.setup_remote_structure(client, repo_name)
        self.push_code(repo_name, commit_hash, bare_repo_path)
        worktree_path = self.create_worktree(client, repo_name)
        
        # Prepare command with bootstrap, but avoid duplicating if caller already does it
        bootstrap_cmd = self.detect_bootstrap_command(client, worktree_path, uv_extra)
        if any(token in command for token in ["uv sync", "pip install -r", "pip install uv"]):
            console.print("📦 Caller handles dependency install; skipping bootstrap to avoid duplication")
            bootstrap_cmd = ""
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

    def deploy_and_execute_detached_workspace(self, client: paramiko.SSHClient, command: str, env_vars: Optional[Dict[str, str]] = None, job_id: Optional[str] = None) -> str:
        """Deploy code to shared workspace and execute command in detached mode.

        Args:
            client: Connected Paramiko SSH client (managed by caller)
            command: Command to execute
            env_vars: Optional environment variables
            job_id: Optional job ID (generated if not provided)
        """

        if not job_id:
            job_id = generate_job_id()
        console.print(f"🆔 Using job ID: {job_id}")

        # Detect git repo
        repo_name, commit_hash = self.detect_git_repo()

        job_manager = JobManager(self.ssh_user, self.ssh_host, self.ssh_port)

        # Deploy to workspace (shared directory) - reuse existing client
        workspace_path = "~/.bifrost/workspace"
        self._deploy_to_existing_workspace(client, workspace_path, repo_name, commit_hash)

        # Update job wrapper to use workspace directory instead of job-specific worktree
        self._upload_workspace_job_wrapper_script(client)

        # Prepare command with bootstrap
        bootstrap_cmd = self.detect_bootstrap_command(client, workspace_path)
        full_command = f"{bootstrap_cmd}{command}"

        # Set up job execution with workspace path
        try:
            job_manager.create_job_metadata(
                client, job_id, full_command, workspace_path, commit_hash, repo_name
            )

            # Start detached execution with workspace wrapper
            tmux_session = job_manager.start_tmux_session(client, job_id, full_command, env_vars, "workspace_job_wrapper.sh")

            console.print(f"🚀 Job {job_id} started in session {tmux_session}")
            console.print(f"💡 Use 'bifrost jobs logs {self.ssh_user}@{self.ssh_host}:{self.ssh_port} {job_id}' to monitor progress")

            return job_id

        except Exception as e:
            console.print(f"❌ Failed to start detached job: {e}")
            console.print(f"🔍 Job data preserved for debugging: ~/.bifrost/jobs/{job_id}")
            raise

    def _upload_workspace_job_wrapper_script(self, client: paramiko.SSHClient) -> None:
        """Upload job wrapper script that uses workspace directory."""
        
        # Create scripts directory
        stdin, stdout, stderr = client.exec_command("mkdir -p ~/.bifrost/scripts")
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to create scripts directory: {error}")
        
        # Job wrapper script for workspace-based jobs
        wrapper_script = '''#!/bin/bash
# Job execution wrapper for workspace-based jobs
set -euo pipefail

JOB_ID=$1
COMMAND=$2
JOB_DIR=~/.bifrost/jobs/$JOB_ID
WORKSPACE_DIR=~/.bifrost/workspace

# Setup job metadata
echo "running" > "$JOB_DIR/status"
echo "$(date -Iseconds)" > "$JOB_DIR/start_time"
echo $$ > "$JOB_DIR/pid"

# Change to workspace directory
cd "$WORKSPACE_DIR"

# Prologue logs
{
  echo "==== BIFROST JOB START (workspace) ===="
  date -Iseconds
  echo "PWD: $(pwd)"
  echo "Command: $COMMAND"
  echo "Env snapshot (selected):"
  echo "  PATH=$PATH"
  echo "  PYTHONPATH=$PYTHONPATH"
  echo "======================================="
} >> "$JOB_DIR/job.log"

# Run command and capture output
set -x
bash -c "$COMMAND" 2>&1 | tee -a "$JOB_DIR/job.log"
EXIT_CODE=${PIPESTATUS[0]}
set +x

# Update job metadata
echo $EXIT_CODE > "$JOB_DIR/exit_code"
echo "$(date -Iseconds)" > "$JOB_DIR/end_time"

if [ $EXIT_CODE -eq 0 ]; then
  echo "completed" > "$JOB_DIR/status"
  echo "==== JOB COMPLETE (exit=0) ====" >> "$JOB_DIR/job.log"
else
  echo "failed" > "$JOB_DIR/status"
  echo "==== JOB FAILED (exit=$EXIT_CODE) ====" >> "$JOB_DIR/job.log"
fi
'''
        
        # Upload wrapper script
        wrapper_cmd = f"cat > ~/.bifrost/scripts/workspace_job_wrapper.sh << 'EOF'\n{wrapper_script}\nEOF"
        stdin, stdout, stderr = client.exec_command(wrapper_cmd)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to upload workspace job wrapper script: {error}")
        
        # Make script executable
        stdin, stdout, stderr = client.exec_command("chmod +x ~/.bifrost/scripts/workspace_job_wrapper.sh")
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to make wrapper script executable: {error}")
        
        console.print("📋 Uploaded workspace job wrapper script")

    def _deploy_to_existing_workspace(self, client: paramiko.SSHClient, workspace_path: str, repo_name: str, commit_hash: str) -> None:
        """Deploy to workspace using existing SSH client."""
        
        # Set up remote structure
        bare_repo_path = self.setup_remote_structure(client, repo_name)
        
        # Push code to remote
        self.push_code(repo_name, commit_hash, bare_repo_path)
        
        # Create or update workspace
        console.print(f"🏗️  Setting up workspace: {workspace_path}")
        
        # Check if workspace exists
        stdin, stdout, stderr = client.exec_command(f"test -d {workspace_path}")
        workspace_exists = stdout.channel.recv_exit_status() == 0
        
        if workspace_exists:
            # Update existing workspace
            console.print("🔄 Updating existing workspace...")
            stdin, stdout, stderr = client.exec_command(f"cd {workspace_path} && git pull origin main")
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                error = stderr.read().decode()
                console.print(f"⚠️ Git pull failed, will recreate workspace: {error}")
                stdin, stdout, stderr = client.exec_command(f"rm -rf {workspace_path}")
                workspace_exists = False
        
        if not workspace_exists:
            # Create new workspace
            console.print("🆕 Creating new workspace...")
            stdin, stdout, stderr = client.exec_command(f"git clone {bare_repo_path} {workspace_path}")
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                error = stderr.read().decode()
                raise RuntimeError(f"Failed to create workspace: {error}")
        
        console.print(f"✅ Workspace ready at {workspace_path}")
