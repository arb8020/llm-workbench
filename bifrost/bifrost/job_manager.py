"""Job management for Bifrost detached execution."""

import json
import time
import uuid
import logging
import shlex
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import paramiko
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def generate_job_id() -> str:
    """Generate unique 8-character job ID."""
    # Use timestamp (4 chars) + random (4 chars) for uniqueness
    timestamp = int(time.time())
    random_part = str(uuid.uuid4()).replace('-', '')[:4]
    return f"{timestamp:x}"[-4:] + random_part


class JobManager:
    """Manages detached jobs on remote instances."""
    
    def __init__(self, ssh_user: str, ssh_host: str, ssh_port: int):
        self.ssh_user = ssh_user
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        
    def create_job_metadata(self, client: paramiko.SSHClient, job_id: str, 
                           command: str, worktree_path: str, git_commit: str, 
                           repo_name: str) -> Dict[str, Any]:
        """Create job metadata on remote instance."""
        
        metadata = {
            "job_id": job_id,
            "command": command,
            "ssh_info": f"{self.ssh_user}@{self.ssh_host}:{self.ssh_port}",
            "status": "starting",
            "start_time": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "end_time": None,
            "exit_code": None,
            "tmux_session": f"bifrost_{job_id}",
            "worktree_path": worktree_path,
            "git_commit": git_commit,
            "repo_name": repo_name
        }
        
        # Create job directory
        job_dir = f"~/.bifrost/jobs/{job_id}"
        stdin, stdout, stderr = client.exec_command(f"mkdir -p {job_dir}")
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to create job directory: {error}")
        
        # Write metadata.json
        metadata_json = json.dumps(metadata, indent=2)
        metadata_cmd = f"cat > {job_dir}/metadata.json << 'EOF'\n{metadata_json}\nEOF"
        stdin, stdout, stderr = client.exec_command(metadata_cmd)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to write job metadata: {error}")
        
        console.print(f"📄 Created job metadata: {job_dir}/metadata.json")
        return metadata
    
    def upload_job_wrapper_script(self, client: paramiko.SSHClient) -> None:
        """Upload job wrapper script to remote instance."""
        
        # Create scripts directory
        stdin, stdout, stderr = client.exec_command("mkdir -p ~/.bifrost/scripts")
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to create scripts directory: {error}")
        
        # Job wrapper script content
        wrapper_script = '''#!/bin/bash
# Job execution wrapper with logging and status tracking

JOB_ID=$1
COMMAND=$2
JOB_DIR=~/.bifrost/jobs/$JOB_ID
WORKTREE_DIR=~/.bifrost/worktrees/$JOB_ID

# Setup job metadata
echo "running" > $JOB_DIR/status
echo "$(date -Iseconds)" > $JOB_DIR/start_time
echo $$ > $JOB_DIR/pid

# Change to worktree directory
cd $WORKTREE_DIR

# Execute command with logging
echo "Starting job $JOB_ID in $(pwd)" >> $JOB_DIR/job.log
echo "Command: $COMMAND" >> $JOB_DIR/job.log
echo "----------------------------------------" >> $JOB_DIR/job.log

# Run command and capture exit code
bash -c "$COMMAND" 2>&1 | tee -a $JOB_DIR/job.log
EXIT_CODE=${PIPESTATUS[0]}

# Update job metadata
echo $EXIT_CODE > $JOB_DIR/exit_code
echo "$(date -Iseconds)" > $JOB_DIR/end_time

if [ $EXIT_CODE -eq 0 ]; then
    echo "completed" > $JOB_DIR/status
    echo "Job $JOB_ID completed successfully" >> $JOB_DIR/job.log
else
    echo "failed" > $JOB_DIR/status
    echo "Job $JOB_ID failed with exit code $EXIT_CODE" >> $JOB_DIR/job.log
fi
'''
        
        # Upload wrapper script
        wrapper_cmd = f"cat > ~/.bifrost/scripts/job_wrapper.sh << 'EOF'\n{wrapper_script}\nEOF"
        stdin, stdout, stderr = client.exec_command(wrapper_cmd)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to upload job wrapper script: {error}")
        
        # Make script executable
        stdin, stdout, stderr = client.exec_command("chmod +x ~/.bifrost/scripts/job_wrapper.sh")
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to make wrapper script executable: {error}")
        
        console.print("📋 Uploaded job wrapper script")
    
    def start_tmux_session(self, client: paramiko.SSHClient, job_id: str, 
                          command: str, env_vars: Optional[Dict[str, str]] = None, 
                          wrapper_script: str = "job_wrapper.sh") -> str:
        """Start tmux session for detached job execution."""
        
        tmux_session = f"bifrost_{job_id}"
        
        # Build the command - use double quotes to avoid nested single quote issues
        wrapper_cmd = f'~/.bifrost/scripts/{wrapper_script} {job_id} "{command}"'
        
        # Add environment variables if provided
        if env_vars:
            env_setup = " && ".join(f'export {k}={shlex.quote(v)}' for k, v in env_vars.items())
            wrapper_cmd = f"{env_setup} && {wrapper_cmd}"
        
        # Start tmux session - use single quotes to wrap the entire command
        tmux_cmd = f"tmux new-session -d -s {tmux_session} '{wrapper_cmd}'"
        
        console.print(f"🖥️  Starting tmux session: {tmux_session}")
        stdin, stdout, stderr = client.exec_command(tmux_cmd)
        exit_code = stdout.channel.recv_exit_status()
        
        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to start tmux session: {error}")
        
        # For fast-completing jobs, we don't need to verify the tmux session is still running
        # Instead, we verify the job was started by checking the job metadata was created
        job_dir = f"~/.bifrost/jobs/{job_id}"
        
        # Give the job a moment to start and create its status file
        time.sleep(1)
        
        # Check if job started by looking for the status file created by the wrapper
        status_check_cmd = f"test -f {job_dir}/status || test -f {job_dir}/job.log"
        stdin, stdout, stderr = client.exec_command(status_check_cmd)
        status_exists = stdout.channel.recv_exit_status() == 0
        
        if not status_exists:
            # Job hasn't started yet, wait a bit more
            console.print("⏳ Job files not found immediately, waiting...")
            time.sleep(2)
            stdin, stdout, stderr = client.exec_command(status_check_cmd)
            status_exists = stdout.channel.recv_exit_status() == 0
        
        if not status_exists:
            # Check tmux sessions to see what's available for debugging
            verify_cmd = "tmux list-sessions 2>/dev/null || echo 'No sessions'"
            stdin, stdout, stderr = client.exec_command(verify_cmd)
            sessions_output = stdout.read().decode()
            console.print(f"🐛 Debug - tmux sessions: {sessions_output}")
            
            # Also check if the job directory was created
            stdin, stdout, stderr = client.exec_command(f"ls -la {job_dir}/ 2>/dev/null || echo 'Job dir not found'")
            job_dir_output = stdout.read().decode()
            console.print(f"🐛 Debug - job directory: {job_dir_output}")
            
            raise RuntimeError(f"Job {job_id} failed to start. No job status or log files were created.")
        
        console.print(f"✅ Job {job_id} started successfully in tmux session {tmux_session}")
        return tmux_session
    
    def check_job_running(self, client: paramiko.SSHClient, job_id: str) -> bool:
        """Check if job is currently running."""
        
        tmux_session = f"bifrost_{job_id}"
        verify_cmd = f"tmux list-sessions | grep {tmux_session}"
        
        stdin, stdout, stderr = client.exec_command(verify_cmd)
        exit_code = stdout.channel.recv_exit_status()
        
        return exit_code == 0
    
    def get_job_status(self, client: paramiko.SSHClient, job_id: str) -> Optional[str]:
        """Get current job status from remote metadata."""
        
        status_cmd = f"cat ~/.bifrost/jobs/{job_id}/status 2>/dev/null || echo 'not_found'"
        stdin, stdout, stderr = client.exec_command(status_cmd)
        
        status = stdout.read().decode().strip()
        return status if status != 'not_found' else None
    
    def wait_for_job_completion(self, client: paramiko.SSHClient, job_id: str, 
                               timeout: int = 300, check_interval: int = 2) -> str:
        """Wait for job to complete, return final status."""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(client, job_id)
            
            if status in ['completed', 'failed', 'killed']:
                return status
            
            time.sleep(check_interval)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def cleanup_job(self, client: paramiko.SSHClient, job_id: str, 
                   keep_worktree: bool = False) -> None:
        """Clean up job resources."""
        
        # Kill tmux session if still running
        tmux_session = f"bifrost_{job_id}"
        stdin, stdout, stderr = client.exec_command(f"tmux kill-session -t {tmux_session} 2>/dev/null || true")
        
        # Remove job directory
        stdin, stdout, stderr = client.exec_command(f"rm -rf ~/.bifrost/jobs/{job_id}")
        
        # Remove worktree (optional)
        if not keep_worktree:
            # Remove worktree from git
            stdin, stdout, stderr = client.exec_command(
                f"cd ~/.bifrost/repos/*.git && git worktree remove ~/.bifrost/worktrees/{job_id} --force 2>/dev/null || true"
            )
            # Remove job branch
            stdin, stdout, stderr = client.exec_command(
                f"cd ~/.bifrost/repos/*.git && git branch -D job/{job_id} 2>/dev/null || true"
            )
        
        console.print(f"🧹 Cleaned up job {job_id}")