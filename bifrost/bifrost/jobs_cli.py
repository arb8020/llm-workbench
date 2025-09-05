"""Bifrost Jobs CLI - Detached job management commands."""

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text
from typing import List, Optional, Dict
import paramiko
import sys
import json
from datetime import datetime
from pathlib import Path

from .deploy import GitDeployment  
from .job_manager import JobManager, generate_job_id

jobs_app = typer.Typer(help="Manage detached bifrost jobs")
console = Console()


def parse_ssh_info(ssh_info: str):
    """Parse SSH connection string into components."""
    # Handle both "user@host:port" and "ssh -p port user@host" formats
    if ssh_info.startswith("ssh -p "):
        # Format: ssh -p port user@host
        import shlex
        parts = shlex.split(ssh_info)
        if len(parts) >= 4 and parts[0] == "ssh" and parts[1] == "-p":
            port = int(parts[2])
            user_host = parts[3]
            if "@" in user_host:
                user, host = user_host.split("@", 1)
            else:
                user = "root"
                host = user_host
            return user, host, port
        else:
            raise ValueError(f"Invalid SSH command format: {ssh_info}")
    else:
        # Format: user@host:port or user@host (default port 22)
        if "@" not in ssh_info:
            raise ValueError(f"SSH connection string must include username: {ssh_info}")
        
        user_host, *port_parts = ssh_info.split(":")
        user, host = user_host.split("@", 1)
        
        if port_parts:
            try:
                port = int(port_parts[0])
            except ValueError:
                raise ValueError(f"Invalid port number: {port_parts[0]}")
        else:
            port = 22
        
        return user, host, port


@jobs_app.command()
def deploy(
    ssh_connection: str = typer.Argument(..., help="SSH connection string (user@host:port or 'ssh -p port user@host')"),
    command: str = typer.Argument(..., help="Command to execute after deployment"),
    env: Optional[List[str]] = typer.Option(None, "--env", help="Environment variables (KEY=VALUE)"),
    env_file: Optional[str] = typer.Option(None, "--env-file", help="Load environment variables from file"),
    isolated: bool = typer.Option(False, "--isolated", help="Use isolated job-specific worktree"),
):
    """Deploy code and execute command in detached mode with job tracking."""
    
    console.print(f"🌈 Starting detached job deployment on {ssh_connection}")
    console.print(f"Command: {command}")
    
    try:
        # Parse SSH connection info
        user, host, port = parse_ssh_info(ssh_connection)
        
        # Create deployment and job manager
        deployment = GitDeployment(user, host, port)
        job_manager = JobManager(user, host, port)
        
        # Parse environment variables
        env_vars = {}
        if env:
            for env_var in env:
                if "=" not in env_var:
                    raise ValueError(f"Environment variable must be in format KEY=VALUE: {env_var}")
                key, value = env_var.split("=", 1)
                env_vars[key] = value
        
        # Load environment variables from file
        if env_file:
            from dotenv import dotenv_values
            file_env = dotenv_values(env_file)
            env_vars.update(file_env)
            console.print(f"📁 Loaded {len(file_env)} environment variables from {env_file}")
        
        # Generate unique job ID
        job_id = generate_job_id()
        console.print(f"🆔 Generated job ID: {job_id}")
        
        # Deploy and start job
        if isolated:
            # Use job-specific worktree (old behavior)
            job_id_actual = deployment.deploy_and_execute_detached(command, env_vars, job_id)
        else:
            # Use shared workspace but with job tracking
            job_id_actual = deployment.deploy_and_execute_detached_workspace(command, env_vars, job_id)
        
        console.print(f"✅ Job {job_id_actual} started successfully")
        console.print(f"💡 Use 'bifrost jobs status {ssh_connection}' to check job status")
        console.print(f"💡 Use 'bifrost jobs logs {ssh_connection} {job_id_actual}' to view logs")
        
    except Exception as e:
        console.print(f"❌ Failed to start detached job: {e}", style="red")
        sys.exit(1)


@jobs_app.command()
def list(
    ssh_connection: Optional[str] = typer.Argument(None, help="SSH connection string (user@host:port) - shows all instances if not specified"),
):
    """List all detached jobs."""
    
    if ssh_connection:
        # Show jobs for specific instance
        try:
            user, host, port = parse_ssh_info(ssh_connection)
            _show_instance_jobs(user, host, port)
        except Exception as e:
            console.print(f"❌ Failed to connect: {e}", style="red")
            sys.exit(1)
    else:
        # TODO: Show jobs from all known instances (future feature)
        console.print("❌ --all option not implemented yet", style="red")
        console.print("💡 Please specify an SSH connection string")
        sys.exit(1)


@jobs_app.command()
def status(
    ssh_connection: str = typer.Argument(..., help="SSH connection string (user@host:port)"),
    job_id: Optional[str] = typer.Argument(None, help="Specific job ID to check (optional)"),
):
    """Show status of detached jobs."""
    
    try:
        user, host, port = parse_ssh_info(ssh_connection)
        
        if job_id:
            _show_job_status(user, host, port, job_id)
        else:
            _show_instance_jobs(user, host, port)
            
    except Exception as e:
        console.print(f"❌ Failed to check status: {e}", style="red")
        sys.exit(1)


@jobs_app.command()
def logs(
    ssh_connection: str = typer.Argument(..., help="SSH connection string (user@host:port)"),
    job_id: str = typer.Argument(..., help="Job ID to show logs for"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow log output"),
    lines: Optional[int] = typer.Option(None, "-n", "--lines", help="Number of lines to show"),
):
    """Show logs for a specific job."""
    
    try:
        user, host, port = parse_ssh_info(ssh_connection)
        
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        console.print(f"📋 Fetching logs for job {job_id} on {user}@{host}:{port}")
        
        # Connect to remote instance
        client.connect(hostname=host, port=port, username=user)
        
        # Check if job exists
        stdin, stdout, stderr = client.exec_command(f"test -d ~/.bifrost/jobs/{job_id}")
        if stdout.channel.recv_exit_status() != 0:
            console.print(f"❌ Job {job_id} not found", style="red")
            client.close()
            sys.exit(1)
        
        # Get job metadata for context
        stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/metadata.json 2>/dev/null")
        if stdout.channel.recv_exit_status() == 0:
            try:
                metadata = json.loads(stdout.read().decode())
                console.print(f"📄 Job: {metadata.get('command', 'Unknown command')}")
                console.print(f"📅 Started: {metadata.get('start_time', 'Unknown')}")
                console.print("─" * 50)
            except json.JSONDecodeError:
                pass
        
        # Build log command
        log_file = f"~/.bifrost/jobs/{job_id}/job.log"
        
        if follow:
            log_cmd = f"tail -f {log_file}"
        elif lines:
            log_cmd = f"tail -n {lines} {log_file}"
        else:
            log_cmd = f"cat {log_file}"
        
        # Stream logs
        stdin, stdout, stderr = client.exec_command(log_cmd)
        
        if follow:
            # For follow mode, stream output
            try:
                while True:
                    line = stdout.readline()
                    if not line:
                        break
                    print(line.rstrip())
            except KeyboardInterrupt:
                console.print("\n📋 Log following stopped")
        else:
            # For regular mode, print all output
            output = stdout.read().decode()
            if output:
                print(output.rstrip())
            else:
                console.print(f"📝 No logs found for job {job_id}")
        
        client.close()
        
    except Exception as e:
        console.print(f"❌ Failed to fetch logs: {e}", style="red")
        sys.exit(1)


@jobs_app.command()
def kill(
    ssh_connection: str = typer.Argument(..., help="SSH connection string (user@host:port)"),
    job_id: str = typer.Argument(..., help="Job ID to kill"),
    cleanup: bool = typer.Option(False, "--cleanup", help="Also cleanup job files"),
):
    """Kill a running job."""
    
    try:
        user, host, port = parse_ssh_info(ssh_connection)
        
        # Create SSH client and job manager
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        job_manager = JobManager(user, host, port)
        
        console.print(f"🛑 Killing job {job_id} on {user}@{host}:{port}")
        
        # Connect to remote instance
        client.connect(hostname=host, port=port, username=user)
        
        # Check if job exists
        status = job_manager.get_job_status(client, job_id)
        if not status:
            console.print(f"❌ Job {job_id} not found", style="red")
            client.close()
            sys.exit(1)
        
        # Kill tmux session
        tmux_session = f"bifrost_{job_id}"
        stdin, stdout, stderr = client.exec_command(f"tmux kill-session -t {tmux_session} 2>/dev/null || true")
        
        # Update job status
        stdin, stdout, stderr = client.exec_command(f"echo 'killed' > ~/.bifrost/jobs/{job_id}/status")
        stdin, stdout, stderr = client.exec_command(f"echo '$(date -Iseconds)' > ~/.bifrost/jobs/{job_id}/end_time")
        
        console.print(f"✅ Job {job_id} killed")
        
        if cleanup:
            job_manager.cleanup_job(client, job_id, keep_worktree=False)
        
        client.close()
        
    except Exception as e:
        console.print(f"❌ Failed to kill job: {e}", style="red")
        sys.exit(1)


def _show_instance_jobs(user: str, host: str, port: int):
    """Show jobs for a specific instance."""
    console.print(f"📊 Checking jobs on {user}@{host}:{port}")
    
    # Create SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to remote instance
        client.connect(hostname=host, port=port, username=user)
        
        # List all jobs
        stdin, stdout, stderr = client.exec_command("ls -1 ~/.bifrost/jobs/ 2>/dev/null || echo 'NO_JOBS'")
        job_list_output = stdout.read().decode().strip()
        
        if job_list_output == "NO_JOBS" or not job_list_output:
            console.print("📭 No detached jobs found on this instance")
            client.close()
            return
        
        job_ids = job_list_output.split('\n')
        console.print(f"📋 Found {len(job_ids)} job(s)")
        
        # Create table for job display
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Job ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Command", style="white")
        table.add_column("Started", style="dim")
        table.add_column("Duration", style="yellow")
        
        for job_id in job_ids:
            if not job_id.strip():
                continue
                
            # Get job metadata
            stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/metadata.json 2>/dev/null")
            metadata = {}
            if stdout.channel.recv_exit_status() == 0:
                try:
                    metadata = json.loads(stdout.read().decode())
                except json.JSONDecodeError:
                    pass
            
            # Get current status
            stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/status 2>/dev/null")
            status = "unknown"
            if stdout.channel.recv_exit_status() == 0:
                status = stdout.read().decode().strip()
            
            # Get end time if available
            stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/end_time 2>/dev/null")
            end_time = None
            if stdout.channel.recv_exit_status() == 0:
                end_time = stdout.read().decode().strip()
            
            # Calculate duration
            start_time = metadata.get('start_time', '')
            duration = _calculate_duration(start_time, end_time)
            
            # Format command (truncate if too long)
            command = metadata.get('command', 'Unknown')
            if len(command) > 40:
                command = command[:37] + "..."
            
            # Format start time
            start_display = start_time[:19].replace('T', ' ') if start_time else 'Unknown'
            
            # Color status
            if status == "completed":
                status_text = Text("✅ completed", style="green")
            elif status == "failed":
                status_text = Text("❌ failed", style="red")
            elif status == "running":
                status_text = Text("🔄 running", style="blue")
            elif status == "killed":
                status_text = Text("🛑 killed", style="yellow")
            else:
                status_text = Text(f"❓ {status}", style="dim")
            
            table.add_row(
                job_id,
                status_text,
                command,
                start_display,
                duration
            )
        
        console.print(table)
        client.close()
        
    except Exception as e:
        console.print(f"❌ Failed to connect to {user}@{host}:{port}: {e}", style="red")
        if client:
            client.close()


def _show_job_status(user: str, host: str, port: int, job_id: str):
    """Show detailed status for a specific job."""
    console.print(f"📊 Checking job {job_id} on {user}@{host}:{port}")
    
    # Create SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to remote instance
        client.connect(hostname=host, port=port, username=user)
        
        # Check if job exists
        stdin, stdout, stderr = client.exec_command(f"test -d ~/.bifrost/jobs/{job_id}")
        if stdout.channel.recv_exit_status() != 0:
            console.print(f"❌ Job {job_id} not found", style="red")
            client.close()
            return
        
        # Get job metadata
        stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/metadata.json 2>/dev/null")
        metadata = {}
        if stdout.channel.recv_exit_status() == 0:
            try:
                metadata = json.loads(stdout.read().decode())
            except json.JSONDecodeError:
                console.print("⚠️ Could not parse job metadata", style="yellow")
        
        # Get current status
        stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/status 2>/dev/null")
        status = "unknown"
        if stdout.channel.recv_exit_status() == 0:
            status = stdout.read().decode().strip()
        
        # Get exit code if available
        stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/exit_code 2>/dev/null")
        exit_code = None
        if stdout.channel.recv_exit_status() == 0:
            try:
                exit_code = int(stdout.read().decode().strip())
            except ValueError:
                pass
        
        # Display job details
        console.print(f"\n📄 Job Details: {job_id}")
        console.print("─" * 50)
        console.print(f"Command: {metadata.get('command', 'Unknown')}")
        console.print(f"Status: {status}")
        if exit_code is not None:
            console.print(f"Exit Code: {exit_code}")
        console.print(f"Started: {metadata.get('start_time', 'Unknown')}")
        if metadata.get('end_time'):
            console.print(f"Ended: {metadata.get('end_time')}")
        console.print(f"Git Commit: {metadata.get('git_commit', 'Unknown')[:8]}")
        console.print(f"Repo: {metadata.get('repo_name', 'Unknown')}")
        console.print(f"Worktree: {metadata.get('worktree_path', 'Unknown')}")
        console.print(f"tmux Session: {metadata.get('tmux_session', 'Unknown')}")
        
        client.close()
        
    except Exception as e:
        console.print(f"❌ Failed to get job status: {e}", style="red")
        if client:
            client.close()


def _calculate_duration(start_time: str, end_time: Optional[str]) -> str:
    """Calculate duration between start and end times."""
    if not start_time:
        return "Unknown"
    
    try:
        from datetime import datetime
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        
        if end_time:
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        else:
            end = datetime.now(start.tzinfo)
        
        duration = end - start
        
        # Format duration nicely
        total_seconds = int(duration.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
            
    except (ValueError, TypeError):
        return "Unknown"