"""Bifrost CLI - Remote GPU execution commands."""

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text
from typing import List, Optional
import paramiko
import sys
import re
import json
from datetime import datetime
from .deploy import GitDeployment
from .job_manager import JobManager

app = typer.Typer(help="Bifrost - Remote GPU execution tool")
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
):
    """Bifrost - Remote GPU execution with automatic code deployment."""
    pass


def _execute_command(ssh_info: str, command: str, env: Optional[List[str]], no_deploy: bool, detach: bool = False):
    """Execute command with or without deployment."""
    console.print(f"🌈 Launching command on {ssh_info}")
    console.print(f"Command: {command}")
    
    try:
        # Parse SSH connection info
        user, host, port = parse_ssh_info(ssh_info)
        
        if no_deploy:
            # Legacy mode - just execute command without deployment
            console.print("⚠️  Legacy mode: skipping code deployment")
            if detach:
                console.print("❌ --detach not supported in legacy mode (--no-deploy)")
                sys.exit(1)
            _execute_legacy(user, host, port, command, env)
        else:
            # Git deployment mode
            console.print("📦 Git deployment mode enabled")
            deployment = GitDeployment(user, host, port)
            
            if detach:
                # Detached execution - start job and return
                console.print("🔄 Starting detached job...")
                job_id = deployment.deploy_and_execute_detached(command, env)
                console.print(f"✅ Job {job_id} started successfully")
                console.print(f"💡 Job will continue running even if SSH disconnects")
                return  # Don't wait for completion
            else:
                # Immediate execution
                exit_code = deployment.deploy_and_execute(command, env)
                
                if exit_code == 0:
                    console.print("✅ Command completed successfully")
                else:
                    console.print(f"❌ Command failed with exit code {exit_code}", style="red")
                    sys.exit(exit_code)
            
    except Exception as e:
        console.print(f"❌ Execution failed: {e}", style="red")
        sys.exit(1)

def parse_ssh_info(ssh_info: str):
    """Parse SSH connection string into components."""
    # Format: user@host:port or user@host (default port 22)
    match = re.match(r'^([^@]+)@([^:]+)(?::(\d+))?$', ssh_info)
    if not match:
        raise ValueError(f"Invalid SSH connection string: {ssh_info}")
    
    user, host, port = match.groups()
    port = int(port) if port else 22
    return user, host, port

@app.command()
def run(
    ssh_info: str = typer.Argument(..., help="SSH connection string (user@host:port)"),
    command: str = typer.Argument(..., help="Command to execute remotely"),
    env: Optional[List[str]] = typer.Option(None, "--env", help="Environment variables (KEY=VALUE)"),
    no_deploy: bool = typer.Option(False, "--no-deploy", help="Skip git deployment (legacy mode)"),
    detach: bool = typer.Option(False, "--detach", help="Run job in background (detached mode)"),
):
    """Run a command on remote GPU instance with automatic code deployment."""
    _execute_command(ssh_info, command, env, no_deploy, detach)

@app.command()
def launch(
    ssh_info: str = typer.Argument(..., help="SSH connection string (user@host:port)"),
    command: str = typer.Argument(..., help="Command to execute remotely"),
    env: Optional[List[str]] = typer.Option(None, "--env", help="Environment variables (KEY=VALUE)"),
    no_deploy: bool = typer.Option(False, "--no-deploy", help="Skip git deployment (legacy mode)"),
    detach: bool = typer.Option(False, "--detach", help="Run job in background (detached mode)"),
):
    """Launch a command on remote GPU instance with automatic code deployment (alias for run)."""
    _execute_command(ssh_info, command, env, no_deploy, detach)


@app.command()
def status(
    ssh_info: Optional[str] = typer.Argument(None, help="SSH connection string (user@host:port) or leave empty for all instances"),
    all_instances: bool = typer.Option(False, "--all", help="Show jobs from all known instances"),
):
    """Show status of all detached jobs."""
    try:
        if ssh_info:
            # Show jobs for specific instance
            user, host, port = parse_ssh_info(ssh_info)
            _show_instance_jobs(user, host, port)
        elif all_instances:
            # TODO: Show jobs from all instances (Phase 3 feature)
            console.print("❌ --all option not implemented yet", style="red")
            console.print("💡 Please specify an SSH connection string")
            sys.exit(1)
        else:
            console.print("❌ Please specify SSH connection or use --all", style="red")
            console.print("💡 Usage: bifrost status user@host:port")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"❌ Status check failed: {e}", style="red")
        sys.exit(1)


@app.command()
def logs(
    ssh_info: str = typer.Argument(..., help="SSH connection string (user@host:port)"),
    job_id: str = typer.Argument(..., help="Job ID to show logs for"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow logs in real-time (like tail -f)"),
    lines: int = typer.Option(100, "-n", "--lines", help="Number of lines to show (default: 100)"),
):
    """Show logs for a specific detached job."""
    try:
        user, host, port = parse_ssh_info(ssh_info)
        _show_job_logs(user, host, port, job_id, follow, lines)
    except Exception as e:
        console.print(f"❌ Failed to show logs: {e}", style="red")
        sys.exit(1)


@app.command()
def copy(
    source: str = typer.Argument(..., help="Source path in format user@host:port:remote_path"),
    destination: str = typer.Argument(..., help="Local destination path"),
    recursive: bool = typer.Option(False, "-r", "--recursive", help="Copy directories recursively"),
):
    """Copy files from remote instance to local machine (like scp)."""
    try:
        _copy_files(source, destination, recursive)
    except Exception as e:
        console.print(f"❌ Copy failed: {e}", style="red")
        sys.exit(1)


def _copy_files(source: str, destination: str, recursive: bool):
    """Copy files from remote to local using SCP-like syntax."""
    # Parse source: user@host:port:remote_path
    if ':' not in source:
        raise ValueError("Source must be in format: user@host:port:remote_path")
    
    # Split on the last colon to separate path from connection info
    connection_part, remote_path = source.rsplit(':', 1)
    
    if '@' not in connection_part or ':' not in connection_part:
        raise ValueError("Invalid source format. Use: user@host:port:remote_path")
    
    # Parse connection info
    user, host_port = connection_part.split('@', 1)
    host, port_str = host_port.rsplit(':', 1)
    
    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(f"Invalid port: {port_str}")
    
    console.print(f"📁 Copying from {user}@{host}:{port}:{remote_path}")
    console.print(f"📂 Destination: {destination}")
    
    # Connect to remote instance
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        console.print("🔗 Connecting...")
        client.connect(hostname=host, port=port, username=user, timeout=30)
        
        # Check if source exists and is file or directory
        stdin, stdout, stderr = client.exec_command(f"test -e '{remote_path}'")
        if stdout.channel.recv_exit_status() != 0:
            console.print(f"❌ Remote path not found: {remote_path}", style="red")
            return
        
        # Check if source is directory
        stdin, stdout, stderr = client.exec_command(f"test -d '{remote_path}'")
        is_directory = stdout.channel.recv_exit_status() == 0
        
        if is_directory and not recursive:
            console.print(f"❌ {remote_path} is a directory. Use -r/--recursive flag", style="red")
            return
        
        # Create SFTP client for file transfer
        sftp = client.open_sftp()
        
        if is_directory:
            _copy_directory(sftp, client, remote_path, destination)
        else:
            _copy_file(sftp, remote_path, destination)
        
        console.print("✅ Copy completed successfully", style="green")
        
    except Exception as e:
        console.print(f"❌ Failed to connect or copy: {e}", style="red")
        raise
    finally:
        try:
            sftp.close()
        except:
            pass
        client.close()


def _copy_file(sftp, remote_path: str, local_path: str):
    """Copy a single file via SFTP."""
    import os
    from pathlib import Path
    
    # Get remote file size for progress reporting
    try:
        remote_stat = sftp.stat(remote_path)
        file_size = remote_stat.st_size
        console.print(f"📄 File size: {file_size:,} bytes")
    except:
        file_size = 0
    
    # Ensure local directory exists
    local_dir = Path(local_path).parent
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy file with progress
    console.print(f"⬇️  Downloading: {remote_path} -> {local_path}")
    
    # Simple progress callback
    def progress_callback(transferred, total):
        if total > 0:
            percent = (transferred / total) * 100
            console.print(f"   Progress: {transferred:,} / {total:,} bytes ({percent:.1f}%)")
    
    # Use progress callback for large files (>1MB)
    if file_size > 1024 * 1024:
        sftp.get(remote_path, local_path, callback=progress_callback)
    else:
        sftp.get(remote_path, local_path)
    
    console.print(f"✅ Downloaded: {local_path}")


def _copy_directory(sftp, client, remote_path: str, local_path: str):
    """Copy a directory recursively via SFTP."""
    import os
    from pathlib import Path
    
    console.print(f"📁 Copying directory: {remote_path}")
    
    # Get directory listing
    stdin, stdout, stderr = client.exec_command(f"find '{remote_path}' -type f")
    if stdout.channel.recv_exit_status() != 0:
        error = stderr.read().decode()
        raise RuntimeError(f"Failed to list directory contents: {error}")
    
    file_list = stdout.read().decode().strip().split('\n')
    file_list = [f for f in file_list if f.strip()]  # Remove empty lines
    
    if not file_list:
        console.print(f"📭 Directory is empty: {remote_path}")
        return
    
    console.print(f"📊 Found {len(file_list)} files to copy")
    
    # Copy each file
    for i, remote_file in enumerate(file_list, 1):
        # Calculate relative path
        rel_path = os.path.relpath(remote_file, remote_path)
        local_file = os.path.join(local_path, rel_path)
        
        console.print(f"[{i}/{len(file_list)}] {rel_path}")
        
        # Ensure local directory exists
        local_dir = Path(local_file).parent
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        try:
            sftp.get(remote_file, local_file)
        except Exception as e:
            console.print(f"⚠️  Failed to copy {rel_path}: {e}", style="yellow")
    
    console.print(f"📁 Directory copy complete: {local_path}")


def _show_job_logs(user: str, host: str, port: int, job_id: str, follow: bool, lines: int):
    """Show logs for a specific job."""
    console.print(f"📄 Getting logs for job {job_id} on {user}@{host}:{port}")
    
    # Connect to remote instance
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        console.print(f"🔗 Connecting...")
        client.connect(hostname=host, port=port, username=user, timeout=30)
        
        # Check if job exists
        stdin, stdout, stderr = client.exec_command(f"test -d ~/.bifrost/jobs/{job_id}")
        if stdout.channel.recv_exit_status() != 0:
            console.print(f"❌ Job {job_id} not found", style="red")
            return
        
        # Get job metadata for context
        stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/metadata.json 2>/dev/null")
        metadata_output = stdout.read().decode().strip()
        if metadata_output:
            try:
                metadata = json.loads(metadata_output)
                command = metadata.get('command', 'N/A')
                console.print(f"Command: {command}")
            except:
                pass
        
        log_file = f"~/.bifrost/jobs/{job_id}/job.log"
        
        if follow:
            # Follow mode - tail -f like behavior
            console.print(f"Following logs for job {job_id} (Ctrl+C to exit)...")
            console.print("=" * 60)
            
            try:
                # Use tail -f to follow the log file
                tail_cmd = f"tail -f {log_file}"
                stdin, stdout, stderr = client.exec_command(tail_cmd)
                
                # Stream output in real-time
                for line in iter(stdout.readline, ""):
                    print(line.rstrip())
                    
            except KeyboardInterrupt:
                console.print("\n📋 Log following stopped")
                return
        else:
            # Show last N lines
            console.print(f"Last {lines} lines of job {job_id}:")
            console.print("=" * 60)
            
            # Check if log file exists
            stdin, stdout, stderr = client.exec_command(f"test -f {log_file}")
            if stdout.channel.recv_exit_status() != 0:
                console.print("📭 No log file found (job may not have started yet)")
                return
            
            # Get last N lines
            tail_cmd = f"tail -n {lines} {log_file}"
            stdin, stdout, stderr = client.exec_command(tail_cmd)
            
            # Show output
            log_content = stdout.read().decode()
            if log_content.strip():
                print(log_content)
            else:
                console.print("📭 Log file is empty")
        
    except Exception as e:
        console.print(f"❌ Failed to connect or retrieve logs: {e}", style="red")
        raise
    finally:
        client.close()


def _show_instance_jobs(user: str, host: str, port: int):
    """Show jobs for a specific instance."""
    console.print(f"📊 Checking jobs on {user}@{host}:{port}")
    
    # Connect to remote instance
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        console.print(f"🔗 Connecting...")
        client.connect(hostname=host, port=port, username=user, timeout=30)
        
        # Get list of job directories
        stdin, stdout, stderr = client.exec_command("ls -1 ~/.bifrost/jobs/ 2>/dev/null || echo 'NO_JOBS'")
        job_dirs_output = stdout.read().decode().strip()
        
        if job_dirs_output == 'NO_JOBS' or not job_dirs_output:
            console.print("📭 No detached jobs found on this instance")
            return
        
        job_dirs = [d for d in job_dirs_output.split('\n') if d.strip()]
        console.print(f"🔍 Found {len(job_dirs)} job(s)")
        
        # Create rich table
        table = Table(title=f"Jobs on {user}@{host}:{port}")
        table.add_column("Job ID", style="cyan", width=10)
        table.add_column("Status", width=12)
        table.add_column("Runtime", width=12)
        table.add_column("Command", style="dim", max_width=40)
        table.add_column("Started", style="dim", width=12)
        
        for job_id in job_dirs:
            job_data = _get_job_data(client, job_id)
            if job_data:
                # Format status with colors
                status = job_data.get('status', 'unknown')
                if status == 'completed':
                    status_text = Text('completed', style='green')
                elif status == 'failed':
                    status_text = Text('failed', style='red')
                elif status == 'running':
                    status_text = Text('running', style='yellow')
                else:
                    status_text = Text(status, style='dim')
                
                # Calculate runtime
                runtime = _calculate_runtime(job_data)
                
                # Truncate command if too long
                command = job_data.get('command', 'N/A')
                if len(command) > 40:
                    command = command[:37] + '...'
                
                # Format start time
                start_time = _format_start_time(job_data.get('start_time'))
                
                table.add_row(
                    job_id,
                    status_text,
                    runtime,
                    command,
                    start_time
                )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Failed to connect: {e}", style="red")
        raise
    finally:
        client.close()


def _get_job_data(client: paramiko.SSHClient, job_id: str) -> dict:
    """Get job metadata from remote instance."""
    try:
        # Get metadata.json
        stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/metadata.json 2>/dev/null")
        metadata_output = stdout.read().decode().strip()
        
        if not metadata_output:
            return {}
        
        metadata = json.loads(metadata_output)
        
        # Get current status (might be different from metadata)
        stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/status 2>/dev/null")
        current_status = stdout.read().decode().strip()
        if current_status:
            metadata['status'] = current_status
        
        # Get end time if available
        stdin, stdout, stderr = client.exec_command(f"cat ~/.bifrost/jobs/{job_id}/end_time 2>/dev/null")
        end_time = stdout.read().decode().strip()
        if end_time:
            metadata['end_time'] = end_time
        
        return metadata
        
    except Exception as e:
        console.print(f"⚠️  Failed to get data for job {job_id}: {e}", style="yellow")
        return {}


def _calculate_runtime(job_data: dict) -> str:
    """Calculate job runtime."""
    try:
        start_time_str = job_data.get('start_time')
        if not start_time_str:
            return 'N/A'
        
        # Parse start time (handle both ISO formats)
        start_time_str = start_time_str.replace('Z', '+00:00')
        start_time = datetime.fromisoformat(start_time_str)
        
        # Get end time or use current time
        end_time_str = job_data.get('end_time')
        if end_time_str:
            end_time_str = end_time_str.replace('Z', '+00:00')
            end_time = datetime.fromisoformat(end_time_str)
        else:
            end_time = datetime.now().astimezone()
        
        # Calculate duration
        duration = end_time - start_time
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m{seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h{minutes}m"
            
    except Exception as e:
        return 'N/A'


def _format_start_time(start_time_str: str) -> str:
    """Format start time for display."""
    try:
        if not start_time_str:
            return 'N/A'
        
        # Parse ISO timestamp
        start_time_str = start_time_str.replace('Z', '+00:00')
        start_time = datetime.fromisoformat(start_time_str)
        
        # Convert to local time and format
        local_time = start_time.astimezone()
        return local_time.strftime('%H:%M:%S')
        
    except Exception:
        return start_time_str[:8] if start_time_str else 'N/A'


def _execute_legacy(user: str, host: str, port: int, command: str, env: Optional[List[str]]):
    """Legacy execution mode without git deployment."""
    # Create SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Build command with environment variables
        full_command = command
        if env:
            env_vars = " ".join(f"{var}" for var in env)
            full_command = f"export {env_vars}; {command}"
            console.print(f"Environment: {env}")
        
        # Connect and execute
        console.print(f"Connecting to {user}@{host}:{port}")
        client.connect(hostname=host, port=port, username=user)
        
        console.print(f"Executing: {full_command}")
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
        
        if exit_code == 0:
            console.print("✅ Command completed successfully")
        else:
            console.print(f"❌ Command failed with exit code {exit_code}", style="red")
            sys.exit(exit_code)
            
    finally:
        client.close()

if __name__ == "__main__":
    app()