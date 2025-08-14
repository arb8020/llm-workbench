"""Bifrost CLI - Remote GPU execution commands."""

import typer
from rich.console import Console
from typing import List, Optional
import paramiko
import sys
import re
from .deploy import GitDeployment

app = typer.Typer(help="Bifrost - Remote GPU execution tool")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    ssh_info: Optional[str] = typer.Argument(None, help="SSH connection string (user@host:port)"),
    command: Optional[str] = typer.Argument(None, help="Command to execute remotely"),
    env: Optional[List[str]] = typer.Option(None, "--env", help="Environment variables (KEY=VALUE)"),
    no_deploy: bool = typer.Option(False, "--no-deploy", help="Skip git deployment (legacy mode)"),
    detach: bool = typer.Option(False, "--detach", help="Run job in background (detached mode)"),
):
    """Bifrost - Remote GPU execution with automatic code deployment."""
    
    if ctx.invoked_subcommand is not None:
        # A subcommand was called, let it handle
        return
    
    if ssh_info is None or command is None:
        # No arguments provided, show help
        console.print("🌈 [bold blue]Bifrost[/bold blue] - Remote GPU execution tool")
        console.print("\n[bold]Usage:[/bold]")
        console.print("  bifrost [user@host:port] \"[command]\" [OPTIONS]")
        console.print("\n[bold]Examples:[/bold]")
        console.print("  bifrost root@1.2.3.4:22 \"python train.py\"")
        console.print("  bifrost root@gpu.example.com:2222 \"nvidia-smi\" --no-deploy")
        console.print("\n[bold]Options:[/bold]")
        console.print("  --env KEY=VALUE    Environment variables")
        console.print("  --no-deploy        Skip git deployment")  
        console.print("  --detach           Run job in background")
        console.print("  --help            Show this message")
        return
    
    # Execute the command (same as launch)
    _execute_command(ssh_info, command, env, no_deploy, detach)


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
def launch(
    ssh_info: str = typer.Argument(..., help="SSH connection string (user@host:port)"),
    command: str = typer.Argument(..., help="Command to execute remotely"),
    env: Optional[List[str]] = typer.Option(None, "--env", help="Environment variables (KEY=VALUE)"),
    no_deploy: bool = typer.Option(False, "--no-deploy", help="Skip git deployment (legacy mode)"),
    detach: bool = typer.Option(False, "--detach", help="Run job in background (detached mode)"),
):
    """Launch a command on remote GPU instance with automatic code deployment."""
    _execute_command(ssh_info, command, env, no_deploy, detach)


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