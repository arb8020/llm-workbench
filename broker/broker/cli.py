"""
Command-line interface for GPU cloud operations
"""

import sys
import json
import typer
import subprocess
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional

from . import search, get_instance, terminate_instance, create, list_instances
from .types import CloudType
from .ssh_clients import start_interactive_ssh_session, execute_command_streaming

app = typer.Typer(help="GPU cloud broker CLI")
console = Console()


@app.command()
def search_gpus(
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type", help="GPU type to search for"),
    max_price: Optional[float] = typer.Option(None, "--max-price", help="Maximum price per hour"),
    min_vram: Optional[int] = typer.Option(None, "--min-vram", help="Minimum VRAM in GB"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Specific provider to search"),
    cloud_type: Optional[str] = typer.Option(None, "--cloud-type", help="Cloud type: 'secure' or 'community'"),
    cuda_version: Optional[str] = typer.Option(None, "--cuda-version", help="CUDA version filter (e.g., '12.0', '11.8')"),
    sort_by: Optional[str] = typer.Option("price", "--sort", help="Sort by: 'price', 'memory', 'value' (memory/price)"),
    reverse: bool = typer.Option(False, "--reverse", help="Sort in descending order"),
    analysis: bool = typer.Option(False, "--analysis", help="Show availability analysis by cloud type")
):
    """Search for available GPU offers"""
    console.print("🔍 Searching for GPU offers...")
    
    # Show analysis if requested
    if analysis:
        show_availability_analysis()
        return
    
    # Build pandas-style query using client interface to avoid TypeError
    from .client import GPUClient
    client = GPUClient()
    
    query_conditions = []
    
    if max_price:
        query_conditions.append(client.price_per_hour < max_price)
    if gpu_type:
        query_conditions.append(client.gpu_type.contains(gpu_type))
    if min_vram:
        query_conditions.append(client.memory_gb >= min_vram)
    if cloud_type:
        cloud_enum = CloudType.SECURE if cloud_type == "secure" else CloudType.COMMUNITY
        query_conditions.append(client.cloud_type == cloud_enum)
    
    # Determine sort function
    sort_func = None
    if sort_by == "memory":
        sort_func = lambda x: x.memory_gb
    elif sort_by == "value":
        sort_func = lambda x: x.memory_gb / x.price_per_hour
    # Default is price, which is handled automatically
    
    # Execute query using client
    if query_conditions:
        combined_query = query_conditions[0]
        for condition in query_conditions[1:]:
            combined_query = combined_query & condition
        offers = client.search(query=combined_query, cuda_version=cuda_version, sort=sort_func, reverse=reverse)
    else:
        offers = client.search(cuda_version=cuda_version, sort=sort_func, reverse=reverse)
    
    if not offers:
        console.print("❌ No GPU offers found matching your criteria")
        return
    
    # Display results in a table with cloud type
    table = Table(title=f"Available GPU Offers ({len(offers)} found)")
    table.add_column("Cloud", style="cyan")
    table.add_column("GPU Type", style="green")  
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("CUDA", justify="center", style="magenta")
    table.add_column("Price/Hr ($)", justify="right", style="yellow")
    table.add_column("Provider", style="blue")
    
    for offer in offers:
        cloud_icon = "🔒 Secure" if offer.cloud_type == CloudType.SECURE else "🌐 Community"
        cuda_display = offer.cuda_version if offer.cuda_version else "N/A"
        table.add_row(
            cloud_icon,
            offer.gpu_type,
            str(offer.memory_gb),
            cuda_display,
            f"{offer.price_per_hour:.3f}",
            offer.provider
        )
    
    console.print(table)
    console.print("\n💡 Tips:")
    console.print("   --sort value    # Best memory/price ratio")
    console.print("   --sort memory   # Highest VRAM")
    console.print("   --analysis      # Availability patterns")


def show_availability_analysis():
    """Show detailed availability analysis by cloud type"""
    console.print("🔍 [bold]GPU Availability Analysis by Cloud Type[/bold]\n")
    
    all_offers = search()
    secure_offers = [o for o in all_offers if o.cloud_type == CloudType.SECURE]
    community_offers = [o for o in all_offers if o.cloud_type == CloudType.COMMUNITY]
    
    # Secure cloud table
    secure_table = Table(title=f"🔒 Secure Cloud ({len(secure_offers)} GPUs) - Better availability, higher prices")
    secure_table.add_column("GPU Type", style="green")
    secure_table.add_column("VRAM (GB)", justify="right")
    secure_table.add_column("Price/Hr ($)", justify="right", style="yellow")
    
    for offer in sorted(secure_offers, key=lambda x: x.price_per_hour):
        secure_table.add_row(offer.gpu_type, str(offer.memory_gb), f"{offer.price_per_hour:.3f}")
    
    console.print(secure_table)
    
    # Community cloud table (show top 10)
    community_table = Table(title=f"🌐 Community Cloud (showing cheapest 10 of {len(community_offers)}) - Lower prices, limited availability")
    community_table.add_column("GPU Type", style="green")
    community_table.add_column("VRAM (GB)", justify="right") 
    community_table.add_column("Price/Hr ($)", justify="right", style="yellow")
    
    for offer in sorted(community_offers, key=lambda x: x.price_per_hour)[:10]:
        community_table.add_row(offer.gpu_type, str(offer.memory_gb), f"{offer.price_per_hour:.3f}")
    
    console.print(community_table)
    
    # Available in both clouds
    secure_types = set(o.gpu_type for o in secure_offers)
    community_types = set(o.gpu_type for o in community_offers)
    both_clouds = secure_types & community_types
    
    if both_clouds:
        both_table = Table(title=f"✨ Available in Both Clouds ({len(both_clouds)} types) - Best flexibility")
        both_table.add_column("GPU Type", style="green")
        both_table.add_column("🔒 Secure ($)", justify="right", style="yellow")
        both_table.add_column("🌐 Community ($)", justify="right", style="yellow")
        both_table.add_column("Savings (%)", justify="right", style="red")
        
        for gpu_type in sorted(both_clouds):
            secure_price = next(o.price_per_hour for o in secure_offers if o.gpu_type == gpu_type)
            community_price = next(o.price_per_hour for o in community_offers if o.gpu_type == gpu_type)
            savings = ((secure_price - community_price) / secure_price * 100)
            both_table.add_row(
                gpu_type, 
                f"{secure_price:.3f}", 
                f"{community_price:.3f}",
                f"{savings:.0f}%"
            )
        
        console.print(both_table)
    
    console.print("\n[bold yellow]Recommendation:[/bold yellow] Use [bold]--cloud-type secure[/bold] for reliable provisioning")


@app.command() 
def create(
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type", help="GPU type to provision"),
    cloud_type: Optional[str] = typer.Option("secure", "--cloud-type", help="Cloud type: 'secure' (default) or 'community'"),
    max_price: Optional[float] = typer.Option(None, "--max-price", help="Maximum price per hour"),
    min_vram: Optional[int] = typer.Option(None, "--min-vram", help="Minimum VRAM in GB"),
    image: str = typer.Option("runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04", "--image", help="Docker image"),
    name: Optional[str] = typer.Option(None, "--name", help="Instance name"),
    sort_by: Optional[str] = typer.Option("price", "--sort", help="Sort by: 'price', 'memory', 'value' (memory/price)"),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Try up to N offers before giving up"),
    print_ssh: bool = typer.Option(False, "--print-ssh", help="Print SSH connection string when ready (for piping to bifrost)"),
    container_disk: Optional[int] = typer.Option(None, "--container-disk", help="Container disk size in GB (default: 50GB)"),
    volume_disk: Optional[int] = typer.Option(None, "--volume-disk", help="Volume disk size in GB (default: 0GB)")
):
    """Provision a new GPU instance using pandas-style search"""
    if not print_ssh:
        console.print("🚀 Provisioning GPU instance...")
    
    # Build query for provisioning using client interface to avoid TypeError
    from .client import GPUClient
    client = GPUClient()
    
    query_conditions = []
    
    if max_price:
        query_conditions.append(client.price_per_hour < max_price)
    if gpu_type:
        query_conditions.append(client.gpu_type.contains(gpu_type))
    if min_vram:
        query_conditions.append(client.memory_gb >= min_vram)
        if not print_ssh:
            console.print(f"🧠 Filtering for GPUs with at least {min_vram}GB VRAM")
    if cloud_type:
        cloud_enum = CloudType.SECURE if cloud_type == "secure" else CloudType.COMMUNITY
        query_conditions.append(client.cloud_type == cloud_enum)
        if not print_ssh:
            console.print(f"🔒 Using {cloud_type} cloud for {'better availability' if cloud_type == 'secure' else 'lower prices'}")
    
    # Determine sort function
    sort_func = None
    if sort_by == "memory":
        sort_func = lambda x: x.memory_gb
    elif sort_by == "value":
        sort_func = lambda x: x.memory_gb / x.price_per_hour
    
    # Build final query
    final_query = None
    if query_conditions:
        final_query = query_conditions[0]
        for condition in query_conditions[1:]:
            final_query = final_query & condition
    
    # Use enhanced create with search + retry using client
    if not print_ssh:
        console.print(f"🎯 Trying up to {max_attempts} offers sorted by {sort_by}...")
    try:
        # Prepare disk configuration
        create_kwargs = {
            "query": final_query,
            "image": image, 
            "name": name,
            "sort": sort_func,
            "reverse": (sort_by != "price"),  # Descending for memory/value, ascending for price
            "max_attempts": max_attempts
        }
        
        # Add disk configuration if specified
        if container_disk is not None:
            create_kwargs["container_disk_gb"] = container_disk
            if not print_ssh:
                console.print(f"💾 Container disk: {container_disk}GB")
        
        if volume_disk is not None:
            create_kwargs["volume_disk_gb"] = volume_disk
            if not print_ssh:
                console.print(f"💾 Volume disk: {volume_disk}GB")
        
        instance = client.create(**create_kwargs)
    except Exception as e:
        console.print(f"❌ Provisioning failed: {e}")
        return
    
    if instance:
        if print_ssh:
            # Wait for instance to be ready and print SSH connection string
            
            import time
            max_wait = 300  # 5 minutes
            waited = 0
            
            while waited < max_wait:
                # Re-fetch instance status
                current_instance = get_instance(instance.id)
                if current_instance and current_instance.public_ip and current_instance.ssh_port:
                    from .types import InstanceStatus
                    if current_instance.status == InstanceStatus.RUNNING:
                        # Print SSH connection string to stdout for piping
                        print(f"{current_instance.ssh_username}@{current_instance.public_ip}:{current_instance.ssh_port}")
                        return
                
                time.sleep(10)
                waited += 10
                # Skip progress updates in print_ssh mode
            
            console.print("❌ Timeout waiting for SSH to be ready")
            console.print(f"   Instance ID: {instance.id}")
            console.print("   Try: broker getssh <instance_id>")
            sys.exit(1)
        else:
            # Normal output
            console.print(f"✅ Instance created: {instance.id}")
            console.print(f"   Status: {instance.status}")
            console.print(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
            console.print(f"   Price: ${instance.price_per_hour:.3f}/hr")
    else:
        console.print("❌ Failed to provision instance")
        sys.exit(1)


@app.command()
def status(instance_id: str):
    """Get status of a specific instance"""
    console.print(f"📊 Getting status for instance: {instance_id}")
    
    instance = get_instance(instance_id)
    
    if instance:
        console.print(f"✅ Instance found")
        console.print(f"   ID: {instance.id}")
        console.print(f"   Status: {instance.status}")
        console.print(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
        console.print(f"   Price: ${instance.price_per_hour:.3f}/hr")
        
        if instance.public_ip:
            console.print(f"   SSH: {instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}")
    else:
        console.print("❌ Instance not found")


@app.command()
def terminate(instance_id: str):
    """Terminate an instance"""
    console.print(f"🗑️  Terminating instance: {instance_id}")
    
    success = terminate_instance(instance_id)
    
    if success:
        console.print("✅ Instance terminated successfully")
    else:
        console.print("❌ Failed to terminate instance")


@app.command()
def info(
    instance_id: str,
    live_metrics: bool = typer.Option(False, "--live-metrics", help="Include real-time disk usage via SSH"),
    json_output: bool = typer.Option(False, "--json", help="JSON output format for scripting")
):
    """Get comprehensive information about a specific instance"""
    
    if not json_output:
        console.print(f"🔍 Getting detailed information for instance: {instance_id}")
    
    # Get instance details using the enhanced GraphQL query
    from .runpod import get_instance_details_enhanced
    
    try:
        instance_data = get_instance_details_enhanced(instance_id)
        
        if not instance_data:
            if json_output:
                print('{"error": "Instance not found"}')
            else:
                console.print("❌ Instance not found")
            return
        
        if json_output:
            import json
            print(json.dumps(instance_data, indent=2))
            return
        
        # Display comprehensive information
        display_enhanced_instance_info(instance_data, live_metrics, instance_id)
        
    except Exception as e:
        if json_output:
            print(f'{{"error": "Failed to get instance info: {e}"}}')
        else:
            console.print(f"❌ Failed to get instance info: {e}")


@app.command()
def list(
    simple: bool = typer.Option(False, "--simple", help="Simple output format for scripting (id,name,status,ssh)"),
    json_output: bool = typer.Option(False, "--json", help="JSON output format for scripting"),
    name: Optional[str] = typer.Option(None, "--name", help="Filter by instance name"),
    ssh_only: bool = typer.Option(False, "--ssh-only", help="Output only SSH connection strings (one per line)")
):
    """List all user instances"""
    if not simple and not json_output and not ssh_only:
        console.print("📋 Listing all instances...")
    
    instances = list_instances()
    
    # Apply name filter if specified
    if name:
        filtered_instances = []
        for instance in instances:
            instance_name = "N/A"
            if hasattr(instance, 'raw_data') and instance.raw_data and 'name' in instance.raw_data:
                instance_name = instance.raw_data['name']
            if instance_name == name:
                filtered_instances.append(instance)
        instances = filtered_instances
    
    if not instances:
        if not simple and not json_output and not ssh_only:
            console.print("📭 No instances found")
        elif json_output:
            print("[]")
        return
    
    # SSH-only output format for scripting
    if ssh_only:
        for instance in instances:
            # Only output running instances with SSH info
            if str(instance.status).lower() == "instancestatus.running" and instance.public_ip and instance.ssh_port:
                ssh_info = f"{instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}"
                print(ssh_info)
        return
    
    # JSON output format for scripting
    if json_output:
        instances_data = []
        for instance in instances:
            # Get name from raw_data if available
            instance_name = "N/A"
            if hasattr(instance, 'raw_data') and instance.raw_data and 'name' in instance.raw_data:
                instance_name = instance.raw_data['name']
            
            # Format SSH info
            ssh_info = ""
            if instance.public_ip and instance.ssh_port:
                ssh_info = f"{instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}"
            
            # Get clean status
            status = str(instance.status).lower()
            
            instances_data.append({
                "id": instance.id,
                "name": instance_name,
                "status": status,
                "ssh": ssh_info,
                "gpu_type": instance.gpu_type,
                "gpu_count": instance.gpu_count,
                "price_per_hour": instance.price_per_hour,
                "public_ip": instance.public_ip,
                "ssh_port": instance.ssh_port,
                "ssh_username": instance.ssh_username
            })
        
        print(json.dumps(instances_data, indent=2))
        return
    
    # Simple output format for scripting
    if simple:
        for instance in instances:
            # Get name from raw_data if available
            instance_name = "N/A"
            if hasattr(instance, 'raw_data') and instance.raw_data and 'name' in instance.raw_data:
                instance_name = instance.raw_data['name']
            
            # Format SSH info
            ssh_info = ""
            if instance.public_ip and instance.ssh_port:
                ssh_info = f"{instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}"
            
            # Get clean status (without color markup)
            status = str(instance.status).lower()
            
            # Output: id,name,status,ssh
            print(f"{instance.id},{instance_name},{status},{ssh_info}")
        return
    
    # Display results in a table
    table = Table(title=f"Your GPU Instances ({len(instances)} found)")
    table.add_column("ID", style="cyan", max_width=20)
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("GPU", style="magenta")
    table.add_column("Price/Hr", justify="right", style="yellow")
    table.add_column("SSH", style="blue")
    
    for instance in instances:
        # Format status with color
        status = instance.status
        if status == "RUNNING":
            status = "[green]RUNNING[/green]"
        elif status in ["STARTING", "PENDING"]:
            status = "[yellow]STARTING[/yellow]"
        elif status in ["STOPPED", "TERMINATED"]:
            status = "[red]STOPPED[/red]"
        
        # Format GPU info
        gpu_info = f"{instance.gpu_type} x{instance.gpu_count}" if instance.gpu_count > 1 else instance.gpu_type
        
        # Format SSH info
        ssh_info = ""
        if instance.public_ip and instance.ssh_port:
            ssh_info = f"{instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}"
        
        # Format price
        price = f"${instance.price_per_hour:.3f}" if instance.price_per_hour else "N/A"
        
        # Get name from raw_data if available
        instance_name = "N/A"
        if hasattr(instance, 'raw_data') and instance.raw_data and 'name' in instance.raw_data:
            instance_name = instance.raw_data['name']
        
        table.add_row(
            instance.id[:12] + "..." if len(instance.id) > 15 else instance.id,
            instance_name,
            status,
            gpu_info,
            price,
            ssh_info
        )
    
    console.print(table)


@app.command()
def getssh(instance_id: str):
    """Get SSH connection string for an instance (machine readable)"""
    # Get instance details
    instance = get_instance(instance_id)
    if not instance:
        # Try to find by partial ID match
        instances = list_instances()
        matching_instances = [inst for inst in instances if inst.id.startswith(instance_id)]
        if len(matching_instances) == 1:
            instance = matching_instances[0]
        elif len(matching_instances) > 1:
            console.print(f"❌ Multiple instances match '{instance_id}': {[inst.id for inst in matching_instances]}")
            sys.exit(1)
        else:
            console.print("❌ Instance not found")
            sys.exit(1)
    
    # Check if instance is running
    from .types import InstanceStatus
    if instance.status != InstanceStatus.RUNNING:
        console.print(f"❌ Instance is not running (status: {instance.status})")
        sys.exit(1)
    
    # Check if SSH is available
    if not instance.public_ip or not instance.ssh_port:
        console.print("❌ SSH connection info not available")
        sys.exit(1)
    
    # Output just the SSH connection string for piping
    print(f"{instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}")


@app.command()
def ssh(instance_id: str):
    """Start an interactive SSH session to an instance"""
    console.print(f"🔗 Connecting to instance: {instance_id}")
    
    # Get instance details
    instance = get_instance(instance_id)
    if not instance:
        console.print("❌ Instance not found")
        return
    
    # Check if instance is running
    from .types import InstanceStatus
    if instance.status != InstanceStatus.RUNNING:
        console.print(f"❌ Instance is not running (status: {instance.status})")
        return
    
    # Check if SSH is available
    if not instance.public_ip or not instance.ssh_port:
        console.print("❌ SSH connection info not available")
        return
    
    console.print(f"📡 SSH: {instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}")
    
    try:
        # Use client's SSH key configuration if available
        ssh_key_path = None
        try:
            from .client import GPUClient
            client = GPUClient()
            ssh_key_path = client.get_ssh_key_path()
        except:
            pass  # Fall back to SSH agent
        
        start_interactive_ssh_session(instance, private_key=ssh_key_path)
    except Exception as e:
        console.print(f"❌ SSH connection failed: {e}")


@app.command()
def exec(instance_id: str, command: str):
    """Execute a command on an instance with streaming output"""
    console.print(f"🔄 Executing command on instance: {instance_id}")
    
    # Get instance details
    instance = get_instance(instance_id)
    if not instance:
        console.print("❌ Instance not found")
        return
    
    # Check if instance is running
    from .types import InstanceStatus
    if instance.status != InstanceStatus.RUNNING:
        console.print(f"❌ Instance is not running (status: {instance.status})")
        return
    
    # Check if SSH is available
    if not instance.public_ip or not instance.ssh_port:
        console.print("❌ SSH connection info not available")
        return
    
    console.print(f"📡 Target: {instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}")
    
    try:
        # Use client's SSH key configuration if available
        ssh_key_path = None
        try:
            from .client import GPUClient
            client = GPUClient()
            ssh_key_path = client.get_ssh_key_path()
        except:
            pass  # Fall back to SSH agent
        
        success = execute_command_streaming(instance, command, private_key=ssh_key_path)
        if not success:
            console.print("❌ Command execution failed")
    except Exception as e:
        console.print(f"❌ Command execution failed: {e}")


@app.command()
def config():
    """Check configuration and validate setup"""
    console.print("🔧 [bold]GPU Broker Configuration Check[/bold]\n")
    
    try:
        from .client import GPUClient
        client = GPUClient()
        
        validation_results = client.validate_configuration()
        
        for component, status in validation_results.items():
            console.print(f"**{component.replace('_', ' ').title()}**: {status}")
            
        # Test API connection
        console.print("\n🌐 **Connection Test**:")
        try:
            offers = search()
            console.print(f"✅ RunPod API connection successful ({len(offers)} GPU offers available)")
        except Exception as e:
            console.print(f"❌ RunPod API connection failed: {e}")
            
    except Exception as e:
        console.print(f"❌ Configuration error: {e}")
        console.print("\n💡 **Setup Help:**")
        console.print("   1. Copy `.env.example` to `.env` and add your RunPod API key")
        console.print("   2. Ensure SSH key exists at `~/.ssh/id_ed25519` or set GPU_BROKER_SSH_KEY")
        console.print("   3. Upload your SSH public key to RunPod dashboard")


def display_enhanced_instance_info(instance_data: dict, live_metrics: bool, instance_id: str):
    """Display comprehensive instance information in a user-friendly format"""
    
    # Instance header
    name = instance_data.get('name', 'unnamed')
    instance_id_short = instance_id[:12] + "..." if len(instance_id) > 12 else instance_id
    
    console.print(f"\n🚀 [bold cyan]Instance Details: {name} ({instance_id_short})[/bold cyan]\n")
    
    # Status & Performance section
    runtime = instance_data.get('runtime') or {}
    machine = instance_data.get('machine') or {}
    
    uptime_seconds = runtime.get('uptimeInSeconds', 0) if runtime else 0
    uptime_str = format_uptime(uptime_seconds)
    
    status_info = [
        f"Status: [green]RUNNING[/green] ({uptime_str} uptime)",
        f"Cost: ~${(instance_data.get('costPerHr', 0) * (uptime_seconds / 3600)):.2f} ({uptime_seconds / 3600:.1f}h × ${instance_data.get('costPerHr', 0):.3f}/hr)"
    ]
    
    # Add utilization if available
    if machine:
        cpu_util = machine.get('cpuUtilPercent', 0)
        mem_util = machine.get('memoryUtilPercent', 0)
        disk_util = machine.get('diskUtilPercent', 0)
        
        util_info = f"Real-time Utilization:\n"
        util_info += f"• CPU: {cpu_util}% ({instance_data.get('vcpuCount', '?')} vCPUs available)\n"
        util_info += f"• Memory: {mem_util}% ({instance_data.get('memoryInGb', '?')}GB RAM available)\n"
        util_info += f"• Disk: {disk_util}% ({instance_data.get('containerDiskInGb', '?')}GB container)"
        
        # Add warning for high disk usage
        if disk_util > 75:
            util_info += " ⚠️ High usage!"
        
        status_info.append(util_info)
    
    console.print(Panel("\n".join(status_info), title="📊 Status & Performance", box=box.ROUNDED))
    
    # Hardware Configuration section
    hardware_info = [
        f"GPU: {instance_data.get('gpuCount', 1)} × [bold]GPU Type[/bold]", # TODO: Get GPU type from machine data
        f"vCPU: {instance_data.get('vcpuCount', '?')} cores",
        f"Memory: {instance_data.get('memoryInGb', '?')}GB RAM",
        f"Container Disk: {instance_data.get('containerDiskInGb', '?')}GB",
        f"Volume Disk: {instance_data.get('volumeInGb', 0)}GB" + (" (not mounted)" if instance_data.get('volumeInGb', 0) == 0 else "")
    ]
    
    console.print(Panel("\n".join(hardware_info), title="🖥️ Hardware Configuration", box=box.ROUNDED))
    
    # Network & Access section
    ports = runtime.get('ports', []) if runtime else []
    ssh_info = None
    
    for port in ports:
        if port.get('privatePort') == 22 and port.get('isIpPublic'):
            ssh_info = f"root@{port.get('ip')}:{port.get('publicPort')}"
            break
    
    # If no SSH info from runtime, this might be a new instance
    if not ssh_info:
        ssh_info = "Starting up... (check broker status for SSH details)"
    
    network_info = [
        f"SSH: [bold]{ssh_info}[/bold]" if ssh_info else "SSH: Not available",
        f"Direct SSH: {'✅ Available' if ssh_info and not 'ssh.runpod.io' in str(ssh_info) else '❌ Proxy only'}",
        f"Exposed Ports: {instance_data.get('ports', 'Default')}"
    ]
    
    console.print(Panel("\n".join(network_info), title="🌐 Network & Access", box=box.ROUNDED))
    
    # Container Details section
    container_info = [
        f"Image: {instance_data.get('imageName', 'Unknown')}",
        f"Machine ID: {instance_data.get('machineId', 'Unknown')}",
        f"Host ID: {machine.get('podHostId', 'Unknown') if machine else 'Unknown'}"
    ]
    
    console.print(Panel("\n".join(container_info), title="🐳 Container Details", box=box.ROUNDED))
    
    # Live disk usage if requested
    if live_metrics and ssh_info:
        console.print(f"\n🔍 [yellow]Fetching live disk usage via SSH...[/yellow]")
        try:
            result = subprocess.run([
                "bifrost", "launch", ssh_info, "df -h / && echo && du -sh /root/.cache /root/.bifrost 2>/dev/null || true"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                console.print(Panel(result.stdout.strip(), title="💾 Live Disk Usage", box=box.ROUNDED))
            else:
                console.print("[yellow]⚠️ Could not fetch live disk usage[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Live metrics failed: {e}[/yellow]")


def format_uptime(seconds: int) -> str:
    """Format uptime seconds into human readable string"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def main():
    app()