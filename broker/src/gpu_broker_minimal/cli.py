"""
Command-line interface for GPU cloud operations
"""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

from . import search, get_instance, terminate_instance, create, list_instances
from .types import CloudType
from .ssh_clients import start_interactive_ssh_session, execute_command_streaming
import gpu_broker_minimal as gpus

app = typer.Typer(help="GPU cloud broker CLI")
console = Console()


@app.command()
def search_gpus(
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type", help="GPU type to search for"),
    max_price: Optional[float] = typer.Option(None, "--max-price", help="Maximum price per hour"),
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
    
    all_offers = gpus.search()
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
def create_instance(
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type", help="GPU type to provision"),
    cloud_type: Optional[str] = typer.Option("secure", "--cloud-type", help="Cloud type: 'secure' (default) or 'community'"),
    max_price: Optional[float] = typer.Option(None, "--max-price", help="Maximum price per hour"),
    image: str = typer.Option("runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04", "--image", help="Docker image"),
    name: Optional[str] = typer.Option(None, "--name", help="Instance name"),
    sort_by: Optional[str] = typer.Option("price", "--sort", help="Sort by: 'price', 'memory', 'value' (memory/price)"),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Try up to N offers before giving up")
):
    """Provision a new GPU instance using pandas-style search"""
    console.print("🚀 Provisioning GPU instance...")
    
    # Build query for provisioning using client interface to avoid TypeError
    from .client import GPUClient
    client = GPUClient()
    
    query_conditions = []
    
    if max_price:
        query_conditions.append(client.price_per_hour < max_price)
    if gpu_type:
        query_conditions.append(client.gpu_type.contains(gpu_type))
    if cloud_type:
        cloud_enum = CloudType.SECURE if cloud_type == "secure" else CloudType.COMMUNITY
        query_conditions.append(client.cloud_type == cloud_enum)
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
    console.print(f"🎯 Trying up to {max_attempts} offers sorted by {sort_by}...")
    try:
        instance = client.create(
            query=final_query,
            image=image, 
            name=name,
            sort=sort_func,
            reverse=(sort_by != "price"),  # Descending for memory/value, ascending for price
            max_attempts=max_attempts
        )
    except Exception as e:
        console.print(f"❌ Provisioning failed: {e}")
        return
    
    if instance:
        console.print(f"✅ Instance created: {instance.id}")
        console.print(f"   Status: {instance.status}")
        console.print(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
        console.print(f"   Price: ${instance.price_per_hour:.3f}/hr")
    else:
        console.print("❌ Failed to provision instance")


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
def list():
    """List all user instances"""
    console.print("📋 Listing all instances...")
    
    instances = list_instances()
    
    if not instances:
        console.print("📭 No instances found")
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


def main():
    app()