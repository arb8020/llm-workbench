"""
Command-line interface for GPU cloud operations
"""

import json
import subprocess
import sys
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .client import GPUClient
from .ssh_clients_compat import execute_command_streaming, start_interactive_ssh_session
from .types import CloudType

app = typer.Typer(help="GPU cloud broker CLI")
console = Console()

# Create subcommand groups
instances_app = typer.Typer(help="Instance management commands")
providers_app = typer.Typer(help="Provider information and balance commands")

# Add subcommands to main app
app.add_typer(instances_app, name="instances")
app.add_typer(providers_app, name="providers")


def _internal_search(gpu_type=None, max_price=None, min_vram=None, provider=None, 
                   cloud_type=None, cuda_version=None, sort_by="price", reverse=False):
    """Internal search function that doesn't trigger analysis"""
    # Build pandas-style query using client interface to avoid TypeError
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
        def sort_by_memory(x):
            return x.memory_gb
        sort_func = sort_by_memory
    elif sort_by == "value":
        def sort_by_value(x):
            return x.memory_gb / x.price_per_hour
        sort_func = sort_by_value
    # Default is price, which is handled automatically
    
    # Execute query using client
    if query_conditions:
        combined_query = query_conditions[0]
        for condition in query_conditions[1:]:
            combined_query = combined_query & condition
        offers = client.search(query=combined_query, cuda_version=cuda_version, sort=sort_func, reverse=reverse)
    else:
        offers = client.search(cuda_version=cuda_version, sort=sort_func, reverse=reverse)
    
    return offers


@app.command("search")
def search(
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
        def sort_by_memory(x):
            return x.memory_gb
        sort_func = sort_by_memory
    elif sort_by == "value":
        def sort_by_value(x):
            return x.memory_gb / x.price_per_hour
        sort_func = sort_by_value
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
    
    # Call the internal search function to avoid Typer parameter issues
    all_offers = _internal_search()
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
    secure_types = {o.gpu_type for o in secure_offers}
    community_types = {o.gpu_type for o in community_offers}
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


@app.command("create")
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
    allow_proxy: bool = typer.Option(False, "--allow-proxy", help="Allow proxy SSH connections (default: wait for direct SSH)"),
    container_disk: Optional[int] = typer.Option(None, "--container-disk", help="Container disk size in GB (default: 50GB)"),
    volume_disk: Optional[int] = typer.Option(None, "--volume-disk", help="Volume disk size in GB (default: 0GB)")
):
    """Provision a new GPU instance using pandas-style search"""
    if not print_ssh:
        console.print("🚀 Provisioning GPU instance...")
    
    # Build query for provisioning using client interface to avoid TypeError
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
        def sort_by_memory(x):
            return x.memory_gb
        sort_func = sort_by_memory
    elif sort_by == "value":
        def sort_by_value(x):
            return x.memory_gb / x.price_per_hour
        sort_func = sort_by_value
    
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
                from .api import get_instance
                current_instance = get_instance(instance.id)
                if current_instance and current_instance.public_ip and current_instance.ssh_port:
                    from .types import InstanceStatus
                    if current_instance.status == InstanceStatus.RUNNING:
                        # Check if we should wait for direct SSH or allow proxy
                        is_proxy = current_instance.public_ip == "ssh.runpod.io"
                        
                        if allow_proxy or not is_proxy:
                            # Print SSH command for easy copy-paste
                            if current_instance.ssh_port == 22:
                                print(f"ssh {current_instance.ssh_username}@{current_instance.public_ip}")
                            else:
                                print(f"ssh -p {current_instance.ssh_port} {current_instance.ssh_username}@{current_instance.public_ip}")
                            return
                        # If it's proxy and we don't allow proxy, continue waiting
                
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


@instances_app.command("status")
def instances_status(instance_id: str):
    """Get status of a specific instance"""
    console.print(f"📊 Getting status for instance: {instance_id}")
    
    client = GPUClient()
    instance = client.get_instance(instance_id)
    
    if instance:
        console.print("✅ Instance found")
        console.print(f"   ID: {instance.id}")
        console.print(f"   Status: {instance.status}")
        console.print(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
        console.print(f"   Price: ${instance.price_per_hour:.3f}/hr")
        
        if instance.public_ip:
            console.print(f"   SSH: {instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}")
    else:
        console.print("❌ Instance not found")


@instances_app.command("terminate")  
def instances_terminate(
    instance_id: Optional[str] = typer.Argument(None, help="Instance ID or partial ID to search for and terminate"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be terminated without actually doing it")
):
    """Search for an instance by ID and terminate it"""
    if not instance_id:
        console.print("❌ Instance ID is required")
        console.print("💡 Usage: broker instances terminate <instance_id>")
        return
    
    console.print(f"🔍 Searching for instance: {instance_id}")
    
    client = GPUClient()
    
    # Try exact match first
    instance = client.get_instance(instance_id)
    
    # If not found, try partial ID match
    if not instance:
        instances = client.list_instances()
        matching_instances = [inst for inst in instances if inst.id.startswith(instance_id)]
        
        if len(matching_instances) == 0:
            console.print(f"❌ No instances found matching '{instance_id}'")
            return
        elif len(matching_instances) > 1:
            console.print(f"❌ Multiple instances match '{instance_id}':")
            for inst in matching_instances:
                console.print(f"   {inst.id} - {inst.status}")
            console.print("💡 Please use a more specific ID")
            return
        else:
            instance = matching_instances[0]
    
    # Display instance details
    console.print(f"✅ Found instance: {instance.id}")
    console.print(f"   Status: {instance.status}")
    console.print(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
    console.print(f"   Price: ${instance.price_per_hour:.3f}/hr")
    
    if dry_run:
        console.print("🔥 [DRY RUN] Would terminate this instance")
        return
    
    # Confirmation unless force is used
    if not force:
        console.print("\n⚠️  This will terminate the instance and stop all billing")
        response = typer.confirm("Are you sure you want to proceed?")
        if not response:
            console.print("❌ Termination cancelled")
            return
    
    # Perform termination
    console.print(f"🗑️ Terminating instance: {instance.id}")
    success = client.terminate_instance(instance.id)
    
    if success:
        console.print("✅ Instance terminated successfully")
        console.print("💰 Billing for this instance has been stopped")
    else:
        console.print("❌ Failed to terminate instance")


@instances_app.command("cleanup")
def instances_cleanup(
    instance_id: Optional[str] = typer.Argument(None, help="Specific instance ID to cleanup (optional)"),
    all_instances: bool = typer.Option(False, "--all", "-a", help="Cleanup all running instances"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be cleaned up without actually doing it")
):
    """Cleanup GPU instances (single instance or bulk cleanup)"""
    
    if instance_id and all_instances:
        console.print("❌ Cannot specify both instance ID and --all flag")
        return
    
    if not instance_id and not all_instances:
        console.print("❌ Must specify either an instance ID or use --all flag")
        console.print("💡 Usage examples:")
        console.print("   broker cleanup <instance_id>     # Cleanup specific instance")
        console.print("   broker cleanup --all             # Cleanup all running instances")
        console.print("   broker cleanup --all --dry-run   # See what would be cleaned up")
        return
    
    if instance_id:
        # Single instance cleanup (enhanced terminate)
        _cleanup_single_instance(instance_id, force, dry_run)
    else:
        # Bulk cleanup all instances
        _cleanup_all_instances(force, dry_run)


@instances_app.command("info")
def instances_info(
    instance_id: str,
    live_metrics: bool = typer.Option(False, "--live-metrics", help="Include real-time disk usage via SSH"),
    json_output: bool = typer.Option(False, "--json", help="JSON output format for scripting")
):
    """Get comprehensive information about a specific instance"""
    
    if not json_output:
        console.print(f"🔍 Getting detailed information for instance: {instance_id}")
    
    # Get instance details using the enhanced GraphQL query
    from .providers.runpod import get_instance_details_enhanced
    
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


@instances_app.command("list")
def instances_list(
    simple: bool = typer.Option(False, "--simple", help="Simple output format for scripting (id,name,status,ssh)"),
    json_output: bool = typer.Option(False, "--json", help="JSON output format for scripting"),
    name: Optional[str] = typer.Option(None, "--name", help="Filter by instance name"),
    ssh_only: bool = typer.Option(False, "--ssh-only", help="Output only SSH connection strings (one per line)")
):
    """List user instances"""
    if not simple and not json_output and not ssh_only:
        console.print("📋 Listing all instances...")
    
    client = GPUClient()
    instances = client.list_instances()
    
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
        # Exit with code 1 for scripting when no instances found
        sys.exit(1)
    
    # SSH-only output format for scripting
    if ssh_only:
        for instance in instances:
            # Only output running instances with SSH info
            if str(instance.status).lower() == "instancestatus.running" and instance.public_ip and instance.ssh_port:
                if instance.ssh_port == 22:
                    ssh_cmd = f"ssh {instance.ssh_username}@{instance.public_ip}"
                else:
                    ssh_cmd = f"ssh -p {instance.ssh_port} {instance.ssh_username}@{instance.public_ip}"
                print(ssh_cmd)
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


@instances_app.command("getssh")
def instances_getssh(instance_id: str):
    """Get SSH command for an instance (ready to copy-paste)"""
    # Get instance details
    client = GPUClient()
    instance = client.get_instance(instance_id)
    if not instance:
        # Try to find by partial ID match
        instances = client.list_instances()
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
    if not instance:
        console.print("❌ Instance is None")
        sys.exit(1)
    
    # Type assertion - we know instance is not None after the check above
    assert instance is not None
    
    if instance.status != InstanceStatus.RUNNING:
        console.print(f"❌ Instance is not running (status: {instance.status})")
        sys.exit(1)
    
    # Check if SSH is available
    if not instance.public_ip or not instance.ssh_port:
        console.print("❌ SSH connection info not available")
        sys.exit(1)
    
    # Output SSH command for easy copy-paste
    if instance.ssh_port == 22:
        # Standard port, no -p needed
        print(f"ssh {instance.ssh_username}@{instance.public_ip}")
    else:
        # Non-standard port, include -p
        print(f"ssh -p {instance.ssh_port} {instance.ssh_username}@{instance.public_ip}")


@instances_app.command("ssh")
def instances_ssh(instance_id: str):
    """Start an interactive SSH session to an instance"""
    console.print(f"🔗 Connecting to instance: {instance_id}")
    
    # Get instance details
    client = GPUClient()
    instance = client.get_instance(instance_id)
    if not instance:
        console.print("❌ Instance not found")
        raise typer.Exit(1)
    
    # Type assertion - we know instance is not None after the check above
    assert instance is not None
    
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
            ssh_key_path = client.get_ssh_key_path()
        except Exception:
            pass  # Fall back to SSH agent
        
        # Extract the wrapped GPUInstance for SSH functions
        start_interactive_ssh_session(instance._instance, private_key=ssh_key_path)
    except Exception as e:
        console.print(f"❌ SSH connection failed: {e}")


@instances_app.command("exec")
def instances_exec(instance_id: str, command: str):
    """Execute a command on an instance with streaming output"""
    console.print(f"🔄 Executing command on instance: {instance_id}")
    
    # Get instance details
    client = GPUClient()
    instance = client.get_instance(instance_id)
    if not instance:
        console.print("❌ Instance not found")
        raise typer.Exit(1)
    
    # Type assertion - we know instance is not None after the check above
    assert instance is not None
    
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
            ssh_key_path = client.get_ssh_key_path()
        except Exception:
            pass  # Fall back to SSH agent
        
        # Extract the wrapped GPUInstance for SSH functions
        exit_code, stdout, stderr = execute_command_streaming(instance._instance, command, private_key=ssh_key_path)
        if exit_code != 0:
            console.print(f"❌ Command execution failed (exit code: {exit_code})")
            if stderr.strip():
                console.print(f"   Error: {stderr.strip()}")
    except Exception as e:
        console.print(f"❌ Command execution failed: {e}")


@app.command()
def config():
    """Check configuration and validate setup"""
    console.print("🔧 [bold]GPU Broker Configuration Check[/bold]\n")
    
    try:
        client = GPUClient()
        
        validation_results = client.validate_configuration()
        
        for component, status in validation_results.items():
            console.print(f"**{component.replace('_', ' ').title()}**: {status}")
            
        # Test API connection
        console.print("\n🌐 **Connection Test**:")
        try:
            offers = _internal_search()
            console.print(f"✅ RunPod API connection successful ({len(offers)} GPU offers available)")
        except Exception as e:
            console.print(f"❌ RunPod API connection failed: {e}")
            
    except Exception as e:
        console.print(f"❌ Configuration error: {e}")
        console.print("\n💡 **Setup Help:**")
        console.print("   1. Copy `.env.example` to `.env` and add your RunPod API key")
        console.print("   2. Ensure SSH key exists at `~/.ssh/id_ed25519` or set GPU_BROKER_SSH_KEY")
        console.print("   3. Upload your SSH public key to RunPod dashboard")


@providers_app.command("balance")
def providers_balance(
    json_output: bool = typer.Option(False, "--json", help="JSON output format for scripting"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Specific provider (default: all available)")
):
    """Show user balance and spending information for each provider"""
    
    if not json_output:
        console.print("💰 [bold]Provider Balance Information[/bold]\n")
    
    # For now, we only support RunPod, but this is designed to be extensible
    providers_to_check = ["runpod"] if not provider else [provider]
    
    balance_data = []
    
    for provider_name in providers_to_check:
        if provider_name.lower() == "runpod":
            try:
                from .providers.runpod import get_user_balance
                balance_info = get_user_balance()
                
                if balance_info:
                    balance_data.append(balance_info)
                    
                    if not json_output:
                        # Display RunPod balance information
                        current_balance = balance_info.get("current_balance", 0)
                        current_spend = balance_info.get("current_spend_per_hour", 0)
                        lifetime_spend = balance_info.get("lifetime_spend", 0)
                        spend_limit = balance_info.get("spend_limit")
                        referral_earnings = balance_info.get("referral_earnings", 0)
                        
                        # Create balance panel
                        balance_info_text = [
                            f"Current Balance: [green]${current_balance:.2f}[/green]",
                            f"Current Hourly Spend: [yellow]${current_spend:.3f}/hr[/yellow]",
                            f"Lifetime Spend: [blue]${lifetime_spend:.2f}[/blue]",
                        ]
                        
                        if spend_limit:
                            balance_info_text.append(f"Spend Limit: [red]${spend_limit:.2f}[/red]")
                        
                        if referral_earnings > 0:
                            balance_info_text.append(f"Referral Earnings: [cyan]${referral_earnings:.2f}[/cyan]")
                        
                        # Calculate burn rate and time remaining
                        if current_spend > 0 and current_balance > 0:
                            hours_remaining = current_balance / current_spend
                            if hours_remaining < 24:
                                time_str = f"{hours_remaining:.1f} hours"
                                warning = " ⚠️"
                            elif hours_remaining < 168:  # 1 week
                                time_str = f"{hours_remaining/24:.1f} days"
                                warning = " ⚠️" if hours_remaining < 48 else ""
                            else:
                                time_str = f"{hours_remaining/24/7:.1f} weeks"
                                warning = ""
                            
                            balance_info_text.append(f"Time Remaining: [magenta]{time_str}[/magenta]{warning}")
                        
                        console.print(Panel(
                            "\n".join(balance_info_text), 
                            title=f"🏃 RunPod Account", 
                            box=box.ROUNDED
                        ))
                else:
                    if not json_output:
                        console.print("❌ Failed to get RunPod balance information")
                        
            except Exception as e:
                if not json_output:
                    console.print(f"❌ Error getting RunPod balance: {e}")
        else:
            if not json_output:
                console.print(f"❌ Provider '{provider_name}' not supported yet")
    
    if json_output:
        import json
        print(json.dumps(balance_data, indent=2))
    elif not balance_data:
        console.print("❌ No balance information available")


@providers_app.command("list")
def providers_list():
    """List all available providers"""
    console.print("📋 [bold]Available Providers[/bold]\n")
    
    providers_info = [
        ("runpod", "RunPod", "✅ Active", "Community & Secure cloud GPU instances"),
        # Add more providers here as they become available
        # ("vast", "Vast.ai", "🚧 Planned", "Community GPU marketplace"),
        # ("lambda", "Lambda Labs", "🚧 Planned", "On-demand cloud GPUs"),
    ]
    
    table = Table(title="GPU Cloud Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Description", style="blue")
    
    for provider_id, name, status, description in providers_info:
        table.add_row(provider_id, name, status, description)
    
    console.print(table)
    console.print("\n💡 Use 'broker providers balance' to check account balances")


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
        
        util_info = "Real-time Utilization:\n"
        util_info += f"• CPU: {cpu_util}% ({instance_data.get('vcpuCount', '?')} vCPUs available)\n"
        util_info += f"• Memory: {mem_util}% ({instance_data.get('memoryInGb', '?')}GB RAM available)\n"
        util_info += f"• Disk: {disk_util}% ({instance_data.get('containerDiskInGb', '?')}GB container)"
        
        # Add warning for high disk usage
        if disk_util > 75:
            util_info += " ⚠️ High usage!"
        
        status_info.append(util_info)
    
    console.print(Panel("\n".join(status_info), title="📊 Status & Performance", box=box.ROUNDED))
    
    # GPU Telemetry section
    gpu_telemetry = machine.get('gpuTelemetry', []) if machine else []
    if gpu_telemetry:
        telemetry_info = []
        for i, gpu in enumerate(gpu_telemetry):
            gpu_id = gpu.get('id', f'GPU-{i}')
            util = gpu.get('percentUtilization', 0)
            temp = gpu.get('temperatureCelcius', 0)
            mem_util = gpu.get('memoryUtilization', 0)
            power = gpu.get('powerWatts', 0)
            
            # Add temperature warning
            temp_warning = ""
            if temp > 80:
                temp_warning = " 🔥 Hot!"
            elif temp > 70:
                temp_warning = " ⚠️ Warm"
            
            # Add power efficiency info
            power_info = f"{power:.1f}W" if power > 0 else "N/A"
            
            telemetry_info.append(f"[bold]{gpu_id}[/bold]:")
            telemetry_info.append(f"• Utilization: {util:.1f}%")
            telemetry_info.append(f"• Memory: {mem_util:.1f}% utilized")
            telemetry_info.append(f"• Temperature: {temp:.1f}°C{temp_warning}")
            telemetry_info.append(f"• Power: {power_info}")
            if i < len(gpu_telemetry) - 1:
                telemetry_info.append("")  # Add spacing between GPUs
        
        console.print(Panel("\n".join(telemetry_info), title="🎮 GPU Telemetry", box=box.ROUNDED))
    
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
        f"Direct SSH: {'✅ Available' if ssh_info and 'ssh.runpod.io' not in str(ssh_info) else '❌ Proxy only'}",
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
        console.print("\n🔍 [yellow]Fetching live disk usage via SSH...[/yellow]")
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


def _cleanup_single_instance(instance_id: str, force: bool, dry_run: bool):
    """Cleanup a single instance with enhanced feedback"""
    
    # Get instance details first
    client = GPUClient()
    instance = client.get_instance(instance_id)
    if not instance:
        # Try to find by partial ID match like in getssh command
        instances = client.list_instances()
        matching_instances = [inst for inst in instances if inst.id.startswith(instance_id)]
        if len(matching_instances) == 1:
            instance = matching_instances[0]
        elif len(matching_instances) > 1:
            console.print(f"❌ Multiple instances match '{instance_id}': {[inst.id for inst in matching_instances]}")
            return
        else:
            console.print("❌ Instance not found")
            return
    
    # Display instance details
    console.print(f"🔍 Found instance: {instance.id}")
    console.print(f"   Status: {instance.status}")
    console.print(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
    console.print(f"   Price: ${instance.price_per_hour:.3f}/hr")
    
    if dry_run:
        console.print("🔥 [DRY RUN] Would terminate this instance")
        return
    
    # Confirmation unless force is used
    if not force:
        console.print("\n⚠️  This will terminate the instance and stop all billing")
        response = typer.confirm("Are you sure you want to proceed?")
        if not response:
            console.print("❌ Cleanup cancelled")
            return
    
    # Perform termination
    console.print(f"🧹 Cleaning up instance: {instance.id}")
    from .api import terminate_instance
    success = terminate_instance(instance.id)
    
    if success:
        console.print("✅ Instance cleanup completed successfully")
        console.print("💰 Billing for this instance has been stopped")
    else:
        console.print("❌ Failed to cleanup instance")


def _cleanup_all_instances(force: bool, dry_run: bool):
    """Cleanup all running instances"""
    console.print("🔍 Scanning for running instances...")
    
    client = GPUClient()
    instances = client.list_instances()
    
    if not instances:
        console.print("✅ No instances found")
        return
    
    # Filter for running instances
    from .types import InstanceStatus
    running_instances = [inst for inst in instances if inst.status == InstanceStatus.RUNNING]
    
    if not running_instances:
        console.print("✅ No running instances to cleanup")
        return
    
    # Display what would be cleaned up
    console.print(f"🎯 Found {len(running_instances)} running instance(s):")
    
    total_cost = 0
    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("GPU", style="green")
    table.add_column("Price/Hr", style="yellow")
    table.add_column("Status", style="magenta")
    
    for instance in running_instances:
        total_cost += instance.price_per_hour
        gpu_info = f"{instance.gpu_type} x{instance.gpu_count}" if instance.gpu_count > 1 else instance.gpu_type
        table.add_row(
            instance.id[:12] + "..." if len(instance.id) > 15 else instance.id,
            gpu_info,
            f"${instance.price_per_hour:.3f}",
            str(instance.status)
        )
    
    console.print(table)
    console.print(f"\n💰 Total hourly cost: ${total_cost:.3f}/hr")
    
    if dry_run:
        console.print(f"🔥 [DRY RUN] Would terminate {len(running_instances)} instance(s)")
        return
    
    # Confirmation unless force is used
    if not force:
        console.print(f"\n⚠️  This will terminate ALL {len(running_instances)} running instances and stop billing")
        response = typer.confirm("Are you sure you want to proceed?")
        if not response:
            console.print("❌ Cleanup cancelled")
            return
    
    # Perform bulk termination
    console.print(f"🧹 Cleaning up {len(running_instances)} instances...")
    
    success_count = 0
    failed_instances = []
    
    for instance in running_instances:
        console.print(f"  Terminating {instance.id[:12]}...")
        success = client.terminate_instance(instance.id)
        
        if success:
            success_count += 1
            console.print("    ✅ Terminated")
        else:
            failed_instances.append(instance.id)
            console.print("    ❌ Failed")
    
    # Summary
    console.print(f"\n🎯 Cleanup complete: {success_count}/{len(running_instances)} instances terminated")
    console.print(f"💰 Estimated savings: ${total_cost:.3f}/hr")
    
    if failed_instances:
        console.print(f"⚠️  Failed to terminate {len(failed_instances)} instance(s):")
        for failed_id in failed_instances:
            console.print(f"   {failed_id}")
        console.print("💡 Try running the cleanup command again in a few seconds")




def main():
    app()