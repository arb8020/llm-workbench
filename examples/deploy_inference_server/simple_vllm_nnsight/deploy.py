#!/usr/bin/env python3
"""Deploy interpretability-enabled vLLM server with nnsight integration."""

import sys
import time
import json
import logging
from pathlib import Path

from shared.logging_config import setup_logging

# Import broker and bifrost modules
from broker.client import GPUClient
from bifrost.client import BifrostClient

logger = logging.getLogger(__name__)


def deploy_interpretability_server(min_vram: int = 12, max_price: float = 0.60, force_new_gpu: bool = False, 
                                   existing_only: bool = False, model: str = "openai-community/gpt2") -> dict:
    """Deploy interpretability-enabled vLLM server on GPU and return connection info.
    
    This function implements smart GPU reuse:
    - Reuses existing 'interp-server' GPU instances that meet requirements (saves cost & time)
    - Always redeploys code, reinstalls dependencies, and restarts server (ensures freshness)
    - Use force_new_gpu=True to provision brand new GPU hardware
    - Use existing_only=True to only use existing GPUs (fail if none found)
    
    Args:
        min_vram: Minimum VRAM required in GB
        max_price: Maximum price per hour
        force_new_gpu: If True, always create new GPU instance instead of reusing
        existing_only: If True, only use existing instances, fail if none found
    
    Returns:
        Connection info dict with URL, instance ID, SSH details, etc.
    """
    
    # Handle special modes
    if existing_only:
        print("🎯 EXISTING-ONLY MODE: Will only use existing instances, fail if none found")
    if force_new_gpu:
        print("⚡ FORCE-NEW-GPU MODE: Will create new instance regardless of existing ones")
    
    print("🚀 Starting interpretability server deployment...")
    print("🧠 This server includes activation collection and intervention capabilities")
    
    gpu_client = GPUClient()
    gpu_instance = None
    
    # 1. CHECK FOR EXISTING INSTANCE (unless force_new_gpu is True)
    if not force_new_gpu:
        print("🔍 Checking for existing interp-server instances...")
        print(f"   Looking for instances with:")
        print(f"     • Name: 'interp-server'")
        print(f"     • Status: 'running'")
        print(f"     • Min VRAM: {min_vram}GB")
        print(f"     • Max price: ${max_price}/hr")
        
        existing_instances = gpu_client.list_instances()
        print(f"📋 Found {len(existing_instances)} total instances")
        
        # Look for running interp-server instances that meet our requirements
        suitable_instances = []
        
        for i, instance in enumerate(existing_instances):
            print(f"\n🔍 Instance {i+1}/{len(existing_instances)}:")
            print(f"   • ID: {getattr(instance, 'id', 'Unknown')}")
            print(f"   • Name: '{getattr(instance, 'name', 'Unknown')}'")
            print(f"   • Status: '{getattr(instance, 'status', 'Unknown')}'")
            print(f"   • VRAM: {getattr(instance, 'vram_gb', 'Unknown')}GB")
            print(f"   • Price: ${getattr(instance, 'price_per_hour', 'Unknown')}/hr")
            print(f"   • GPU Type: {getattr(instance, 'gpu_type', 'Unknown')}")
            
            # Check each requirement individually for better debugging
            name_match = instance.name == "interp-server"
            status_match = instance.status == "running"
            has_vram = hasattr(instance, 'vram_gb')
            vram_sufficient = has_vram and instance.vram_gb >= min_vram
            has_price = hasattr(instance, 'price_per_hour')
            price_acceptable = has_price and instance.price_per_hour <= max_price
            
            print(f"   • Name matches 'interp-server': {name_match}")
            print(f"   • Status is 'running': {status_match}")
            print(f"   • Has VRAM attribute: {has_vram}")
            if has_vram:
                print(f"   • VRAM sufficient (>={min_vram}GB): {vram_sufficient}")
            else:
                print(f"   • VRAM sufficient (>={min_vram}GB): Skipped (existing running instance assumed sufficient)")
                vram_sufficient = True  # Assume existing running instance has sufficient VRAM
            print(f"   • Has price attribute: {has_price}")
            print(f"   • Price acceptable (<=${max_price}): {price_acceptable}")
            
            all_requirements_met = (name_match and status_match and 
                                   vram_sufficient and 
                                   has_price and price_acceptable)
            print(f"   • ✅ All requirements met: {all_requirements_met}")
            
            if all_requirements_met:
                suitable_instances.append(instance)
                print(f"   → Added to suitable instances")
        
        print(f"\n📊 Summary: Found {len(suitable_instances)} suitable instances out of {len(existing_instances)} total")
        
        if suitable_instances:
            # Use the first suitable instance
            gpu_instance = suitable_instances[0]
            print(f"✨ Selected existing interp-server: {gpu_instance.id}")
            print(f"   GPU: {getattr(gpu_instance, 'gpu_type', 'Unknown')}")
            print(f"   VRAM: {getattr(gpu_instance, 'vram_gb', 'Unknown')}GB")
            print(f"   Price: ${getattr(gpu_instance, 'price_per_hour', 'Unknown')}/hr")
            print("🔄 Will redeploy code, dependencies, and server on existing GPU")
        else:
            print("📝 No suitable existing interp-server instances found")
            if len(existing_instances) == 0:
                print("   → No instances exist at all")
            else:
                print("   → Instances exist but none meet all requirements")
            
            # If existing_only mode is enabled, fail here instead of creating new GPU
            if existing_only:
                print("❌ EXISTING-ONLY MODE: No suitable existing instances found, failing as requested")
                print("💡 To create a new GPU instance, run without --existing-only flag")
                sys.exit(1)
    
    # 2. PROVISION NEW GPU (if no existing instance or force_new_gpu)
    if gpu_instance is None:
        print(f"📡 Creating new GPU instance (min {min_vram}GB VRAM, max ${max_price}/hr)...")
        
        # Build query for GPU with minimum VRAM and max price
        query = (gpu_client.vram_gb >= min_vram) & (gpu_client.price_per_hour <= max_price)
        
        gpu_instance = gpu_client.create(
            query=query,
            exposed_ports=[8000],  # Expose port 8000 for server
            enable_http_proxy=True,  # Enable RunPod proxy
            name="interp-server",
            cloud_type="secure",
            sort=lambda x: x.price_per_hour,  # Sort by price (cheapest first)
            reverse=False
        )
        
        print(f"✅ GPU ready: {gpu_instance.id}")
        
        # Wait for SSH to be ready
        print("⏳ Waiting for SSH connection to be ready...")
        if not gpu_instance.wait_until_ssh_ready(timeout=300):  # 5 minutes
            print("❌ Failed to get SSH connection ready")
            sys.exit(1)
    else:
        # For existing instances, verify SSH is still ready
        print("⏳ Verifying SSH connection to existing instance...")
        if not gpu_instance.wait_until_ssh_ready(timeout=60):  # Shorter timeout for existing
            print("❌ SSH connection to existing instance failed")
            print("💡 Try running with --force-new-gpu to create a fresh instance")
            sys.exit(1)
    
    ssh_connection = gpu_instance.ssh_connection_string()
    print(f"✅ SSH ready: {ssh_connection}")
    
    # 2. DEPLOY CODE (always fresh)
    print("📦 Deploying fresh codebase...")
    bifrost_client = BifrostClient(ssh_connection)
    
    # Deploy the codebase to remote workspace
    workspace_path = bifrost_client.push()
    bifrost_client.exec("echo 'Codebase deployed successfully'")
    print(f"✅ Code deployed to: {workspace_path}")
    
    # 3. INSTALL ENGINE WITH INTERPRETABILITY DEPENDENCIES (always fresh)
    print("🔧 Installing fresh engine[interp] with nnsight and interpretability dependencies...")
    print("   Note: Using uv lock and sync for consistent dependency resolution")
    
    # Install engine with interpretability dependencies using uv
    result = bifrost_client.exec("uv sync --extra interp")
    if result and ("Project synced" in result or "already up-to-date" in result.lower()):
        print("✅ engine[interp] installed successfully")
        print("   Includes: nnsight, vLLM 0.9.2, FastAPI, uvicorn with resolved transformers")
    else:
        print("⚠️  engine[interp] installation may have issues - check logs")
    
    # 4. RESTART INTERPRETABILITY SERVER (always fresh)
    print("🌟 Starting fresh interpretability server in tmux session...")
    
    # Kill any existing tmux session first  
    bifrost_client.exec("tmux kill-session -t interp 2>/dev/null || true")
    
    # Create tmux session and start interpretability server with persistent logging
    tmux_cmd = f"tmux new-session -d -s interp 'cd ~/.bifrost/workspace && uv run python -m engine.scripts.deploy.deploy_simple.interp --model {model} 2>&1 | tee ~/interp_server.log'"
    bifrost_client.exec(tmux_cmd)
    
    print("✅ Interpretability server starting in tmux session 'interp'")
    
    # 5. POLL UNTIL READY
    print("⏳ Waiting for server to be ready (this may take 3-4 minutes for nnsight model loading)...")
    
    max_wait_time = 720  # 12 minutes max (nnsight can take longer)
    start_time = time.time()
    server_ready = False
    
    while not server_ready and (time.time() - start_time) < max_wait_time:
        try:
            # Check if OpenAI-compatible server is responding
            models_check = bifrost_client.exec("curl -s --connect-timeout 5 http://localhost:8000/v1/models")
            if models_check and "gpt2" in models_check.lower():
                # Also check if interpretability capabilities are available
                caps_check = bifrost_client.exec("curl -s --connect-timeout 5 http://localhost:8000/v1/capabilities")
                if caps_check and "activation_collection" in caps_check.lower():
                    server_ready = True
                    break
                    
        except Exception as e:
            # Server not ready yet, continue waiting
            pass
        
        elapsed = int(time.time() - start_time)
        if elapsed % 45 == 0:  # Print update every 45 seconds
            print(f"   Still loading model with nnsight... ({elapsed}s elapsed)")
            
        time.sleep(15)
    
    if not server_ready:
        print("❌ Server failed to start within timeout")
        print("   Check logs with:")
        print(f"   # Persistent log file: bifrost exec {ssh_connection} 'cat ~/interp_server.log'")
        print(f"   # tmux session logs: bifrost exec {ssh_connection} 'tmux capture-pane -t interp -p'")
        sys.exit(1)
    
    print("✅ Interpretability server is ready and responding!")
    
    # 6. CONSTRUCT PROXY URL  
    proxy_url = gpu_instance.get_proxy_url(8000)
    
    if not proxy_url:
        print("⚠️  No proxy URL available - instance may not be RunPod")
        proxy_url = f"http://{gpu_instance.public_ip}:8000"
    
    # 7. RETURN CONNECTION INFO
    connection_info = {
        "url": proxy_url,
        "instance_id": gpu_instance.id,
        "ssh": ssh_connection,
        "provider": gpu_instance.provider,
        "status": "ready",
        "capabilities": ["activation_collection", "activation_patching", "interventions"]
    }
    
    print("\n🎉 Interpretability server deployment complete!")
    print(f"   Server URL: {proxy_url}")
    print(f"   Instance ID: {gpu_instance.id}")
    print(f"   SSH: {ssh_connection}")
    print(f"   Capabilities: activation collection, interventions")
    
    print("\n🧪 Test your server:")
    print("   # Standard completion:")
    print(f"   curl -X POST {proxy_url}/v1/chat/completions \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model":"openai-community/gpt2","messages":[{"role":"user","content":"Hello!"}],"max_tokens":20}\'')
    
    print("\n   # With activation collection:")
    print(f"   curl -X POST {proxy_url}/v1/chat/completions \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model":"openai-community/gpt2","messages":[{"role":"user","content":"Hello!"}],"max_tokens":20,"collect_activations":{"layers":[6,12],"hook_points":["output"]}}\'')
    
    print("\n   # Check capabilities:")
    print(f"   curl {proxy_url}/v1/capabilities")
    
    print("\n🔧 Management commands:")
    print(f"   # View tmux session: bifrost exec {ssh_connection} 'tmux attach -t interp'")
    print(f"   # Check persistent logs: bifrost exec {ssh_connection} 'cat ~/interp_server.log'")
    print(f"   # Check tmux session: bifrost exec {ssh_connection} 'tmux capture-pane -t interp -p'")
    print(f"   # Stop server: bifrost exec {ssh_connection} 'tmux kill-session -t interp'")
    print(f"   # Terminate GPU: broker terminate {gpu_instance.id}")
    
    return connection_info


def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Deploy interpretability-enabled vLLM server on remote GPU",
        epilog="Default behavior: Reuse existing 'interp-server' GPU if available, always redeploy code/deps/server"
    )
    parser.add_argument("--min-vram", type=int, default=12, help="Minimum VRAM in GB (default: 12 for interpretability features)")
    parser.add_argument("--max-price", type=float, default=0.60, help="Maximum price per hour (default: 0.60)")
    parser.add_argument("--force-new-gpu", action="store_true", help="Force creation of new GPU instance instead of reusing existing one")
    parser.add_argument("--existing-only", action="store_true", help="Only use existing GPU instances, fail if none suitable found (no new GPU creation)")
    parser.add_argument("--json", action="store_true", help="Output connection info as JSON")
    
    args = parser.parse_args()
    
    # Setup logging for the example
    setup_logging()
    
    try:
        connection_info = deploy_interpretability_server(args.min_vram, args.max_price, args.force_new_gpu, args.existing_only)
        
        if args.json:
            print(json.dumps(connection_info, indent=2))
        
        return connection_info
        
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()