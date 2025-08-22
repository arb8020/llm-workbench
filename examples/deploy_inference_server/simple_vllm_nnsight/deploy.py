#!/usr/bin/env python3
"""Deploy interpretability-enabled vLLM server with nnsight integration."""

import sys
import time
import json
from pathlib import Path

# Import broker and bifrost modules
from broker.client import GPUClient
from bifrost.client import BifrostClient


def deploy_interpretability_server(min_vram: int = 12, max_price: float = 0.60) -> dict:
    """Deploy interpretability-enabled vLLM server on GPU and return connection info."""
    
    print("🚀 Starting interpretability server deployment...")
    print("🧠 This server includes activation collection and intervention capabilities")
    
    # 1. PROVISION GPU (needs more VRAM for interpretability features)
    print(f"📡 Creating GPU instance (min {min_vram}GB VRAM, max ${max_price}/hr)...")
    gpu_client = GPUClient()
    
    # Build query for GPU with minimum VRAM and max price
    query = (gpu_client.vram_gb >= min_vram) & (gpu_client.price_per_hour <= max_price)
    
    gpu_instance = gpu_client.create(
        query=query,
        exposed_ports=[8000],  # Expose port 8000 for server
        enable_http_proxy=True,  # Enable RunPod proxy
        name="interp-server",
        cloud_type="secure"
    )
    
    print(f"✅ GPU ready: {gpu_instance.id}")
    
    # Wait for SSH to be ready
    print("⏳ Waiting for SSH connection to be ready...")
    if not gpu_instance.wait_until_ssh_ready(timeout=300):  # 5 minutes
        print("❌ Failed to get SSH connection ready")
        sys.exit(1)
    
    ssh_connection = gpu_instance.ssh_connection_string()
    print(f"✅ SSH ready: {ssh_connection}")
    
    # 2. DEPLOY CODE
    print("📦 Deploying codebase...")
    bifrost_client = BifrostClient(ssh_connection)
    
    # Deploy the codebase to remote workspace
    workspace_path = bifrost_client.push()
    bifrost_client.exec("echo 'Codebase deployed successfully'")
    print(f"✅ Code deployed to: {workspace_path}")
    
    # 3. INSTALL ENGINE WITH INTERPRETABILITY DEPENDENCIES
    print("🔧 Installing engine[interp] with nnsight and interpretability dependencies...")
    print("   Note: Using uv lock and sync for consistent dependency resolution")
    
    # Install engine with interpretability dependencies using uv lock/sync
    bifrost_client.exec("uv lock --all-extras")
    result = bifrost_client.exec("uv sync --extra interp")
    if result and ("Project synced" in result or "already up-to-date" in result.lower()):
        print("✅ engine[interp] installed successfully")
        print("   Includes: nnsight, vLLM 0.9.2, FastAPI, uvicorn with resolved transformers")
    else:
        print("⚠️  engine[interp] installation may have issues - check logs")
    
    # 4. START SERVER IN TMUX
    print("🌟 Starting interpretability server in tmux session...")
    
    # Create tmux session and start interpretability server with persistent logging
    tmux_cmd = "tmux new-session -d -s interp 'python -m engine.scripts.deploy_simple.interp 2>&1 | tee ~/interp_server.log'"
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
    
    parser = argparse.ArgumentParser(description="Deploy interpretability-enabled vLLM server on remote GPU")
    parser.add_argument("--min-vram", type=int, default=12, help="Minimum VRAM in GB (default: 12 for interpretability features)")
    parser.add_argument("--max-price", type=float, default=0.60, help="Maximum price per hour (default: 0.60)")
    parser.add_argument("--json", action="store_true", help="Output connection info as JSON")
    
    args = parser.parse_args()
    
    try:
        connection_info = deploy_interpretability_server(args.min_vram, args.max_price)
        
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