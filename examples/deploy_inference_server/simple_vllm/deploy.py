#!/usr/bin/env python3
"""Push-button vLLM deployment script."""

import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import broker and bifrost modules
# Note: Run with PYTHONPATH=/path/to/broker:/path/to/bifrost for clean imports

from broker.client import GPUClient
from bifrost.client import BifrostClient
from engine.engine.deployment import wait_for_server_ready


def deploy_vllm(min_vram: int = 8, max_price: float = 0.40) -> dict:
    """Deploy vLLM server on GPU and return connection info."""
    
    print("🚀 Starting vLLM deployment...")
    
    # 1. PROVISION GPU
    print(f"📡 Creating GPU instance (min {min_vram}GB VRAM, max ${max_price}/hr)...")
    gpu_client = GPUClient()
    
    # Build query for GPU with minimum VRAM and max price
    query = (gpu_client.vram_gb >= min_vram) & (gpu_client.price_per_hour <= max_price)
    
    gpu_instance = gpu_client.create(
        query=query,
        exposed_ports=[8000],  # Expose port 8000 for vLLM
        enable_http_proxy=True,  # Enable RunPod proxy
        name="vllm-server",
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
    
    ssh_connection = gpu_instance.ssh_connection_string()
    print(f"✅ SSH ready: {ssh_connection}")
    
    # 2. DEPLOY CODE
    print("📦 Deploying codebase...")
    bifrost_client = BifrostClient(ssh_connection)
    
    # Deploy the codebase to remote workspace
    workspace_path = bifrost_client.push()
    bifrost_client.exec("echo 'Codebase deployed successfully'")
    print(f"✅ Code deployed to: {workspace_path}")
    
    # 3. START SERVER IN TMUX
    print("🌟 Starting vLLM server in tmux session...")
    
    # Create tmux session and start vLLM server with persistent logging
    tmux_cmd = "tmux new-session -d -s vllm 'python examples/deploy_inference_server/simple_vllm/start_server.py 2>&1 | tee ~/vllm_server.log'"
    bifrost_client.exec(tmux_cmd)
    
    print("✅ vLLM server starting in tmux session 'vllm'")
    
    # 4. POLL UNTIL READY
    print("⏳ Waiting for server to be ready (this may take 2-3 minutes for model loading)...")
    
    server_ready = wait_for_server_ready(bifrost_client, "gpt2", timeout=600)
    
    if not server_ready:
        print(f"❌ Server failed to start within timeout")
        print("   Check logs with:")
        print(f"   # Persistent log file: bifrost exec {ssh_connection} 'cat ~/vllm_server.log'")
        print(f"   # tmux session logs: bifrost exec {ssh_connection} 'tmux capture-pane -t vllm -p'")
        sys.exit(1)
    
    print("✅ Server is ready and responding!")
    
    # 5. CONSTRUCT PROXY URL  
    proxy_url = gpu_instance.get_proxy_url(8000)
    
    if not proxy_url:
        print("⚠️  No proxy URL available - instance may not be RunPod")
        proxy_url = f"http://{gpu_instance.public_ip}:8000"
    
    # 6. RETURN CONNECTION INFO
    connection_info = {
        "url": proxy_url,
        "instance_id": gpu_instance.id,
        "ssh": ssh_connection,
        "provider": gpu_instance.provider,
        "status": "ready"
    }
    
    print("\n🎉 vLLM deployment complete!")
    print(f"   Server URL: {proxy_url}")
    print(f"   Instance ID: {gpu_instance.id}")
    print(f"   SSH: {ssh_connection}")
    
    print("\n🧪 Test your server:")
    print(f"   curl -X POST {proxy_url}/v1/completions \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model":"openai-community/gpt2","prompt":"Hello!","max_tokens":20}\'')
    
    print("\n🔧 Management commands:")
    print(f"   # View tmux session: bifrost exec {ssh_connection} 'tmux attach -t vllm'")
    print(f"   # Check persistent logs: bifrost exec {ssh_connection} 'cat ~/vllm_server.log'")
    print(f"   # Check tmux session: bifrost exec {ssh_connection} 'tmux capture-pane -t vllm -p'")
    print(f"   # Stop server: bifrost exec {ssh_connection} 'tmux kill-session -t vllm'")
    print(f"   # Terminate GPU: broker terminate {gpu_instance.id}")
    
    return connection_info


def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy vLLM server on remote GPU")
    parser.add_argument("--min-vram", type=int, default=8, help="Minimum VRAM in GB")
    parser.add_argument("--max-price", type=float, default=0.40, help="Maximum price per hour")
    parser.add_argument("--json", action="store_true", help="Output connection info as JSON")
    
    args = parser.parse_args()
    
    try:
        connection_info = deploy_vllm(args.min_vram, args.max_price)
        
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
