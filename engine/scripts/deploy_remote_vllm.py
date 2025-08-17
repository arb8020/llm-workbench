#!/usr/bin/env python3
"""Deploy vLLM backend on remote GPU using broker + bifrost."""

import asyncio
import sys
import subprocess
import time
import requests
from pathlib import Path

# Add packages to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "broker"))
sys.path.insert(0, str(root_path / "bifrost"))

from broker.client import GPUClient
from bifrost.client import BifrostClient
from engine.backends.vllm import VLLMConfig


async def deploy_vllm_remote():
    """Deploy vLLM backend on remote GPU instance."""
    print("🚀 Deploying vLLM backend on remote GPU...")
    
    # Step 1: Provision GPU with broker
    print("\n📡 Step 1: Provisioning GPU instance...")
    gpu_client = GPUClient()
    
    # Search for cheap GPU (as requested)
    instance = gpu_client.create(
        cloud_type="secure",  # As requested
        max_price=0.30,       # As requested
        min_vram=4,           # GPT-2 needs very little
        name="vllm-gpt2-test"
    )
    
    print(f"✅ GPU instance created: {instance.id}")
    print(f"   Waiting for SSH to be ready...")
    
    # Wait for SSH to be ready
    instance.wait_until_ssh_ready(timeout=600)  # 10 minutes max
    ssh_conn = f"{instance.ssh_username}@{instance.public_ip}:{instance.ssh_port}"
    print(f"✅ SSH ready: {ssh_conn}")
    
    # Step 2: Deploy vLLM via bifrost
    print(f"\n🌈 Step 2: Deploying vLLM via bifrost...")
    bifrost = BifrostClient(ssh_conn)
    
    # Install vLLM (this may take a while)
    print("   Installing vLLM...")
    install_result = await bifrost.run(
        "pip install vllm requests",
        timeout=600  # 10 minutes for installation
    )
    
    if not install_result.success:
        print(f"❌ vLLM installation failed: {install_result.stderr}")
        return None, None
    
    print("✅ vLLM installed successfully")
    
    # Step 3: Create and deploy vLLM startup script
    print("   Creating vLLM startup script...")
    
    config = VLLMConfig.for_gpt2_testing(port=8000)
    vllm_args = " ".join(config.to_vllm_args())
    
    startup_script = f'''#!/bin/bash
echo "🚀 Starting vLLM server..."
echo "Model: {config.model_name}"
echo "Port: {config.port}"
echo "Args: {vllm_args}"

# Start vLLM server
python -m vllm.entrypoints.api_server {vllm_args}
'''
    
    # Upload and run the startup script
    await bifrost.run(f"cat > start_vllm.sh << 'EOF'\n{startup_script}\nEOF")
    await bifrost.run("chmod +x start_vllm.sh")
    
    print("   Starting vLLM server in background...")
    job = await bifrost.run_detached("./start_vllm.sh")
    print(f"✅ vLLM job started: {job.job_id}")
    
    # Step 4: Wait for vLLM to be ready
    print(f"\n⏳ Step 3: Waiting for vLLM to be ready...")
    vllm_ready = False
    
    for attempt in range(60):  # Wait up to 5 minutes
        try:
            # Check job status
            status = await bifrost.get_job_status(job.job_id)
            if status and status.status.value == "failed":
                print("❌ vLLM job failed!")
                logs = await bifrost.get_logs(job.job_id, lines=50)
                print("Last 50 lines of logs:")
                print(logs)
                break
                
            # Test vLLM health via SSH tunnel
            health_result = await bifrost.run(
                f"curl -s http://localhost:{config.port}/v1/models",
                timeout=5
            )
            
            if health_result.success and "openai-community/gpt2" in health_result.stdout:
                vllm_ready = True
                print("✅ vLLM server is ready!")
                break
                
        except Exception as e:
            pass  # Keep trying
            
        if attempt % 10 == 0:
            print(f"   Still waiting... ({attempt * 5}s elapsed)")
        time.sleep(5)
    
    if not vllm_ready:
        print("❌ vLLM server did not become ready in time")
        return instance, None
    
    # Step 5: Set up SSH tunnel for local access
    print(f"\n🔗 Step 4: Setting up SSH tunnel...")
    tunnel_cmd = [
        "ssh", "-L", f"{config.port}:localhost:{config.port}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        ssh_conn, "-N"
    ]
    
    print(f"   Tunnel command: {' '.join(tunnel_cmd)}")
    tunnel_process = subprocess.Popen(tunnel_cmd)
    
    # Wait a moment for tunnel to establish
    time.sleep(3)
    
    # Step 6: Test inference locally
    print(f"\n🧪 Step 5: Testing inference...")
    local_url = f"http://localhost:{config.port}"
    
    try:
        response = requests.post(
            f"{local_url}/v1/chat/completions",
            json={
                "model": config.model_name,
                "messages": [{"role": "user", "content": "Hello from remote GPU!"}],
                "max_tokens": 20,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            message = data["choices"][0]["message"]["content"]
            print(f"✅ Inference successful!")
            print(f"   Response: '{message.strip()}'")
        else:
            print(f"❌ Inference failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Inference test error: {e}")
    
    print(f"\n🎉 Deployment complete!")
    print(f"   Local endpoint: {local_url}")
    print(f"   SSH connection: {ssh_conn}")
    print(f"   GPU instance: {instance.id}")
    print(f"   vLLM job: {job.job_id}")
    print(f"\n   To access vLLM:")
    print(f"   curl -X POST {local_url}/v1/chat/completions \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"model\":\"{config.model_name}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello!\"}}],\"max_tokens\":10}}'")
    
    return instance, tunnel_process


async def main():
    """Main deployment function."""
    print("=" * 60)
    print("Remote vLLM Deployment via Broker + Bifrost")
    print("=" * 60)
    
    try:
        instance, tunnel = await deploy_vllm_remote()
        
        if instance:
            print(f"\n⚠️  Remember to terminate the GPU instance when done:")
            print(f"   broker terminate {instance.id}")
            
            if tunnel:
                print(f"\n🔗 SSH tunnel running (PID: {tunnel.pid})")
                print(f"   Kill tunnel with: kill {tunnel.pid}")
                
                # Keep tunnel alive
                input("\nPress Enter to terminate everything...")
                tunnel.terminate()
                
        else:
            print("❌ Deployment failed")
            
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted")
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())