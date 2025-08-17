#!/usr/bin/env python3
"""Simple vLLM deployment using broker CLI + bifrost like the README examples."""

import subprocess
import time
import requests
import sys
import json
from pathlib import Path


def run_command(cmd, description, timeout=300):
    """Run a command and return success, stdout, stderr."""
    print(f"🔄 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True, result.stdout, result.stderr
        else:
            print(f"❌ {description} - Failed (exit code {result.returncode})")
            print(f"   stderr: {result.stderr}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - Timeout after {timeout}s")
        return False, "", "Timeout"
    except Exception as e:
        print(f"💥 {description} - Error: {e}")
        return False, "", str(e)


def main():
    """Main deployment using broker CLI + bifrost CLI like README."""
    print("=" * 60)
    print("Simple vLLM Deployment via Broker CLI + Bifrost CLI")
    print("=" * 60)
    
    # Step 1: Create GPU instance using broker CLI
    print("\n📡 Step 1: Provisioning GPU instance...")
    
    # Using exact pattern from README
    create_cmd = (
        "broker create --cloud-type secure --max-price 0.30 --min-vram 8 "
        "--name vllm-gpt2-test --print-ssh"
    )
    
    success, stdout, stderr = run_command(create_cmd, "Creating GPU instance", timeout=600)
    
    if not success:
        print("❌ Failed to create GPU instance")
        return False
        
    # Extract SSH connection from output
    ssh_conn = stdout.strip().split('\n')[-1]  # Last line should be SSH connection
    print(f"✅ SSH connection: {ssh_conn}")
    
    # Step 2: Deploy vLLM using bifrost
    print(f"\n🌈 Step 2: Deploying vLLM via bifrost...")
    
    # Install vLLM
    install_cmd = f'bifrost launch {ssh_conn} "pip install vllm requests"'
    success, stdout, stderr = run_command(install_cmd, "Installing vLLM", timeout=600)
    
    if not success:
        print("❌ Failed to install vLLM")
        return False
    
    # Start vLLM server in background
    vllm_cmd = (
        f'bifrost launch {ssh_conn} '
        f'"python -m vllm.entrypoints.api_server '
        f'--model openai-community/gpt2 '
        f'--host 0.0.0.0 --port 8000 '
        f'--gpu-memory-utilization 0.7 '
        f'--max-model-len 512" --detach'
    )
    
    success, stdout, stderr = run_command(vllm_cmd, "Starting vLLM server", timeout=60)
    
    if not success:
        print("❌ Failed to start vLLM server") 
        return False
        
    # Extract job ID from output
    job_id = None
    for line in stdout.split('\n'):
        if 'Job' in line and 'started successfully' in line:
            # Format: "Job a8ff639d started successfully"
            parts = line.split()
            if len(parts) >= 2:
                job_id = parts[1]
                break
    
    if job_id:
        print(f"✅ vLLM job started: {job_id}")
    else:
        print("⚠️  Could not extract job ID from output")
        print(f"   stdout: {stdout}")
    
    # Step 3: Wait for vLLM to be ready
    print(f"\n⏳ Step 3: Waiting for vLLM to be ready...")
    
    for attempt in range(60):  # Wait up to 5 minutes
        # Check if vLLM is responding
        health_cmd = f'bifrost launch {ssh_conn} "curl -s http://localhost:8000/v1/models"'
        success, stdout, stderr = run_command(health_cmd, f"Health check (attempt {attempt+1})", timeout=10)
        
        if success and "openai-community/gpt2" in stdout:
            print("✅ vLLM server is ready!")
            break
            
        if attempt % 10 == 0 and attempt > 0:
            print(f"   Still waiting... ({attempt * 5}s elapsed)")
            
        time.sleep(5)
    else:
        print("❌ vLLM server did not become ready in time")
        if job_id:
            print("   Checking job status...")
            status_cmd = f'bifrost status {ssh_conn}'
            run_command(status_cmd, "Job status check", timeout=30)
        return False
    
    # Step 4: Set up SSH tunnel (manual step)
    print(f"\n🔗 Step 4: Setting up SSH tunnel...")
    print(f"   Run this command in another terminal:")
    print(f"   ssh -L 8000:localhost:8000 -o StrictHostKeyChecking=no {ssh_conn} -N")
    print(f"   (Keep that terminal open)")
    
    # Wait for user to set up tunnel
    input("\n   Press Enter after setting up the SSH tunnel...")
    
    # Step 5: Test inference locally
    print(f"\n🧪 Step 5: Testing inference...")
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "openai-community/gpt2", 
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
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Inference test error: {e}")
        print(f"   Make sure SSH tunnel is running: ssh -L 8000:localhost:8000 {ssh_conn} -N")
        return False
    
    print(f"\n🎉 Deployment successful!")
    print(f"   Local endpoint: http://localhost:8000")
    print(f"   SSH connection: {ssh_conn}")
    print(f"   vLLM job: {job_id or 'unknown'}")
    print(f"\n   Example usage:")
    print(f"   curl -X POST http://localhost:8000/v1/chat/completions \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"model\":\"openai-community/gpt2\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello!\"}}],\"max_tokens\":10}}'")
    
    print(f"\n⚠️  Remember to clean up when done:")
    # Get instance ID for cleanup
    list_cmd = "broker list --name vllm-gpt2-test --simple"
    success, stdout, stderr = run_command(list_cmd, "Getting instance ID", timeout=30)
    if success:
        for line in stdout.split('\n'):
            if 'vllm-gpt2-test' in line:
                parts = line.split(',')
                if len(parts) >= 1:
                    instance_id = parts[0]
                    print(f"   broker terminate {instance_id}")
                    break
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted")
        sys.exit(1)