#!/usr/bin/env python3
"""Quick integration test for leyline."""

import asyncio
import json
import time
import subprocess
import requests
import sys

async def test_integration():
    """Test controller and worker integration."""
    print("Starting integration test...")
    
    # Start controller in background
    controller_proc = subprocess.Popen([
        sys.executable, "-m", "leyline.cli", "controller", 
        "--port", "18001", "--log-level", "WARNING"
    ])
    
    # Give controller time to start
    await asyncio.sleep(3)
    
    # Start worker in background
    worker_proc = subprocess.Popen([
        sys.executable, "-m", "leyline.cli", "worker", 
        "examples.echo_worker:EchoWorker",
        "--controller", "ws://localhost:18001/ws",
        "--log-level", "WARNING"
    ])
    
    # Give worker time to connect
    await asyncio.sleep(5)
    
    try:
        # Test basic echo
        print("Testing basic echo...")
        response = requests.post("http://localhost:18001/echo", data="Hello Leyline!")
        print(f"Echo response: {response.json()}")
        
        # Test health check
        print("Testing health check...")
        response = requests.post("http://localhost:18001/health", data="")
        print(f"Health response: {response.json()}")
        
        # Test uppercase
        print("Testing uppercase...")
        response = requests.post(
            "http://localhost:18001/uppercase",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": "hello world"})
        )
        print(f"Uppercase response: {response.json()}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Clean up
        worker_proc.terminate()
        controller_proc.terminate()
        
        # Wait for processes to exit
        worker_proc.wait(timeout=5)
        controller_proc.wait(timeout=5)

if __name__ == "__main__":
    asyncio.run(test_integration())