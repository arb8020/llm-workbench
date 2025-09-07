#!/usr/bin/env python3
"""
Deploy and test transformers compatibility on a cheap GPU.
"""

import subprocess
import sys
import time
from typing import Optional

def run_broker_command(cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run broker command and return exit code, stdout, stderr."""
    print(f"🔄 Running: {cmd}")
    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"

def deploy_and_test() -> int:
    """Deploy to cheap GPU and run transformers compatibility test."""
    print("🚀 Deploying transformers compatibility test to cheap GPU...")
    
    # Deploy to cheapest available GPU
    deploy_cmd = """broker run 
        --image pytorch/pytorch:2.4.0-devel-cuda12.1-cudnn9-runtime 
        --gpu-count 1 
        --gpu-filter "RTX,GTX,Tesla" 
        --max-price 0.5 
        --min-vram 8 
        --script-inline '
            cd /workspace &&
            git clone https://github.com/yourusername/llm-workbench.git . &&
            pip install uv &&
            uv sync --extra transformers-test &&
            cd examples/outlier_features_moe &&
            python scripts/smoke_test_transformers.py
        '""".replace('\n', ' ')
    
    # For now, let's use a simpler bifrost approach
    print("📡 Starting transformers test instance...")
    
    # Create a simple test script
    test_script = '''
#!/bin/bash
set -e
cd /workspace
echo "🔧 Installing dependencies..."
pip install uv
uv sync --extra transformers-test

echo "🔬 Running transformers smoke test..."
cd examples/outlier_features_moe
python scripts/smoke_test_transformers.py

echo "✅ Test completed!"
'''
    
    # We'll use broker for this
    print("ℹ️  For now, run this manually:")
    print("broker instances create --gpu-count 1 --gpu-filter 'RTX' --max-price 0.5")
    print("# Then SSH in and run the smoke test")
    
    return 0

if __name__ == "__main__":
    sys.exit(deploy_and_test())