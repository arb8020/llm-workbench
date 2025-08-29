#!/usr/bin/env python3
"""Start vLLM server with GPT-2 model."""

import os
import subprocess
import sys
from pathlib import Path

# Add engine to path so we can import our config
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.backends.vllm import VLLMConfig


def main():
    """Start vLLM server with optimized GPT-2 config."""
    print("🚀 Starting vLLM server with GPT-2...")
    
    # Create config optimized for cheap GPUs
    config = VLLMConfig.for_gpt2_testing(port=8000)
    
    print(f"Model: {config.model_name}")
    print(f"Port: {config.port}")
    print(f"GPU Memory Util: {config.gpu_memory_utilization}")
    print(f"Max Context: {config.max_model_len}")
    
    # Build vLLM command using uv or virtual environment python
    if os.path.exists("uv.lock"):
        # Use uv run for uv-managed projects
        cmd = ["uv", "run", "python", "-m", "vllm.entrypoints.api_server"] + config.to_vllm_args()
    elif os.path.exists(".venv/bin/python"):
        # Use virtual environment python
        cmd = [".venv/bin/python", "-m", "vllm.entrypoints.api_server"] + config.to_vllm_args()
    else:
        # Fallback to system python
        cmd = ["python", "-m", "vllm.entrypoints.api_server"] + config.to_vllm_args()
    
    print(f"Command: {' '.join(cmd)}")
    print("🔄 Starting server...")
    
    # Start vLLM server (this will run indefinitely)
    try:
        # Run with output capture for better debugging
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=False,  # Let output stream to console
            text=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed with exit code {e.returncode}")
        print("🔍 Debug information:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Working directory: {os.getcwd()}")
        print(f"   Python path: {sys.path[:3]}...")
        print(f"   UV lock exists: {os.path.exists('uv.lock')}")
        print(f"   Venv exists: {os.path.exists('.venv/bin/python')}")
        
        # Try to run a simple test to see if vllm is available
        try:
            test_cmd = cmd[:-len(config.to_vllm_args())] + ["-c", "import vllm; print('vLLM available')"]
            print(f"🧪 Testing vLLM availability: {' '.join(test_cmd)}")
            test_result = subprocess.run(test_cmd, capture_output=True, text=True)
            if test_result.returncode == 0:
                print(f"✅ vLLM import test passed: {test_result.stdout.strip()}")
            else:
                print(f"❌ vLLM import test failed: {test_result.stderr}")
        except Exception as test_e:
            print(f"❌ Could not test vLLM availability: {test_e}")
            
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()