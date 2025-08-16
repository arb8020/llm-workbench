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
    
    # Build vLLM command using virtual environment python
    venv_python = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python"
    cmd = [venv_python, "-m", "vllm.entrypoints.api_server"] + config.to_vllm_args()
    
    print(f"Command: {' '.join(cmd)}")
    print("🔄 Starting server...")
    
    # Start vLLM server (this will run indefinitely)
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed with exit code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()