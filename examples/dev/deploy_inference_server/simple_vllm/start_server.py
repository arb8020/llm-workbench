#!/usr/bin/env python3
"""Minimal vLLM server startup script."""

import subprocess


def main():
    """Start vLLM server with GPT-2."""
    # Simple vLLM command with optimized GPT-2 settings
    cmd = [
        "uv", "run", "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "openai-community/gpt2",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", "0.6",
        "--max-model-len", "512",
        "--disable-log-stats"
    ]
    
    print("Starting vLLM server...")
    print("Model: openai-community/gpt2")
    print("Server URL: http://0.0.0.0:8000")
    print(f"Command: {' '.join(cmd)}")
    
    # Start server
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
