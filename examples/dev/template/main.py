#!/usr/bin/env python3
"""
Basic GPU detection and test script.
"""

import subprocess
import sys

def check_nvidia_smi():
    """Check nvidia-smi output."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("=== nvidia-smi ===")
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("nvidia-smi not found")
        return False

def check_jax():
    """Check JAX GPU detection."""
    try:
        import jax
        devices = jax.devices()
        print(f"=== JAX ===")
        print(f"Available devices: {devices}")
        if jax.devices('gpu'):
            print(f"GPU devices: {jax.devices('gpu')}")
        return True
    except ImportError:
        print("JAX not installed")
        return False

def check_torch():
    """Check PyTorch GPU detection."""
    try:
        import torch
        print(f"=== PyTorch ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False

def main():
    print("🔍 GPU Detection Test")
    print("=" * 50)

    check_nvidia_smi()
    print()
    check_jax()
    print()
    check_torch()

    print("✅ GPU detection complete!")

if __name__ == "__main__":
    main()