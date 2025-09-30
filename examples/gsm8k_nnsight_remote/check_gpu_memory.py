#!/usr/bin/env python3
"""
GPU Memory Checker for NNsight Server

This script connects to a remote GPU instance and checks:
1. GPU memory usage (via nvidia-smi)
2. Process memory (to verify no model duplication)
3. Number of Python processes running

Usage:
    python check_gpu_memory.py --gpu-id gpu_12345
    python check_gpu_memory.py --ssh root@194.68.245.163:22
"""

import argparse
import sys
from bifrost.client import BifrostClient
from broker.client import GPUClient
import json


def check_gpu_memory_via_ssh(ssh_command: str):
    """Check GPU memory using SSH"""
    bc = BifrostClient()

    print("🔍 Checking GPU Memory Usage")
    print("="*60)

    # Run nvidia-smi to get GPU memory usage
    print("\n1. GPU Memory (nvidia-smi):")
    result = bc.exec(ssh_command, "nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits")
    if result.returncode == 0:
        memory_info = result.stdout.strip()
        used, total, free = map(int, memory_info.split(','))
        print(f"   Used: {used} MB")
        print(f"   Total: {total} MB")
        print(f"   Free: {free} MB")
        print(f"   Usage: {used/total*100:.1f}%")

        # Estimate model size (Qwen3-0.6B is ~600M params * 2 bytes/param for fp16 = ~1.2GB)
        expected_model_size_mb = 1200
        if used < expected_model_size_mb * 2:
            print(f"   ✅ Memory usage suggests single model instance (~{expected_model_size_mb}MB expected)")
        else:
            print(f"   ⚠️  High memory usage - possible model duplication? (>{expected_model_size_mb*2}MB)")
    else:
        print(f"   ❌ Could not get GPU memory info: {result.stderr}")

    # Check Python processes
    print("\n2. Python Processes:")
    result = bc.exec(ssh_command, "ps aux | grep python | grep -v grep | wc -l")
    if result.returncode == 0:
        num_processes = int(result.stdout.strip())
        print(f"   Running Python processes: {num_processes}")
        if num_processes == 1:
            print(f"   ✅ Single Python process (good - model not duplicated across processes)")
        else:
            print(f"   ℹ️  Multiple Python processes detected")

    # Show detailed process info
    print("\n3. Detailed Process Info:")
    result = bc.exec(ssh_command, "ps aux | grep 'python.*server' | grep -v grep | head -5")
    if result.returncode == 0:
        processes = result.stdout.strip()
        if processes:
            print("   Server processes:")
            for line in processes.split('\n'):
                # Parse ps aux output to get RSS (memory)
                parts = line.split()
                if len(parts) >= 6:
                    user, pid, cpu, mem, vsz, rss = parts[:6]
                    command = ' '.join(parts[10:])[:60]
                    print(f"   PID {pid}: RSS={int(rss)/1024:.1f}MB, MEM={mem}%, CMD={command}")
        else:
            print("   No server processes found")

    # Check for torch tensors in memory
    print("\n4. PyTorch Memory Stats (if available):")
    python_script = """
import torch
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
else:
    print("CUDA not available")
"""
    result = bc.exec(ssh_command, f"python3 -c '{python_script}'")
    if result.returncode == 0:
        print(f"   {result.stdout.strip()}")
    else:
        print(f"   ⚠️  Could not get PyTorch memory stats")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Check GPU memory usage on remote instance")
    parser.add_argument("--gpu-id", help="GPU instance ID")
    parser.add_argument("--ssh", help="Direct SSH string (e.g., root@IP:PORT)")
    args = parser.parse_args()

    if args.gpu_id:
        # Get SSH command from GPU ID
        gc = GPUClient()
        instance = gc.get_instance(args.gpu_id)
        if not instance:
            print(f"❌ GPU instance {args.gpu_id} not found")
            sys.exit(1)

        bc = BifrostClient()
        ssh_command = bc.get_ssh_command(instance)
        print(f"📡 Connected to GPU: {args.gpu_id}")
        print(f"   SSH: {ssh_command}")

    elif args.ssh:
        ssh_command = args.ssh
        print(f"📡 Using SSH: {ssh_command}")

    else:
        print("❌ Must provide either --gpu-id or --ssh")
        sys.exit(1)

    check_gpu_memory_via_ssh(ssh_command)


if __name__ == "__main__":
    main()