#!/usr/bin/env python3
"""
Outlier Features Analysis with automated GPU deployment via broker/bifrost.

This script automatically:
1. Provisions a GPU with sufficient disk space
2. Deploys the codebase
3. Runs outlier analysis on specified model
4. Syncs results back to local
5. Cleans up GPU instance

Usage:
    python examples/outlier_features_moe/deploy_and_analyze.py --model "Qwen/Qwen3-30B-A3B"
    python examples/outlier_features_moe/deploy_and_analyze.py --model "allenai/OLMoE-1B-7B-0125-Instruct" --keep-running
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import broker and bifrost for deployment
from broker.client import GPUClient
from bifrost.client import BifrostClient

logger = logging.getLogger(__name__)


def deploy_outlier_analysis_instance(
    model_name: str,
    min_vram: int = None,  # Will be auto-estimated if None
    min_cpu_ram: int = 64,  # GB of CPU RAM for large MoE model loading
    max_price: float = 3.50,
    container_disk: int = 150,
    volume_disk: int = 0,  # Volume disk size in GB (for large model caching)
    num_sequences: int = 4,
    sequence_length: int = 2048,
    batch_size: int = 1,
    threshold: float = 6.0,
    safety_factor: float = 1.3,
    gpu_count: int = 1,
    gpu_filter: str = None,  # Optional GPU name filter (e.g., "A100", "H100", "RTX 4090")
    chunk_layers: int = None  # Number of layers to process at once
) -> dict:
    """Deploy GPU instance and run outlier analysis."""
    
    # Validate gpu_count
    if not 1 <= gpu_count <= 8:
        raise ValueError(f"gpu_count must be between 1 and 8, got {gpu_count}")
    
    print(f"🚀 Starting outlier analysis deployment for {model_name}...")
    if gpu_count > 1:
        print(f"🔥 Multi-GPU setup: {gpu_count}x GPUs with device_map='auto'")
    
    # 0. ESTIMATE VRAM REQUIREMENTS
    if min_vram is None:
        print(f"🔍 Estimating VRAM requirements for {model_name}...")
        from scripts.estimate_vram import estimate_vram_requirements
        vram_estimate = estimate_vram_requirements(model_name, safety_factor=safety_factor, sequence_length=sequence_length, batch_size=batch_size)
        min_vram = vram_estimate['recommended_vram']
        print(f"📊 Estimated VRAM: {min_vram}GB (effective params: {vram_estimate['effective_params_billions']:.1f}B)")
    
    # 1. PROVISION GPU
    gpu_desc = f"{gpu_count}x GPU" if gpu_count > 1 else "GPU"
    disk_desc = f"{container_disk}GB container"
    if volume_disk > 0:
        disk_desc += f" + {volume_disk}GB volume"
    print(f"📡 Creating {gpu_desc} instance (min {min_vram}GB VRAM per GPU, {min_cpu_ram}GB CPU RAM, max ${max_price}/hr, {disk_desc})...")
    gpu_client = GPUClient()
    
    # Build query for GPU with minimum VRAM, CPU RAM, max price, and NVIDIA manufacturer
    query = (gpu_client.vram_gb >= min_vram) & (gpu_client.memory_gb >= min_cpu_ram) & (gpu_client.price_per_hour <= max_price) & (gpu_client.manufacturer == 'Nvidia')
    
    # Add GPU type filter if specified
    if gpu_filter:
        query = query & (gpu_client.gpu_type.contains(gpu_filter))
        print(f"🔍 Searching for {gpu_count}x {gpu_filter} GPU: ≥{min_vram}GB VRAM per GPU, ≥{min_cpu_ram}GB CPU RAM, ≤${max_price}/hr")
    else:
        print(f"🔍 Searching for {gpu_count}x NVIDIA GPU: ≥{min_vram}GB VRAM per GPU, ≥{min_cpu_ram}GB CPU RAM, ≤${max_price}/hr")
    
    gpu_instance = gpu_client.create(
        query=query,
        name=f"outlier-analysis-{model_name.replace('/', '-').lower()}",
        cloud_type="secure",
        gpu_count=gpu_count,
        sort=lambda x: x.price_per_hour,  # Sort by price (cheapest first)
        reverse=False,
        container_disk_gb=container_disk,  # Container disk for system files
        volume_disk_gb=volume_disk if volume_disk > 0 else None  # Volume disk for model caching
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
    
    # Deploy the codebase to remote workspace with outlier dependencies
    print("📦 Installing outlier analysis dependencies (nnsight, triton 3.4+, no vllm conflict)...")
    workspace_path = bifrost_client.push(uv_extra="outlier")  # Use outlier-specific dependencies
    bifrost_client.exec("echo 'Codebase deployed successfully'")
    print(f"✅ Code deployed to: {workspace_path}")
    print("✅ Dependencies installed successfully")
    
    # 4. CHECK DISK SPACE
    print("💾 Checking disk space...")
    disk_info = bifrost_client.exec("df -h /")
    print(f"Disk space:\n{disk_info}")
    
    # Configure HuggingFace cache to use volume disk if available
    if volume_disk > 0:
        print("🗂️  Configuring HuggingFace cache to use volume disk...")
        # Check if volume is mounted and configure HF cache
        volume_check = bifrost_client.exec("df -h | grep -E '(volume|workspace)' || echo 'NO_VOLUME'")
        if "NO_VOLUME" not in volume_check:
            # Set HF cache to volume disk location
            hf_cache_cmd = '''
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
echo "HF cache configured to use volume disk at /workspace/hf_cache"
'''
            bifrost_client.exec(hf_cache_cmd)
            print("✅ HuggingFace cache configured to use volume disk")
        else:
            print("⚠️  Volume disk not detected, using container disk for HF cache")
    
    # 5. RUN OUTLIER ANALYSIS IN TMUX
    print(f"🔬 Starting outlier analysis for {model_name} in tmux session...")
    
    # Create analysis command
    # Set HF environment variables based on volume disk availability
    hf_env = ""
    if volume_disk > 0:
        hf_env = """export HF_HOME=/workspace/hf_cache 2>/dev/null || export HF_HOME=~/.cache/huggingface && \\
export HUGGINGFACE_HUB_CACHE=$HF_HOME && \\
export TRANSFORMERS_CACHE=$HF_HOME/transformers && \\"""
    
    analysis_cmd = f"""cd ~/.bifrost/workspace && \\
{hf_env}
uv run python examples/outlier_features_moe/run_full_analysis.py \\
    --model "{model_name}" \\
    --num-sequences {num_sequences} \\
    --sequence-length {sequence_length} \\
    --batch-size {batch_size} \\
    --threshold {threshold} \\
    2>&1 | tee examples/outlier_features_moe/outlier_analysis.log"""
    
    # Create log file first, then run command with simple error capture
    simple_cmd = f"""cd ~/.bifrost/workspace/examples/outlier_features_moe && \\
exec > outlier_analysis.log 2>&1 && \\
{hf_env}uv run python run_full_analysis.py \\
    --model "{model_name}" \\
    --num-sequences {num_sequences} \\
    --sequence-length {sequence_length} \\
    --batch-size {batch_size} \\
    --threshold {threshold}{' --chunk-layers ' + str(chunk_layers) if chunk_layers else ''} \\
|| echo "ANALYSIS FAILED with exit code $?"
"""
    
    tmux_cmd = f"tmux new-session -d -s outlier-analysis '{simple_cmd}'"
    bifrost_client.exec(tmux_cmd)
    
    print("✅ Outlier analysis started in tmux session 'outlier-analysis'")
    
    # 6. ANALYSIS STARTED - NO POLLING
    print("✅ Analysis started successfully!")
    print("📋 To monitor progress in real-time:")
    print(f"   ssh {ssh_connection.replace(':', ' -p ')}")
    print(f"   cd ~/.bifrost/workspace/examples/outlier_features_moe")
    print(f"   tail -f outlier_analysis.log")
    print("")
    print("📋 Or check recent progress with:")
    print(f"   bifrost exec '{ssh_connection}' 'cd ~/.bifrost/workspace/examples/outlier_features_moe && tail -20 outlier_analysis.log'")
    print("")
    print("📋 To check/attach to tmux session:")
    print(f"   bifrost exec '{ssh_connection}' 'tmux list-sessions'")
    print(f"   ssh {ssh_connection.replace(':', ' -p ')} -t 'tmux attach-session -t outlier-analysis'")
    print("")
    print("⏳ Analysis will take 10-30 minutes depending on model size...")
    
    # 7. RETURN CONNECTION INFO
    connection_info = {
        "instance_id": gpu_instance.id,
        "ssh": ssh_connection,
        "provider": gpu_instance.provider,
        "model": model_name,
        "status": "running"
    }
    
    print(f"\n🎉 Outlier analysis deployment complete!")
    print(f"   Instance ID: {gpu_instance.id}")
    print(f"   SSH: {ssh_connection}")
    print(f"   Status: {connection_info['status']}")
    
    print("\n🔧 Management commands:")
    print(f"   # Check tmux session: bifrost exec '{ssh_connection}' 'tmux list-sessions'")
    print(f"   # View analysis log: bifrost exec '{ssh_connection}' 'cd ~/.bifrost/workspace/examples/outlier_features_moe && cat outlier_analysis.log'")
    print(f"   # List results: bifrost exec '{ssh_connection}' 'cd ~/.bifrost/workspace && find examples/outlier_features_moe -name \"*.json\" -o -name \"*.log\"'")
    print(f"   # Terminate GPU: broker terminate {gpu_instance.id}")
    
    return connection_info


def sync_results_from_remote(bifrost_client: BifrostClient, local_output_dir: Path) -> None:
    """Sync analysis results from remote GPU to local directory using efficient SFTP."""
    print("💾 Syncing results from remote GPU...")
    
    # Create local results directory
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check what files exist on remote
    remote_files = bifrost_client.exec(
        "cd ~/.bifrost/workspace && "
        "find examples/outlier_features_moe -name '*.json' -o -name '*.log' -o -name 'full_analysis_results' -type d | head -20"
    )
    
    if not remote_files or remote_files.strip() == "":
        print("⚠️ No result files found on remote")
        return
    
    print(f"📁 Found result files:\n{remote_files}")
    
    # Sync main log file using SFTP
    try:
        result = bifrost_client.download_files(
            remote_path="~/.bifrost/workspace/examples/outlier_features_moe/outlier_analysis.log",
            local_path=str(local_output_dir / "outlier_analysis.log")
        )
        if result and result.success and result.files_copied > 0:
            print(f"✅ Synced: outlier_analysis.log ({result.files_copied} files, {result.total_bytes} bytes)")
        else:
            print("⚠️ Log file not found or empty")
    except Exception as e:
        print(f"⚠️ Could not sync log file: {e}")
    
    # Sync outlier analysis results (metadata and summary only, skip large activation files)
    try:
        # Check if results directory exists
        results_check = bifrost_client.exec(
            "cd ~/.bifrost/workspace && "
            "ls -la examples/outlier_features_moe/full_analysis_results/ 2>/dev/null || echo 'NO_RESULTS_DIR'"
        )
        
        if "NO_RESULTS_DIR" not in results_check:
            print("📦 Downloading outlier analysis metadata (skipping large activation files)...")
            
            # Create local results directory
            local_results_dir = local_output_dir / "full_analysis_results"
            local_results_dir.mkdir(exist_ok=True)
            
            # Get list of run directories
            run_dirs = bifrost_client.exec(
                "cd ~/.bifrost/workspace/examples/outlier_features_moe/full_analysis_results && "
                "ls -1 | grep '^run_'"
            ).strip().split('\n')
            
            total_files = 0
            for run_dir in run_dirs:
                if run_dir and run_dir.startswith('run_'):
                    print(f"  Syncing metadata from {run_dir}...")
                    
                    # Create local run directory
                    local_run_dir = local_results_dir / run_dir
                    local_run_dir.mkdir(exist_ok=True)
                    
                    # Only sync metadata.json files (skip .pt activation files)
                    try:
                        result = bifrost_client.download_files(
                            remote_path=f"~/.bifrost/workspace/examples/outlier_features_moe/full_analysis_results/{run_dir}/metadata.json",
                            local_path=str(local_run_dir / "metadata.json")
                        )
                        if result and result.success and result.files_copied > 0:
                            total_files += result.files_copied
                    except Exception as e:
                        print(f"    ⚠️ No metadata.json in {run_dir}: {e}")
            
            print(f"✅ Synced {total_files} metadata files (activation files skipped to save bandwidth)")
        else:
            print("⚠️ No full_analysis_results directory found on remote")
    
    except Exception as e:
        print(f"⚠️ Could not sync results directory: {e}")
    
    print(f"✅ Results synced to local: {local_output_dir}")


def main(
    model_name: str,
    keep_running: bool = False,
    min_vram: int = None,  # Auto-estimate if None
    min_cpu_ram: int = 64,  # GB of CPU RAM for MoE models
    max_price: float = 3.50,
    container_disk: int = 150,
    volume_disk: int = 0,
    num_sequences: int = 4,
    sequence_length: int = 2048,
    batch_size: int = 1,
    threshold: float = 6.0,
    safety_factor: float = 1.3,
    gpu_count: int = 1,
    gpu_filter: str = None
):
    """Run outlier analysis using automated GPU deployment."""
    from datetime import datetime
    
    # Create timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace('/', '_').replace('-', '_')
    experiment_name = f"outlier_analysis_{model_safe_name}_{timestamp}"
    
    print(f"🎯 Outlier Features Analysis - Automated Deployment")
    print(f"📅 Experiment: {experiment_name}")
    print(f"🤖 Model: {model_name}")
    print("=" * 60)
    
    # 1. DEPLOY AND RUN ANALYSIS
    print("🚀 Step 1: Deploying and running outlier analysis...")
    connection_info = deploy_outlier_analysis_instance(
        model_name=model_name,
        min_vram=min_vram,
        min_cpu_ram=min_cpu_ram,
        max_price=max_price,
        container_disk=container_disk,
        volume_disk=volume_disk,
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        batch_size=batch_size,
        threshold=threshold,
        safety_factor=safety_factor,
        gpu_count=gpu_count,
        gpu_filter=gpu_filter
    )
    
    bifrost_client = BifrostClient(connection_info["ssh"])
    
    print(f"\n📁 When analysis completes, sync results with:")
    output_dir = Path(f"remote_results/{experiment_name}")
    print(f"   # Create local results directory")
    print(f"   mkdir -p {output_dir}")
    print(f"   # Download analysis log")
    print(f"   bifrost copy '{connection_info['ssh']}:~/.bifrost/workspace/examples/outlier_features_moe/outlier_analysis.log' {output_dir}/")
    print(f"   # Download any result files")
    print(f"   bifrost exec '{connection_info['ssh']}' 'cd ~/.bifrost/workspace && find examples/outlier_features_moe -name \"*.json\" -o -name \"full_analysis_results\" -type d'")
    
    finally:
        # 4. CLEANUP (CONDITIONAL)
        if not keep_running:
            print(f"\n🧹 Step 3: Cleaning up GPU instance...")
            gpu_client = GPUClient()
            
            try:
                # Stop analysis session
                print("   Stopping analysis session...")
                bifrost_client.exec("tmux kill-session -t outlier-analysis 2>/dev/null || true")
                
                # Terminate the GPU instance
                print(f"   Terminating GPU instance {connection_info['instance_id']}...")
                # Use broker CLI for termination
                import subprocess
                result = subprocess.run(
                    ["broker", "instances", "terminate", connection_info['instance_id']],
                    input="y\n",
                    text=True,
                    capture_output=True
                )
                
                if result.returncode == 0:
                    print("✅ Cleanup complete")
                else:
                    print(f"⚠️ Cleanup may have failed: {result.stderr}")
                    print(f"   Manual cleanup: broker instances terminate {connection_info['instance_id']}")
                
            except Exception as e:
                print(f"⚠️ Cleanup error (instance may still be running): {e}")
                print(f"   Manual cleanup: broker instances terminate {connection_info['instance_id']}")
        else:
            print(f"\n🎯 Step 3: Keeping GPU instance running (--keep-running flag)")
            print(f"   Instance ID: {connection_info['instance_id']}")
            print(f"   SSH: {connection_info['ssh']}")
            print(f"   Manual cleanup: broker instances terminate {connection_info['instance_id']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Outlier Features Analysis with GPU Deployment")
    
    # Core analysis args
    parser.add_argument("--model", required=True, help="Model name (e.g. 'Qwen/Qwen3-30B-A3B')")
    parser.add_argument("--num-sequences", type=int, default=4, help="Number of text sequences (default: 4)")
    parser.add_argument("--sequence-length", type=int, default=2048, help="Sequence length in tokens (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--threshold", type=float, default=6.0, help="Outlier magnitude threshold (default: 6.0)")
    parser.add_argument("--chunk-layers", type=int, default=None, help="Number of layers to process at once (default: process all together)")
    
    # Deployment args
    parser.add_argument("--keep-running", action="store_true", 
                       help="Keep GPU instance running after analysis")
    parser.add_argument("--min-vram", type=int, default=None, 
                       help="Minimum VRAM in GB (default: auto-estimate)")
    parser.add_argument("--min-cpu-ram", type=int, default=64,
                       help="Minimum CPU RAM in GB (default: 64)")
    parser.add_argument("--max-price", type=float, default=3.50, 
                       help="Maximum price per hour (default: 3.50)")
    parser.add_argument("--container-disk", type=int, default=150,
                       help="Container disk size in GB (default: 150)")
    parser.add_argument("--volume-disk", type=int, default=0,
                       help="Volume disk size in GB for large model caching (default: 0)")
    parser.add_argument("--gpu-count", type=int, default=1, choices=range(1, 9),
                       help="Number of GPUs to provision (1-8, default: 1)")
    parser.add_argument("--gpu-filter", type=str, default=None,
                       help="GPU type filter (e.g., 'A100', 'H100', 'RTX 4090')")
    parser.add_argument("--safety-factor", type=float, default=1.3,
                       help="VRAM safety factor multiplier (default: 1.3)")
    
    args = parser.parse_args()
    
    main(
        model_name=args.model,
        keep_running=args.keep_running,
        min_vram=args.min_vram,
        min_cpu_ram=args.min_cpu_ram,
        max_price=args.max_price,
        container_disk=args.container_disk,
        volume_disk=args.volume_disk,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        threshold=args.threshold,
        safety_factor=args.safety_factor,
        gpu_count=args.gpu_count,
        gpu_filter=args.gpu_filter,
        chunk_layers=args.chunk_layers
    )
