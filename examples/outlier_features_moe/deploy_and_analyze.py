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
    chunk_layers: int = None,  # Number of layers to process at once
    log_level: str = "INFO"  # Logging level
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
        from scripts.estimate_vram import estimate_vram_requirements
        vram_estimate = estimate_vram_requirements(model_name, safety_factor=safety_factor, sequence_length=sequence_length, batch_size=batch_size)
        min_vram = vram_estimate['recommended_vram']
        effective_params = vram_estimate.get('effective_params_billions', 0.0)
        print(f"📊 Estimated VRAM: {min_vram}GB (effective params: {effective_params:.1f}B)")
    
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
    if not gpu_instance.wait_until_ssh_ready(timeout=300):  # 5 minutes
        print("❌ Failed to get SSH connection ready")
        sys.exit(1)
    
    ssh_connection = gpu_instance.ssh_connection_string()
    print(f"✅ SSH ready: {ssh_connection}")
    
    # 2. DEPLOY CODE
    bifrost_client = BifrostClient(ssh_connection)
    
    # Deploy the codebase to remote workspace with outlier dependencies
    workspace_path = bifrost_client.push(uv_extra="outlier")  # Use outlier-specific dependencies
    print(f"✅ Code deployed and dependencies installed")
    
    # Configure HuggingFace cache to use volume disk if available
    if volume_disk > 0:
        # Set HF cache to volume disk location (assume volume is mounted if requested)
        hf_cache_cmd = '''
mkdir -p /workspace/hf_cache 2>/dev/null || mkdir -p ~/.cache/huggingface
export HF_HOME=/workspace/hf_cache 2>/dev/null || export HF_HOME=~/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME/transformers
'''
        bifrost_client.exec(hf_cache_cmd)
    
    # 5. RUN OUTLIER ANALYSIS IN TMUX
    print(f"🔬 Starting outlier analysis...")
    
    # Create analysis command
    # Set HF environment variables based on volume disk availability and load .env
    import os
    from dotenv import load_dotenv
    
    # Load .env file to get HF_TOKEN
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    
    hf_env = f"export HF_TOKEN='{hf_token}' && \\\n"
    hf_env += "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \\\n"
    if volume_disk > 0:
        hf_env += """export HF_HOME=/workspace/hf_cache 2>/dev/null || export HF_HOME=~/.cache/huggingface && \\
export HUGGINGFACE_HUB_CACHE=$HF_HOME && \\
export TRANSFORMERS_CACHE=$HF_HOME/transformers && \\"""
    
    # Create log file first, then run command with simple error capture
    simple_cmd = f"""cd ~/.bifrost/workspace/examples/outlier_features_moe && \\
exec > outlier_analysis.log 2>&1 && \\
{hf_env}uv run python run_full_analysis.py \\
    --model "{model_name}" \\
    --num-sequences {num_sequences} \\
    --sequence-length {sequence_length} \\
    --batch-size {batch_size} \\
    --threshold {threshold}{' --chunk-layers ' + str(chunk_layers) if chunk_layers else ''}{' --log-level ' + log_level if log_level != 'INFO' else ''} \\
|| echo "ANALYSIS FAILED with exit code $?"
"""
    
    tmux_cmd = f"tmux new-session -d -s outlier-analysis '{simple_cmd}'"
    bifrost_client.exec(tmux_cmd)
    
    print("✅ Analysis started - will take 10-30 minutes")
    print(f"📊 Monitor: bifrost exec '{ssh_connection}' 'cd ~/.bifrost/workspace/examples/outlier_features_moe && tail -20 outlier_analysis.log'")
    
    # 7. RETURN CONNECTION INFO
    connection_info = {
        "instance_id": gpu_instance.id,
        "ssh": ssh_connection,
        "provider": gpu_instance.provider,
        "model": model_name,
        "status": "running"
    }
    
    print(f"🎉 Deployment complete!")
    print(f"   Instance: {gpu_instance.id} | SSH: {ssh_connection}")
    print(f"   Terminate: broker terminate {gpu_instance.id}")
    
    return connection_info


def sync_results_from_remote(bifrost_client: BifrostClient, local_output_dir: Path) -> None:
    """Sync analysis results from remote GPU to local directory using efficient SFTP."""
    print("💾 Syncing results from remote GPU...")
    
    # Create local results directory
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sync main analysis log file
    try:
        result = bifrost_client.download_files(
            remote_path="~/.bifrost/workspace/examples/outlier_features_moe/outlier_analysis.log",
            local_path=str(local_output_dir / "outlier_analysis.log")
        )
        if result and result.success and result.files_copied > 0:
            print(f"✅ Synced: outlier_analysis.log")
        else:
            print(f"⚠️ Analysis log not ready yet")
    except Exception as e:
        print(f"⚠️ Could not sync analysis log: {e}")
    
    # Sync final aggregated results file
    try:
        result = bifrost_client.download_files(
            remote_path="~/.bifrost/workspace/examples/outlier_features_moe/full_analysis_results/final_analysis_results.json",
            local_path=str(local_output_dir / "final_analysis_results.json")
        )
        if result and result.success and result.files_copied > 0:
            print(f"✅ Synced: final_analysis_results.json")
        else:
            print("ℹ️ Final results not ready yet")
    except Exception as e:
        print(f"ℹ️ Final results not ready yet")
    
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
    gpu_filter: str = None,
    chunk_layers: int = None,
    log_level: str = "INFO"
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
        gpu_filter=gpu_filter,
        chunk_layers=chunk_layers,
        log_level=log_level
    )
    
    bifrost_client = BifrostClient(connection_info["ssh"])
    output_dir = Path(f"remote_results/{experiment_name}")
    
    try:
        # 2. SYNC RESULTS FROM REMOTE
        print("\n💾 Step 2: Syncing results to local...")
        sync_results_from_remote(bifrost_client, output_dir)
        
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
            print(f"\n🎯 Keeping GPU running")
            print(f"   Cleanup: broker instances terminate {connection_info['instance_id']}")


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
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level (default: INFO)")
    
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
        chunk_layers=args.chunk_layers,
        log_level=args.log_level
    )
