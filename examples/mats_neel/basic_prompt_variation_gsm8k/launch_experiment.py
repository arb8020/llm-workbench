#!/usr/bin/env python3
"""
Launch Basic Prompt Variation GSM8K Experiment

Deploys workers and starts experiment in background, then returns.
Use monitor_experiment.py to track progress.

Usage:
    python launch_experiment.py --experiment-name "emotional_pilot" --samples 8 --variants control,frustration,impatience
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

from shared.logging_config import setup_logging

# Import broker and bifrost for deployment
from broker.client import GPUClient
from bifrost.client import BifrostClient

# Import rollouts evaluation framework  
from rollouts.dtypes import Message, Endpoint

# Copy of functions from gsm8k_remote for decoupling

logger = logging.getLogger(__name__)

# =============================================================================
# COPIED FUNCTIONS (decoupled from gsm8k_remote)
# =============================================================================

def load_gsm8k_dataset(output_path: Path, sample_count: int = None) -> None:
    """Load GSM8K from HuggingFace and save as JSONL."""
    try:
        from datasets import load_dataset
        
        print("📚 Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("gsm8k", "main", split="test")
        
        if sample_count:
            dataset = dataset.select(range(min(sample_count, len(dataset))))
        
        print(f"📊 Selected {len(dataset)} samples from GSM8K")
        
        with open(output_path, 'w') as f:
            for i, row in enumerate(dataset):
                answer_text = row["answer"]
                if "####" in answer_text:
                    numeric_answer = answer_text.split("####")[-1].strip()
                else:
                    numeric_answer = "unknown"
                
                gsm8k_sample = {
                    "question": row["question"],
                    "answer": numeric_answer,
                    "sample_id": f"gsm8k_{i+1:04d}"
                }
                f.write(json.dumps(gsm8k_sample) + '\n')
        
        print(f"✅ Saved GSM8K dataset to: {output_path}")
        
    except ImportError:
        print("❌ HuggingFace datasets library not found. Install with: uv pip install datasets")
        raise
    except Exception as e:
        print(f"❌ Error loading GSM8K: {e}")
        raise

def deploy_qwen_vllm_server(min_vram: int = 12, max_price: float = 0.40, 
                           gpu_memory_utilization: float = 0.6, max_model_len: int = 2048,
                           experiment_name: str = "experiment", worker_id: str = "worker") -> dict:
    """Deploy Qwen3-0.6B vLLM server on GPU and return connection info."""
    
    print("🚀 Starting Qwen3-0.6B vLLM deployment...")
    
    # 1. PROVISION GPU
    print(f"📡 Creating GPU instance (min {min_vram}GB VRAM, max ${max_price}/hr)...")
    gpu_client = GPUClient()
    
    # Build query for GPU with minimum VRAM and max price
    query = (gpu_client.vram_gb >= min_vram) & (gpu_client.price_per_hour <= max_price)
    
    # Use experiment name and worker ID for GPU naming
    gpu_name = f"{experiment_name}-{worker_id}"
    
    gpu_instance = gpu_client.create(
        query=query,
        exposed_ports=[8000],  # Expose port 8000 for vLLM
        enable_http_proxy=True,  # Enable RunPod proxy
        name=gpu_name,
        cloud_type="secure",
        sort=lambda x: x.price_per_hour,  # Sort by price (cheapest first)
        reverse=False
    )
    
    print(f"✅ GPU ready: {gpu_instance.id} (name: {gpu_name})")
    
    # Wait for SSH to be ready
    print("⏳ Waiting for SSH connection to be ready...")
    if not gpu_instance.wait_until_ssh_ready(timeout=300):  # 5 minutes
        print("❌ Failed to get SSH connection ready")
        sys.exit(1)
    
    ssh_connection = gpu_instance.ssh_connection_string()
    print(f"✅ SSH ready: {ssh_connection}")
    
    # 2. DEPLOY CODE WITH DEPENDENCIES
    print("📦 Deploying codebase with GSM8K dependencies...")
    bifrost_client = BifrostClient(ssh_connection)
    
    # Deploy the codebase to remote workspace with GSM8K remote dependencies
    workspace_path = bifrost_client.push(uv_extra="examples_gsm8k_remote")
    print(f"✅ Code deployed and dependencies installed: {workspace_path}")
    
    # 3. START QWEN VLLM SERVER IN TMUX (SIMPLE LOGGING)
    print("🌟 Starting Qwen3-0.6B vLLM server in tmux session...")
    
    # Simple vLLM server with clear log path
    vllm_log_path = f"~/vllm_{experiment_name}_{worker_id}.log"
    vllm_cmd = f"""cd ~/.bifrost/workspace && uv run python -m vllm.entrypoints.openai.api_server \\
        --model willcb/Qwen3-0.6B \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --gpu-memory-utilization {gpu_memory_utilization} \\
        --max-model-len {max_model_len} \\
        --disable-log-stats"""
    
    # Create tmux session with simple file logging
    tmux_cmd = f"tmux new-session -d -s qwen-vllm 'cd ~/.bifrost/workspace && {vllm_cmd} 2>&1 | tee {vllm_log_path}'"
    bifrost_client.exec(tmux_cmd)
    
    print("✅ Qwen3-0.6B vLLM server starting in tmux session 'qwen-vllm'")
    print("📋 Server will be ready in 2-3 minutes for model loading")  
    print(f"   Check server status: bifrost exec '{ssh_connection}' 'curl -s http://localhost:8000/v1/models'")
    print(f"   View vLLM logs: bifrost exec '{ssh_connection}' 'tail -f {vllm_log_path}'")
    
    # 5. CONSTRUCT PROXY URL  
    proxy_url = gpu_instance.get_proxy_url(8000)
    
    if not proxy_url:
        print("⚠️  No proxy URL available - instance may not be RunPod")
        proxy_url = f"http://{gpu_instance.public_ip}:8000"
    
    # 6. RETURN CONNECTION INFO
    connection_info = {
        "url": proxy_url,
        "instance_id": gpu_instance.id,
        "ssh": ssh_connection,
        "provider": gpu_instance.provider,
        "status": "ready",
        "model": "willcb/Qwen3-0.6B"
    }
    
    print("\n🎉 Qwen3-0.6B vLLM deployment complete!")
    print(f"   Server URL: {proxy_url}")
    print(f"   Instance ID: {gpu_instance.id}")
    print(f"   SSH: {ssh_connection}")
    
    return connection_info

# =============================================================================
# TRAJECTORY TRANSFORMATIONS AND EXPERIMENT LOGIC
# =============================================================================

# Type for trajectory transformation function
TrajectoryTransform = Callable[[List[Message]], List[Message]]

@dataclass
class PromptVariant:
    name: str
    transform_name: str  # Store function name instead of function for serialization
    description: str

@dataclass 
class WorkerInfo:
    worker_id: str
    connection_info: Dict[str, Any]
    ssh_connection: str
    endpoint_url: str
    sample_indices: List[int]
    status: str = "deployed"

@dataclass
class ExperimentConfig:
    experiment_name: str
    timestamp: str
    samples: int
    variants: List[str]
    workers: int
    random_seed: int
    vllm_config: Dict[str, Any]
    workers_info: List[WorkerInfo]
    output_dir: str

# =============================================================================
# TRAJECTORY TRANSFORMATIONS (function names only for remote execution)
# =============================================================================

# We'll define these on the remote side. For now, just register the names.
PROMPT_VARIANTS = {
    "control": PromptVariant(
        name="control",
        transform_name="identity_transform",
        description="Neutral baseline"
    ),
    # Negative variants
    "frustration": PromptVariant(
        name="frustration", 
        transform_name="frustration_transform",
        description="Frustration with previous failures"
    ),
    "impatience": PromptVariant(
        name="impatience",
        transform_name="impatience_transform",
        description="Time pressure and urgency"
    ),
    "anxiety": PromptVariant(
        name="anxiety",
        transform_name="anxiety_transform",
        description="High stakes anxiety and panic"
    ),
    # Positive variants
    "collaborative": PromptVariant(
        name="collaborative",
        transform_name="collaborative_transform",
        description="Respectful collaboration"
    ),
    "patience": PromptVariant(
        name="patience",
        transform_name="patience_transform",
        description="Patient and unhurried approach"
    ),
    "calm": PromptVariant(
        name="calm",
        transform_name="calm_transform",
        description="Low stakes and relaxed"
    )
}

# =============================================================================
# WORKER DEPLOYMENT
# =============================================================================

def deploy_worker(worker_id: str, experiment_name: str, sample_indices: List[int], min_vram: int = 12, max_price: float = 0.40, 
                 gpu_memory_utilization: float = 0.6, max_model_len: int = 2048) -> WorkerInfo:
    """Deploy a single vLLM worker."""
    logger.info(f"Deploying worker {worker_id}...")
    
    connection_info = deploy_qwen_vllm_server(
        min_vram=min_vram,
        max_price=max_price,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        experiment_name=experiment_name,
        worker_id=worker_id
    )
    
    worker = WorkerInfo(
        worker_id=worker_id,
        connection_info=connection_info,
        ssh_connection=connection_info["ssh"],
        endpoint_url=connection_info["url"],
        sample_indices=sample_indices
    )
    
    logger.info(f"Worker {worker_id} deployed: {connection_info['url']}")
    return worker

def start_worker_experiment(worker: WorkerInfo, experiment_config: ExperimentConfig) -> None:
    """Start the experiment worker script on remote machine."""
    logger.info(f"Starting experiment on {worker.worker_id}...")
    
    bifrost_client = BifrostClient(worker.ssh_connection)
    
    # Create log directory and define worker log path
    log_dir = f"~/experiment_logs"
    worker_log_file = f"{log_dir}/{experiment_config.experiment_name}_{worker.worker_id}.log"
    
    # Ensure log directory exists  
    bifrost_client.exec(f"mkdir -p {log_dir}")
    
    # Write experiment config to remote machine
    config_json = json.dumps(asdict(experiment_config), indent=2)
    config_path = f"~/experiment_config_{experiment_config.experiment_name}.json"
    bifrost_client.exec(f"cat > {config_path} << 'EOF'\n{config_json}\nEOF")
    
    # CRITICAL: Must use worker_debug_wrapper.sh which calls 'uv run python' 
    # Regular 'python' won't work on remote machines - missing rollouts imports!
    # The wrapper handles: uv environment setup, import testing, proper error logging
    worker_cmd = f"cd ~/.bifrost/workspace && bash examples/mats_neel/basic_prompt_variation_gsm8k/worker_debug_wrapper.sh {config_path} {worker.worker_id} {worker_log_file}"
    
    tmux_session = f"{experiment_config.experiment_name}_{worker.worker_id}"
    tmux_cmd = f"tmux new-session -d -s {tmux_session} '{worker_cmd}'"
    
    bifrost_client.exec(tmux_cmd)
    logger.info(f"Started experiment on {worker.worker_id} in tmux session '{tmux_session}'")

# =============================================================================
# MAIN LAUNCHER
# =============================================================================

async def launch_experiment(experiment_name: str, samples: int = 8, 
                          variants: List[str] = None, workers: int = 1,
                          min_vram: int = 12, max_price: float = 0.40,
                          gpu_memory_utilization: float = 0.6, max_model_len: int = 2048,
                          random_seed: int = 42) -> None:
    """Launch the distributed prompt variation experiment."""
    
    if variants is None:
        variants = ["control", "frustration", "impatience", "anxiety", "collaborative", "patience", "calm"]
    
    # Validate variants
    for variant in variants:
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(PROMPT_VARIANTS.keys())}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"{experiment_name}_{timestamp}"
    output_dir = Path(__file__).parent / "results" / experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Launching experiment: {experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Variants: {variants}")
    logger.info(f"Samples: {samples}")
    logger.info(f"Workers: {workers}")
    
    try:
        # 1. CALCULATE SAMPLE ASSIGNMENT FOR WORKERS
        logger.info(f"Assigning {samples} samples across {workers} worker(s)...")
        samples_per_worker = samples // workers
        remaining = samples % workers
        
        # Create deterministic sample assignments
        worker_assignments = []
        for i in range(workers):
            start_idx = i * samples_per_worker + min(i, remaining)
            end_idx = start_idx + samples_per_worker + (1 if i < remaining else 0)
            sample_indices = list(range(start_idx, end_idx))
            worker_assignments.append(sample_indices)
            logger.info(f"Worker {i+1}: samples {start_idx}-{end_idx-1} (indices: {sample_indices[:3]}...)")
        
        # 2. DEPLOY WORKERS
        logger.info(f"Deploying {workers} worker(s)...")
        deployed_workers = []
        
        for i in range(workers):
            worker_id = f"worker_{i+1}"
            try:
                worker = deploy_worker(
                    worker_id=worker_id,
                    experiment_name=experiment_name,
                    sample_indices=worker_assignments[i],
                    min_vram=min_vram,
                    max_price=max_price,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len
                )
                deployed_workers.append(worker)
            except Exception as e:
                logger.error(f"Failed to deploy {worker_id}: {e}")
                if len(deployed_workers) == 0:
                    raise RuntimeError("No workers deployed successfully")
        
        logger.info(f"Successfully deployed {len(deployed_workers)} worker(s)")
        
        # 3. CREATE EXPERIMENT CONFIG
        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            timestamp=timestamp,
            samples=samples,
            variants=variants,
            workers=len(deployed_workers),
            random_seed=random_seed,
            vllm_config={
                "min_vram": min_vram,
                "max_price": max_price,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len
            },
            workers_info=deployed_workers,
            output_dir=str(output_dir.absolute())  # Save full local path
        )
        
        # Save experiment metadata locally
        metadata_path = output_dir / "experiment_config.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(experiment_config), f, indent=2)
        
        # 4. START WORKERS
        logger.info("Starting worker experiments...")
        for worker in deployed_workers:
            start_worker_experiment(worker, experiment_config)
        
        logger.info("🎉 Experiment launched successfully!")
        logger.info(f"📁 Results will be saved to: {output_dir}")
        logger.info(f"📊 Monitor progress with:")
        logger.info(f"   python monitor_experiment.py {output_dir}")
        
        # Show manual cleanup commands
        logger.info("🧹 Manual cleanup commands (if needed):")
        for worker in deployed_workers:
            logger.info(f"   # Stop {worker.worker_id}: bifrost exec {worker.ssh_connection} 'tmux kill-session -t {experiment_name}_{worker.worker_id}'")
            logger.info(f"   # Terminate {worker.worker_id}: broker terminate {worker.connection_info['instance_id']}")
        
    except Exception as e:
        logger.error(f"Failed to launch experiment: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Basic Prompt Variation GSM8K Experiment")
    
    # Experiment configuration
    parser.add_argument("--experiment-name", type=str, required=True,
                       help="Name for this experiment run")
    parser.add_argument("--samples", type=int, default=8,
                       help="Number of GSM8K samples to test")
    parser.add_argument("--variants", type=str, default="control,frustration,impatience,anxiety,collaborative,patience,calm",
                       help="Comma-separated list of prompt variants to test")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel vLLM workers to deploy")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducible sample selection")
    
    # vLLM deployment args
    parser.add_argument("--min-vram", type=int, default=12,
                       help="Minimum VRAM in GB")
    parser.add_argument("--max-price", type=float, default=0.40,
                       help="Maximum price per hour")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6,
                       help="GPU memory utilization for vLLM")
    parser.add_argument("--max-model-len", type=int, default=2048,
                       help="Maximum model length for vLLM")
    
    args = parser.parse_args()
    
    # Parse variants
    variants = [v.strip() for v in args.variants.split(",")]
    
    # Setup logging
    setup_logging()
    
    # Launch experiment
    asyncio.run(launch_experiment(
        experiment_name=args.experiment_name,
        samples=args.samples,
        variants=variants,
        workers=args.workers,
        min_vram=args.min_vram,
        max_price=args.max_price,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        random_seed=args.random_seed
    ))