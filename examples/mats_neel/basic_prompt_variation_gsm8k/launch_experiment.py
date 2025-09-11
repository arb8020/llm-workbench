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

# Reuse components from gsm8k_remote
from examples.gsm8k_remote.deploy_and_evaluate import (
    deploy_qwen_vllm_server,
    load_gsm8k_dataset
)

logger = logging.getLogger(__name__)

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
    dataset_path: str
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

def deploy_worker(worker_id: str, min_vram: int = 12, max_price: float = 0.40, 
                 gpu_memory_utilization: float = 0.6, max_model_len: int = 2048) -> WorkerInfo:
    """Deploy a single vLLM worker."""
    logger.info(f"Deploying worker {worker_id}...")
    
    connection_info = deploy_qwen_vllm_server(
        min_vram=min_vram,
        max_price=max_price,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    
    worker = WorkerInfo(
        worker_id=worker_id,
        connection_info=connection_info,
        ssh_connection=connection_info["ssh"],
        endpoint_url=connection_info["url"]
    )
    
    logger.info(f"Worker {worker_id} deployed: {connection_info['url']}")
    return worker

def start_worker_experiment(worker: WorkerInfo, experiment_config: ExperimentConfig) -> None:
    """Start the experiment worker script on remote machine."""
    logger.info(f"Starting experiment on {worker.worker_id}...")
    
    bifrost_client = BifrostClient(worker.ssh_connection)
    
    # Create log directory
    log_dir = f"~/experiment_logs"
    bifrost_client.exec(f"mkdir -p {log_dir}")
    
    # Write experiment config to remote machine
    config_json = json.dumps(asdict(experiment_config), indent=2)
    config_path = f"~/experiment_config_{experiment_config.experiment_name}.json"
    bifrost_client.exec(f"cat > {config_path} << 'EOF'\n{config_json}\nEOF")
    
    # Start worker script in tmux
    log_file = f"{log_dir}/{experiment_config.experiment_name}_{worker.worker_id}.log"
    worker_script = f"cd ~/.bifrost/workspace && python examples/mats_neel/basic_prompt_variation_gsm8k/worker_experiment.py {config_path} {worker.worker_id}"
    
    tmux_session = f"{experiment_config.experiment_name}_{worker.worker_id}"
    tmux_cmd = f"tmux new-session -d -s {tmux_session} '{worker_script} 2>&1 | tee {log_file}'"
    
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
        # 1. PREPARE DATASET
        logger.info("Preparing GSM8K dataset...")
        assets_dir = Path(__file__).parent / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = assets_dir / f"gsm8k_samples_{samples}.jsonl"
        
        load_gsm8k_dataset(dataset_path, samples)
        
        # 2. DEPLOY WORKERS
        logger.info(f"Deploying {workers} worker(s)...")
        deployed_workers = []
        
        for i in range(workers):
            worker_id = f"worker_{i+1}"
            try:
                worker = deploy_worker(
                    worker_id=worker_id,
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
            dataset_path=str(dataset_path.absolute()),
            output_dir=str(output_dir.absolute())
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