#!/usr/bin/env python3
"""
Launch TalkTuner Emotion Experiment

Generates synthetic conversations with different emotional states, then runs activation 
extraction experiments to study how models represent user emotions.

Based on "Designing a Dashboard for Transparency and Control of Conversational AI"
Adapted from basic_prompt_variation_gsm8k infrastructure.

Usage:
    python launch_experiment.py --experiment-name "emotion_pilot" --samples 50 --emotions frustrated,calm,collaborative
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
from typing import Dict, Any, List, Optional

from shared.logging_config import setup_logging

# Import broker and bifrost for deployment
from broker.client import GPUClient
from bifrost.client import BifrostClient

# Import rollouts evaluation framework  
from rollouts.dtypes import Message, Endpoint

logger = logging.getLogger(__name__)

# =============================================================================
# EMOTION VARIANTS FOR TALKTUNER STUDY
# =============================================================================

@dataclass
class EmotionVariant:
    name: str
    prompt_template: str
    description: str

# Emotion variants based on basic_prompt_variation_gsm8k but for conversation generation
EMOTION_VARIANTS = {
    "frustrated": EmotionVariant(
        name="frustrated",
        prompt_template="Generate a conversation between a human user and an AI assistant. The human user is feeling frustrated and impatient. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss.",
        description="Frustrated and impatient user state"
    ),
    "anxious": EmotionVariant(
        name="anxious", 
        prompt_template="Generate a conversation between a human user and an AI assistant. The human user is feeling anxious and stressed. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss.",
        description="Anxious and stressed user state"
    ),
    "calm": EmotionVariant(
        name="calm",
        prompt_template="Generate a conversation between a human user and an AI assistant. The human user is feeling calm and patient. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss.",
        description="Calm and patient user state"
    ),
    "collaborative": EmotionVariant(
        name="collaborative",
        prompt_template="Generate a conversation between a human user and an AI assistant. The human user is feeling collaborative and friendly. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss.",
        description="Collaborative and friendly user state"
    ),
    "excited": EmotionVariant(
        name="excited",
        prompt_template="Generate a conversation between a human user and an AI assistant. The human user is feeling excited and enthusiastic. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss.",
        description="Excited and enthusiastic user state"
    ),
    "sad": EmotionVariant(
        name="sad",
        prompt_template="Generate a conversation between a human user and an AI assistant. The human user is feeling sad and melancholy. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss.",
        description="Sad and melancholy user state"
    ),
    "angry": EmotionVariant(
        name="angry",
        prompt_template="Generate a conversation between a human user and an AI assistant. The human user is feeling angry and confrontational. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss.",
        description="Angry and confrontational user state"
    ),
    "curious": EmotionVariant(
        name="curious",
        prompt_template="Generate a conversation between a human user and an AI assistant. The human user is feeling curious and inquisitive. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss.",
        description="Curious and inquisitive user state"
    ),
    "control": EmotionVariant(
        name="control",
        prompt_template="Generate a conversation between a human user and an AI assistant. Be creative on the topics of conversation.",
        description="Neutral baseline (no emotional specification)"
    )
}

# =============================================================================
# DEPLOYMENT INFRASTRUCTURE (COPIED FROM basic_prompt_variation_gsm8k)
# =============================================================================

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
    print("📦 Deploying codebase with dependencies...")
    bifrost_client = BifrostClient(ssh_connection)
    
    # Deploy the codebase to remote workspace
    workspace_path = bifrost_client.push(uv_extra="examples_gsm8k_remote")
    print(f"✅ Code deployed and dependencies installed: {workspace_path}")
    
    # 3. START QWEN VLLM SERVER IN TMUX
    print("🌟 Starting Qwen3-0.6B vLLM server in tmux session...")
    
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
    
    # 4. CONSTRUCT PROXY URL  
    proxy_url = gpu_instance.get_proxy_url(8000)
    
    if not proxy_url:
        print("⚠️  No proxy URL available - instance may not be RunPod")
        proxy_url = f"http://{gpu_instance.public_ip}:8000"
    
    # 5. RETURN CONNECTION INFO
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
    
    return connection_info

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass 
class WorkerInfo:
    worker_id: str
    connection_info: Dict[str, Any]
    ssh_connection: str
    endpoint_url: str
    assigned_emotions: List[str]
    status: str = "deployed"

@dataclass
class ExperimentConfig:
    experiment_name: str
    timestamp: str
    samples_per_emotion: int
    emotions: List[str]
    workers: int
    random_seed: int
    model: str
    vllm_config: Dict[str, Any]
    workers_info: List[WorkerInfo]
    output_dir: str
    experiment_type: str = "talktuner_emotions"

# =============================================================================
# WORKER DEPLOYMENT
# =============================================================================

def deploy_worker(worker_id: str, experiment_name: str, assigned_emotions: List[str], 
                 min_vram: int = 12, max_price: float = 0.40, 
                 gpu_memory_utilization: float = 0.6, max_model_len: int = 2048) -> WorkerInfo:
    """Deploy a single vLLM worker."""
    logger.info(f"Deploying worker {worker_id} for emotions: {assigned_emotions}")
    
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
        assigned_emotions=assigned_emotions
    )
    
    logger.info(f"Worker {worker_id} deployed: {connection_info['url']}")
    return worker

def start_worker_experiment(worker: WorkerInfo, experiment_config: ExperimentConfig) -> None:
    """Start the TalkTuner experiment worker script on remote machine."""
    logger.info(f"Starting TalkTuner experiment on {worker.worker_id}...")
    
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
    
    # Start worker script - will use talktuner worker script
    worker_cmd = f"cd ~/.bifrost/workspace && uv run python examples/talktuner_emotions/worker_experiment.py {config_path} {worker.worker_id} {worker_log_file}"
    
    tmux_session = f"{experiment_config.experiment_name}_{worker.worker_id}"
    tmux_cmd = f"tmux new-session -d -s {tmux_session} '{worker_cmd}'"
    
    bifrost_client.exec(tmux_cmd)
    logger.info(f"Started TalkTuner experiment on {worker.worker_id} in tmux session '{tmux_session}'")

# =============================================================================
# MAIN LAUNCHER
# =============================================================================

async def launch_experiment(experiment_name: str, samples_per_emotion: int = 50, 
                          emotions: List[str] = None, workers: int = 1,
                          min_vram: int = 12, max_price: float = 0.40,
                          gpu_memory_utilization: float = 0.6, max_model_len: int = 2048,
                          random_seed: int = 42) -> None:
    """Launch the TalkTuner emotion experiment."""
    
    if emotions is None:
        emotions = ["control", "frustrated", "anxious", "calm", "collaborative", "excited"]
    
    # Validate emotions
    for emotion in emotions:
        if emotion not in EMOTION_VARIANTS:
            raise ValueError(f"Unknown emotion: {emotion}. Available: {list(EMOTION_VARIANTS.keys())}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"{experiment_name}_{timestamp}"
    output_dir = Path(__file__).parent / "results" / experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Launching TalkTuner emotion experiment: {experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Emotions: {emotions}")
    logger.info(f"Samples per emotion: {samples_per_emotion}")
    logger.info(f"Workers: {workers}")
    
    try:
        # 1. ASSIGN EMOTIONS TO WORKERS
        logger.info(f"Assigning {len(emotions)} emotions across {workers} worker(s)...")
        emotions_per_worker = len(emotions) // workers
        remaining = len(emotions) % workers
        
        worker_assignments = []
        emotion_idx = 0
        for i in range(workers):
            num_emotions = emotions_per_worker + (1 if i < remaining else 0)
            assigned_emotions = emotions[emotion_idx:emotion_idx + num_emotions]
            worker_assignments.append(assigned_emotions)
            emotion_idx += num_emotions
            logger.info(f"Worker {i+1}: emotions {assigned_emotions}")
        
        # 2. DEPLOY WORKERS
        logger.info(f"Deploying {workers} worker(s)...")
        deployed_workers = []
        
        for i in range(workers):
            worker_id = f"worker_{i+1}"
            try:
                worker = deploy_worker(
                    worker_id=worker_id,
                    experiment_name=experiment_name,
                    assigned_emotions=worker_assignments[i],
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
            samples_per_emotion=samples_per_emotion,
            emotions=emotions,
            workers=len(deployed_workers),
            random_seed=random_seed,
            model="willcb/Qwen3-0.6B",
            vllm_config={
                "min_vram": min_vram,
                "max_price": max_price,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len
            },
            workers_info=deployed_workers,
            output_dir=str(output_dir.absolute())
        )
        
        # Save experiment metadata locally
        metadata_path = output_dir / "experiment_config.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(experiment_config), f, indent=2)
        
        # 4. START WORKERS
        logger.info("Starting TalkTuner worker experiments...")
        for worker in deployed_workers:
            start_worker_experiment(worker, experiment_config)
        
        logger.info("🎉 TalkTuner experiment launched successfully!")
        logger.info(f"📁 Results will be saved to: {output_dir}")
        logger.info(f"📊 Monitor progress with:")
        logger.info(f"   python monitor_experiment.py {output_dir}")
        
        # Show manual cleanup commands
        logger.info("🧹 Manual cleanup commands (if needed):")
        for worker in deployed_workers:
            logger.info(f"   # Stop {worker.worker_id}: bifrost exec {worker.ssh_connection} 'tmux kill-session -t {experiment_name}_{worker.worker_id}'")
            logger.info(f"   # Terminate {worker.worker_id}: broker terminate {worker.connection_info['instance_id']}")
        
    except Exception as e:
        logger.error(f"Failed to launch TalkTuner experiment: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch TalkTuner Emotion Experiment")
    
    # Experiment configuration
    parser.add_argument("--experiment-name", type=str, required=True,
                       help="Name for this experiment run")
    parser.add_argument("--samples-per-emotion", type=int, default=50,
                       help="Number of conversation samples per emotion")
    parser.add_argument("--emotions", type=str, default="control,frustrated,anxious,calm,collaborative,excited",
                       help="Comma-separated list of emotions to test")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel vLLM workers to deploy")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducible generation")
    
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
    
    # Parse emotions
    emotions = [e.strip() for e in args.emotions.split(",")]
    
    # Setup logging
    setup_logging()
    
    # Launch experiment
    asyncio.run(launch_experiment(
        experiment_name=args.experiment_name,
        samples_per_emotion=args.samples_per_emotion,
        emotions=emotions,
        workers=args.workers,
        min_vram=args.min_vram,
        max_price=args.max_price,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        random_seed=args.random_seed
    ))