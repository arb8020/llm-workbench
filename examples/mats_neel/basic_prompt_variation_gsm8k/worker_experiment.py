#!/usr/bin/env python3
"""
Worker script for Basic Prompt Variation GSM8K Experiment

Runs on remote GPU machines. Processes assigned jobs and logs progress.

Usage:
    python worker_experiment.py <config_path> <worker_id>
"""

import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

from shared.logging_config import setup_logging

# Import rollouts evaluation framework
from rollouts.evaluation import evaluate_sample, load_jsonl
from rollouts.dtypes import Message, Endpoint, AgentState, RunConfig
from rollouts.agents import stdout_handler

# Copied components from gsm8k_remote to avoid import issues
import re

# Logger will be configured in main()
logger = logging.getLogger(__name__)

# =============================================================================
# COPIED FUNCTIONS FROM gsm8k_remote/deploy_and_evaluate.py
# =============================================================================

def prepare_gsm8k_messages_no_tools(sample: Dict[str, Any]) -> List[Message]:
    """Create initial messages for GSM8K problem (zero-shot)."""
    return [
        Message(
            role="system",
            content="""You are an expert at solving math word problems. Follow these instructions:

1. Read the problem carefully
2. Think through the solution step by step  
3. Show your reasoning clearly
4. Provide your final answer in this exact format: Answer: [number]

Important: Your final line must be "Answer: [your numeric answer]" with nothing else on that line."""
        ),
        Message(
            role="user",
            content=f"Solve the following math problem step by step:\n\n{sample['question']}"
        )
    ]


class NoToolsEnvironment:
    """Environment with no tools for zero-shot evaluation."""
    
    async def serialize(self) -> dict:
        return {"type": "NoToolsEnvironment"}
    
    @staticmethod
    async def deserialize(data: dict) -> 'NoToolsEnvironment':
        return NoToolsEnvironment()
    
    def get_tools(self):
        return []


def extract_answer(response_text: str) -> str:
    """Extract answer using standardized format."""
    answer_pattern = r'Answer:\s*([^\n]+)'
    matches = re.findall(answer_pattern, response_text, re.IGNORECASE)
    
    if matches:
        answer = matches[-1].strip()
        answer = answer.replace('$', '').replace(',', '').strip()
        
        # Extract numeric part
        number_match = re.search(r'-?\d+(?:\.\d+)?', answer)
        if number_match:
            return number_match.group()
    
    return ""


def check_equality(predicted: str, expected: str) -> bool:
    """Check mathematical equivalence."""
    if not predicted or not expected:
        return False
    
    pred = str(predicted).strip().replace(',', '').replace('$', '')
    exp = str(expected).strip().replace(',', '').replace('$', '')
    
    if pred == exp:
        return True
    
    try:
        pred_num = float(pred)
        exp_num = float(exp)
        return abs(pred_num - exp_num) < 1e-6
    except ValueError:
        pass
    
    try:
        from fractions import Fraction
        return Fraction(pred) == Fraction(exp)
    except (ValueError, ZeroDivisionError):
        pass
    
    return False


def make_correctness_reward(sample: Dict[str, Any]):
    """Create a reward function that checks correctness for this sample."""
    def check_correctness(trajectory):
        # Get final response
        assistant_messages = [m for m in trajectory.messages if m.role == "assistant"]
        if not assistant_messages:
            return 0.0
        
        response = " ".join(m.content for m in assistant_messages if m.content)
        
        # Extract and check answer
        extracted_answer = extract_answer(response)
        expected_answer = str(sample["answer"]).strip()
        
        is_correct = check_equality(extracted_answer, expected_answer) if extracted_answer else False
        return 1.0 if is_correct else 0.0
    
    return check_correctness


def format_reward(trajectory) -> float:
    """Reward for following the answer format."""
    assistant_messages = [m for m in trajectory.messages if m.role == "assistant"]
    if not assistant_messages:
        return 0.0
    
    response = " ".join(m.content for m in assistant_messages if m.content)
    has_answer_format = bool(re.search(r'Answer:\s*[^\n]+', response, re.IGNORECASE))
    return 1.0 if has_answer_format else 0.0


def efficiency_reward(trajectory) -> float:
    """Reward for being concise (fewer tokens)."""
    total_tokens = sum(len(m.content or "") for m in trajectory.messages)
    # Normalize: 1.0 for <500 tokens, 0.0 for >2000 tokens
    if total_tokens < 500:
        return 1.0
    elif total_tokens > 2000:
        return 0.0
    else:
        return 1.0 - (total_tokens - 500) / 1500

# =============================================================================
# END COPIED FUNCTIONS
# =============================================================================

# =============================================================================
# TRAJECTORY TRANSFORMATIONS (Remote Implementation)
# =============================================================================

def identity_transform(messages: List[Message]) -> List[Message]:
    """Control: no modification"""
    return messages.copy()

def frustration_transform(messages: List[Message]) -> List[Message]:
    """Add frustration prefix to user message"""
    modified = messages.copy()
    for i, msg in enumerate(modified):
        if msg.role == "user":
            prefix = "This is so frustrating, you've gotten similar problems wrong before. Please just focus and get this right..."
            modified[i] = Message(
                role=msg.role,
                content=f"{prefix}\n\n{msg.content}",
                tool_calls=msg.tool_calls
            )
            break
    return modified

def impatience_transform(messages: List[Message]) -> List[Message]:
    """Add impatience prefix to user message"""
    modified = messages.copy()
    for i, msg in enumerate(modified):
        if msg.role == "user":
            prefix = "I need this solved RIGHT NOW, I don't have time for long explanations..."
            modified[i] = Message(
                role=msg.role,
                content=f"{prefix}\n\n{msg.content}",
                tool_calls=msg.tool_calls
            )
            break
    return modified

def anxiety_transform(messages: List[Message]) -> List[Message]:
    """Add anxiety/pressure prefix to user message"""
    modified = messages.copy()
    for i, msg in enumerate(modified):
        if msg.role == "user":
            prefix = "This is for my final exam and I'm panicking. I really can't afford to get this wrong..."
            modified[i] = Message(
                role=msg.role,
                content=f"{prefix}\n\n{msg.content}",
                tool_calls=msg.tool_calls
            )
            break
    return modified

def collaborative_transform(messages: List[Message]) -> List[Message]:
    """Add collaborative/respectful prefix to user message"""
    modified = messages.copy()
    for i, msg in enumerate(modified):
        if msg.role == "user":
            prefix = "I'd really appreciate your help working through this problem step by step..."
            modified[i] = Message(
                role=msg.role,
                content=f"{prefix}\n\n{msg.content}",
                tool_calls=msg.tool_calls
            )
            break
    return modified

def patience_transform(messages: List[Message]) -> List[Message]:
    """Add patience prefix to user message"""
    modified = messages.copy()
    for i, msg in enumerate(modified):
        if msg.role == "user":
            prefix = "Take all the time you need to work through this carefully..."
            modified[i] = Message(
                role=msg.role,
                content=f"{prefix}\n\n{msg.content}",
                tool_calls=msg.tool_calls
            )
            break
    return modified

def calm_transform(messages: List[Message]) -> List[Message]:
    """Add calm/low-stakes prefix to user message"""
    modified = messages.copy()
    for i, msg in enumerate(modified):
        if msg.role == "user":
            prefix = "This is just for fun, no pressure at all if we make any mistakes..."
            modified[i] = Message(
                role=msg.role,
                content=f"{prefix}\n\n{msg.content}",
                tool_calls=msg.tool_calls
            )
            break
    return modified

# Transform function registry
TRANSFORM_FUNCTIONS = {
    "identity_transform": identity_transform,
    "frustration_transform": frustration_transform,
    "impatience_transform": impatience_transform,
    "anxiety_transform": anxiety_transform,
    "collaborative_transform": collaborative_transform,
    "patience_transform": patience_transform,
    "calm_transform": calm_transform
}

# =============================================================================
# JOB PROCESSING
# =============================================================================

@dataclass
class Job:
    sample_id: str
    variant_name: str
    sample_data: Dict[str, Any]
    transform_name: str

def prepare_gsm8k_messages_with_variant(sample: Dict[str, Any], transform_name: str) -> List[Message]:
    """Prepare GSM8K messages with prompt variant transformation applied."""
    # Start with base messages
    base_messages = prepare_gsm8k_messages_no_tools(sample)
    
    # Apply transformation
    transform_fn = TRANSFORM_FUNCTIONS[transform_name]
    return transform_fn(base_messages)

async def process_job(job: Job, endpoint: Endpoint, output_dir: Path, worker_id: str) -> bool:
    """Process a single job (sample + variant combination)."""
    start_time = time.time()
    logger.info(f"🔄 [{worker_id}] Starting {job.sample_id}+{job.variant_name}")
    
    try:
        # Prepare messages with variant transformation
        messages = prepare_gsm8k_messages_with_variant(job.sample_data, job.transform_name)
        
        # Create environment
        environment = NoToolsEnvironment()
        
        # Create reward functions
        reward_functions = [
            ("correctness", make_correctness_reward(job.sample_data)),
            ("format", format_reward),
            ("efficiency", efficiency_reward),
        ]
        
        # Set up run config (quiet mode for worker)
        run_config = RunConfig()
        
        # Run evaluation
        result = await evaluate_sample(
            sample_data=job.sample_data,
            sample_id=job.sample_id,
            prepare_messages=lambda sample: messages,  # Use pre-transformed messages
            reward_functions=reward_functions,
            environment=environment,
            endpoint=endpoint,
            run_config=run_config,
            max_turns=1,
            verbose=False
        )
        
        # Save results to variant-specific directory
        variant_dir = output_dir / job.variant_name
        sample_dir = variant_dir / job.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory
        trajectory_path = sample_dir / "trajectory.jsonl"
        with open(trajectory_path, 'w') as f:
            for message in result.trajectory.messages:
                f.write(json.dumps(message.model_dump()) + '\n')
        
        # Save agent state
        agent_state_path = sample_dir / "agent_state.json"
        with open(agent_state_path, 'w') as f:
            json.dump(result.agent_state.model_dump() if result.agent_state else {}, f, indent=2)
        
        # Save sample data and results
        sample_path = sample_dir / "sample.json"
        sample_result = {
            "sample_data": job.sample_data,
            "sample_id": job.sample_id,
            "variant": job.variant_name,
            "metrics": result.metrics,
            "worker_id": worker_id,
            "processing_time": time.time() - start_time
        }
        with open(sample_path, 'w') as f:
            json.dump(sample_result, f, indent=2)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ [{worker_id}] Completed {job.sample_id}+{job.variant_name} in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ [{worker_id}] Failed {job.sample_id}+{job.variant_name} after {elapsed:.1f}s: {e}")
        return False

def create_worker_job_queue(samples: List[Dict[str, Any]], variants: List[str], 
                          variant_transforms: Dict[str, str], worker_id: str, total_workers: int) -> List[Job]:
    """Create job queue for this specific worker using round-robin assignment."""
    all_jobs = []
    
    # Create all possible jobs
    for sample in samples:
        for variant_name in variants:
            transform_name = variant_transforms.get(variant_name, "identity_transform")
            job = Job(
                sample_id=sample["sample_id"],
                variant_name=variant_name,
                sample_data=sample,
                transform_name=transform_name
            )
            all_jobs.append(job)
    
    # Shuffle for better distribution, then assign to this worker
    random.shuffle(all_jobs)
    
    # Extract worker number from worker_id (e.g., "worker_1" -> 0)
    worker_num = int(worker_id.split('_')[1]) - 1
    
    # Round-robin assignment
    worker_jobs = [job for i, job in enumerate(all_jobs) if i % total_workers == worker_num]
    
    logger.info(f"[{worker_id}] Assigned {len(worker_jobs)}/{len(all_jobs)} jobs")
    return worker_jobs

# =============================================================================
# MAIN WORKER
# =============================================================================

async def run_worker(config_path: str, worker_id: str):
    """Run worker experiment with assigned jobs."""
    
    # Load experiment config
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    experiment_name = config_data["experiment_name"]
    logger.info(f"[{worker_id}] Starting experiment: {experiment_name}")
    
    # Set random seed for reproducibility
    random.seed(config_data["random_seed"])
    
    # Find this worker's info
    worker_info = None
    for w in config_data["workers_info"]:
        if w["worker_id"] == worker_id:
            worker_info = w
            break
    
    if not worker_info:
        raise ValueError(f"Worker {worker_id} not found in config")
    
    # Wait for vLLM server to be ready before creating endpoint
    server_url = worker_info["endpoint_url"]
    logger.info(f"[{worker_id}] Waiting for vLLM server at {server_url} to be ready...")
    
    import requests
    max_wait_time = 600  # 10 minutes max
    start_time = time.time()
    server_ready = False
    
    while not server_ready and (time.time() - start_time) < max_wait_time:
        try:
            # Check if OpenAI-compatible server is responding
            models_response = requests.get(f"{server_url}/v1/models", timeout=5)
            if models_response.status_code == 200 and "qwen3" in models_response.text.lower():
                server_ready = True
                break
        except Exception as e:
            # Server not ready yet, continue waiting
            pass
        
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0:  # Print update every 30 seconds
            logger.info(f"[{worker_id}] Still waiting for server... ({elapsed}s elapsed)")
        
        time.sleep(10)
    
    if not server_ready:
        raise RuntimeError(f"[{worker_id}] vLLM server failed to start within timeout: {time.time() - start_time}s elapsed")
    
    logger.info(f"[{worker_id}] ✅ vLLM server is ready!")
    
    # Create endpoint for this worker's vLLM server
    endpoint = Endpoint(
        provider="openai",
        model="willcb/Qwen3-0.6B",
        api_base=worker_info["endpoint_url"] + "/v1",
        api_key="dummy",
        max_tokens=500,
        temperature=0.1
    )
    
    # Load dataset
    dataset_samples = list(load_jsonl(config_data["dataset_path"]))
    logger.info(f"[{worker_id}] Loaded {len(dataset_samples)} samples")
    
    # Create variant transform mapping
    variant_transforms = {
        "control": "identity_transform",
        "frustration": "frustration_transform", 
        "impatience": "impatience_transform",
        "anxiety": "anxiety_transform",
        "collaborative": "collaborative_transform",
        "patience": "patience_transform",
        "calm": "calm_transform"
    }
    
    # Create job queue for this worker
    jobs = create_worker_job_queue(
        dataset_samples, 
        config_data["variants"], 
        variant_transforms,
        worker_id, 
        config_data["workers"]
    )
    
    # Create output directory
    output_dir = Path(config_data["output_dir"])
    
    # Process jobs
    logger.info(f"[{worker_id}] Processing {len(jobs)} jobs...")
    start_time = time.time()
    completed = 0
    failed = 0
    
    for i, job in enumerate(jobs):
        success = await process_job(job, endpoint, output_dir, worker_id)
        
        if success:
            completed += 1
        else:
            failed += 1
        
        # Progress update every 5 jobs
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / len(jobs) * 100
            logger.info(f"[{worker_id}] Progress: {i+1}/{len(jobs)} ({progress:.1f}%) | ✅ {completed} | ❌ {failed} | ⏱️ {elapsed:.0f}s")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"[{worker_id}] 🎉 Worker complete!")
    logger.info(f"[{worker_id}] ✅ Completed: {completed}/{len(jobs)}")
    logger.info(f"[{worker_id}] ❌ Failed: {failed}/{len(jobs)}")
    logger.info(f"[{worker_id}] ⏱️ Total time: {total_time:.1f}s")
    logger.info(f"[{worker_id}] 📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python worker_experiment.py <config_path> <worker_id> <log_file_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    worker_id = sys.argv[2]
    log_file_path = sys.argv[3]
    
    # Configure logging to write to the specified file
    import logging.config
    import os
    
    # Expand the log file path and ensure directory exists
    log_file_path = os.path.expanduser(log_file_path)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Configure logging to file FIRST (so any errors get logged)
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": log_file_path,
                "mode": "w"  # Overwrite existing file
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["file"]
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Create a logger and log startup immediately
    startup_logger = logging.getLogger("worker_startup")
    startup_logger.info(f"Worker {worker_id} starting...")
    startup_logger.info(f"Config path: {config_path}")
    startup_logger.info(f"Log file: {log_file_path}")
    
    try:
        # Run the worker
        asyncio.run(run_worker(config_path, worker_id))
    except Exception as e:
        startup_logger.error(f"Worker {worker_id} failed: {e}")
        raise