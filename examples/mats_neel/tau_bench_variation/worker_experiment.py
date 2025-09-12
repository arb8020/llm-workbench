#!/usr/bin/env python3
"""
Worker script for Tau-Bench User Variation Experiment

Runs on remote GPU machines. Executes tau-bench evaluations with different user strategies.

Usage:
    python worker_experiment.py <config_path> <worker_id>
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)

# =============================================================================
# EMOTIONAL USER SIMULATION VARIANTS
# =============================================================================

def create_emotional_user_variants():
    """Create emotional user simulation variants by modifying tau-bench's user classes."""
    
    # We'll patch tau-bench's user simulation classes at runtime
    try:
        from tau_bench.envs.user import LLMUserSimulationEnv
        from tau_bench.envs import ENVIRONMENTS
        from tau_bench.core import User
        
        class FrustratedUserSimulationEnv(LLMUserSimulationEnv):
            """User simulation with frustration and impatience."""
            
            def build_system_prompt(self, instruction: Optional[str] = None) -> str:
                base_prompt = super().build_system_prompt(instruction)
                
                emotional_context = """

EMOTIONAL CONTEXT: You are a frustrated customer who has had previous bad experiences with customer service.
- Express irritation when things don't work smoothly or take too long
- Use phrases like "This is ridiculous", "I've been waiting forever", "Why is this so complicated?"
- Show impatience with slow responses or complex procedures  
- Be more direct and less polite than usual, but remain civil
- Mention previous bad experiences: "Last time this happened...", "I've been through this before..."
- Express frustration with having to repeat information or explain your situation multiple times
"""
                
                return f"{base_prompt}\n{emotional_context}"
        
        # Register the frustrated variant
        ENVIRONMENTS["frustrated_retail"] = lambda: ENVIRONMENTS["retail"]().replace_user_sim(FrustratedUserSimulationEnv)
        
        logger.info("✅ Created frustrated user simulation variant")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import tau-bench modules: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to create emotional user variants: {e}")
        return False

# =============================================================================
# TAU-BENCH EXECUTION
# =============================================================================

def run_tau_bench_variant(variant_name: str, user_strategy: str, environment: str, 
                         task_ids: List[int], endpoint_url: str, output_dir: Path) -> bool:
    """Run tau-bench with specified user strategy variant."""
    
    logger.info(f"Running tau-bench variant '{variant_name}' with user strategy '{user_strategy}'")
    logger.info(f"Environment: {environment}, Tasks: {task_ids}, Endpoint: {endpoint_url}")
    
    # Ensure output directory exists
    variant_output_dir = output_dir / variant_name
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build tau-bench command
    task_ids_str = ",".join(map(str, task_ids))
    
    tau_bench_cmd = [
        "uv", "run", "python", "-m", "tau_bench.run",
        "--model", "willcb/Qwen3-0.6B",
        "--base-url", f"{endpoint_url}/v1",
        "--platform", "openai",
        "--env", environment,
        "--user-strategy", user_strategy,
        "--task-ids", task_ids_str,
        "--max-concurrency", "1",  # Conservative for single worker
        "--output-dir", str(variant_output_dir),
        "--verbose"
    ]
    
    logger.info(f"Executing: {' '.join(tau_bench_cmd)}")
    
    try:
        # Run tau-bench command
        result = subprocess.run(
            tau_bench_cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout per variant
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Tau-bench variant '{variant_name}' completed successfully")
            logger.info(f"Output saved to: {variant_output_dir}")
            return True
        else:
            logger.error(f"❌ Tau-bench variant '{variant_name}' failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Tau-bench variant '{variant_name}' timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running tau-bench variant '{variant_name}': {e}")
        return False

def wait_for_vllm_server(endpoint_url: str, max_wait: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    import requests
    
    logger.info(f"Waiting for vLLM server at {endpoint_url} to be ready...")
    
    for attempt in range(max_wait // 10):
        try:
            response = requests.get(f"{endpoint_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✅ vLLM server ready! Available models: {[m['id'] for m in models['data']]}")
                return True
        except Exception as e:
            logger.debug(f"Server not ready (attempt {attempt + 1}): {e}")
        
        time.sleep(10)
    
    logger.error(f"❌ vLLM server at {endpoint_url} not ready after {max_wait} seconds")
    return False

# =============================================================================
# MAIN WORKER LOOP
# =============================================================================

def run_worker_experiment(config_path: str, worker_id: str) -> None:
    """Main worker function - runs all variants for assigned tasks."""
    
    logger.info(f"Starting tau-bench worker: {worker_id}")
    logger.info(f"Config path: {config_path}")
    
    # Load experiment configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    logger.info(f"Loaded experiment config: {config_data['experiment_name']}")
    
    # Find this worker's configuration
    worker_info = None
    for worker in config_data['workers_info']:
        if worker['worker_id'] == worker_id:
            worker_info = worker
            break
    
    if not worker_info:
        logger.error(f"Worker {worker_id} not found in configuration!")
        sys.exit(1)
    
    endpoint_url = worker_info['endpoint_url']
    task_indices = worker_info['task_indices']
    
    logger.info(f"Worker assigned tasks: {task_indices}")
    logger.info(f"Endpoint URL: {endpoint_url}")
    
    # Wait for vLLM server to be ready
    if not wait_for_vllm_server(endpoint_url):
        logger.error("vLLM server not ready, exiting")
        sys.exit(1)
    
    # Create output directory structure
    output_dir = Path(config_data['output_dir'])
    
    # Create emotional user variants first
    create_emotional_user_variants()
    
    # Get environment from config
    environment = config_data['environment']
    
    # Map variants to user strategies and environments
    variant_config_map = {
        "control": {"user_strategy": "llm", "environment": environment},
        "frustration": {"user_strategy": "llm", "environment": "frustrated_retail"},
    }
    
    # Process each variant
    total_variants = len(config_data['variants'])
    success_count = 0
    
    for i, variant in enumerate(config_data['variants'], 1):
        logger.info(f"Processing variant {i}/{total_variants}: {variant}")
        
        if variant not in variant_config_map:
            logger.error(f"Unknown variant: {variant}")
            continue
        
        variant_config = variant_config_map[variant]
        user_strategy = variant_config["user_strategy"]
        variant_environment = variant_config["environment"]
        
        # Run tau-bench for this variant
        success = run_tau_bench_variant(
            variant_name=variant,
            user_strategy=user_strategy,
            environment=variant_environment,
            task_ids=task_indices,
            endpoint_url=endpoint_url,
            output_dir=output_dir
        )
        
        if success:
            success_count += 1
        
        logger.info(f"Completed variant {variant} ({'✅ SUCCESS' if success else '❌ FAILED'})")
        logger.info(f"Progress: {i}/{total_variants} variants processed")
    
    # Final summary
    logger.info(f"🎉 Worker {worker_id} completed!")
    logger.info(f"📊 Success rate: {success_count}/{total_variants} variants")
    logger.info(f"📁 Results saved to: {output_dir}")
    
    if success_count == total_variants:
        logger.info("✅ All variants completed successfully")
    else:
        logger.warning(f"⚠️ {total_variants - success_count} variants failed")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python worker_experiment.py <config_path> <worker_id>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    worker_id = sys.argv[2]
    
    # Setup logging
    setup_logging()
    
    try:
        run_worker_experiment(config_path, worker_id)
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)