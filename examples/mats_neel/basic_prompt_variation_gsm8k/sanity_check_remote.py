#!/usr/bin/env python3
"""
Remote Instance Sanity Check

Tests the complete worker pipeline on actual remote GPU instances (like real deployment).
This provisions a single worker, runs tests, and cleans up.

Usage:
    python sanity_check_remote.py --remote  # Test on actual GPU instance
    python sanity_check_remote.py          # Test locally (same as sanity_check_full.py)
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Import deployment logic
from launch_experiment import deploy_worker, PROMPT_VARIANTS

# Import worker logic  
from worker_experiment import Job

def load_environment():
    """Load API keys from .env file."""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
    else:
        print("⚠️  .env file not found, using system environment variables")

def create_test_gsm8k_sample() -> dict:
    """Create a simple GSM8K-style test sample."""
    return {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "18",  # 16 - 3 - 4 = 9 eggs left, 9 * $2 = $18
        "sample_id": "sanity_remote_001"
    }

async def test_local_pipeline():
    """Test locally using OpenAI API (same as sanity_check_full.py)."""
    print("🏠 Testing LOCAL pipeline with OpenAI API...")
    
    # Import local testing
    from sanity_check_full import test_full_pipeline
    return await test_full_pipeline()

async def test_remote_pipeline():
    """Test on actual remote GPU instance (full deployment simulation)."""
    print("🚀 Testing REMOTE pipeline with actual GPU instance...")
    print("⚠️  This will provision real GPU resources and incur costs!")
    
    load_environment()
    
    # Create test sample
    sample_data = create_test_gsm8k_sample()
    
    print("1️⃣ Provisioning remote GPU worker...")
    try:
        # Deploy a single worker (same as real deployment)
        worker = deploy_worker(
            worker_id="sanity_remote",
            experiment_name="sanity_check",
            sample_indices=[0],  # Just one sample
            min_vram=12,
            max_price=0.40,
            gpu_memory_utilization=0.6,
            max_model_len=2048
        )
        print(f"✅ Worker deployed: {worker.endpoint_url}")
    except Exception as e:
        print(f"❌ Failed to deploy worker: {e}")
        return 1
    
    # Import bifrost for remote execution
    from bifrost.client import BifrostClient
    
    try:
        print("2️⃣ Setting up remote worker environment...")
        bifrost_client = BifrostClient(worker.ssh_connection)
        
        # Test variants
        variants_to_test = ["control", "frustration"]
        
        for variant_name in variants_to_test:
            print(f"3️⃣ Testing remote variant: {variant_name}")
            
            # Create job
            variant_info = PROMPT_VARIANTS[variant_name]
            job = Job(
                sample_id=f"remote_{variant_name}",
                sample_data=sample_data,
                variant_name=variant_name,
                transform_name=variant_info.transform_name
            )
            
            # Create job JSON for remote worker
            job_json = {
                "sample_id": job.sample_id,
                "sample_data": job.sample_data,
                "variant_name": job.variant_name,
                "transform_name": job.transform_name
            }
            
            # Write job to remote machine
            job_path = f"~/sanity_job_{variant_name}.json"
            job_content = json.dumps(job_json, indent=2)
            bifrost_client.exec(f"cat > {job_path} << 'EOF'\n{job_content}\nEOF")
            
            # Run the job on remote machine
            print(f"   🤖 Executing job on remote GPU...")
            remote_cmd = f"cd ~/.bifrost/workspace && timeout 120 python -c \"\nimport json\nimport asyncio\nfrom pathlib import Path\nfrom examples.mats_neel.basic_prompt_variation_gsm8k.worker_experiment import process_job, Job\nfrom rollouts.dtypes import Endpoint\nimport os\n\n# Load job\nwith open('{job_path}') as f:\n    job_data = json.load(f)\n\njob = Job(**job_data)\n\n# Create endpoint (uses deployed vLLM server)\nendpoint = Endpoint(\n    provider='vllm',\n    model='willcb/Qwen3-0.6B',\n    api_base='{worker.endpoint_url}',\n    api_key='dummy'\n)\n\n# Process job\nresult = asyncio.run(process_job(job, endpoint, Path('/tmp/sanity_output'), 'remote_test'))\nprint(f'SUCCESS: Metrics={{result.metrics}}')\n\""
            
            # Execute with timeout
            start_time = time.time()
            try:
                output = bifrost_client.exec(remote_cmd, timeout=180)  # 3 minute timeout
                elapsed = time.time() - start_time
                
                if "SUCCESS:" in output:
                    print(f"   ✅ Remote execution successful in {elapsed:.1f}s")
                    print(f"   📊 Output: {output.split('SUCCESS:')[-1].strip()}")
                else:
                    print(f"   ❌ Remote execution failed: {output}")
                    return 1
                    
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"   ❌ Remote execution error after {elapsed:.1f}s: {e}")
                return 1
        
        print("🎉 REMOTE SANITY CHECK PASSED!")
        print("🚀 Remote pipeline is ready for full deployment")
        return 0
        
    finally:
        print("4️⃣ Cleaning up remote resources...")
        try:
            # Terminate the GPU instance
            from broker.client import GPUClient
            gpu_client = GPUClient()
            gpu_client.terminate(worker.connection_info["instance_id"])
            print("✅ Remote GPU instance terminated")
        except Exception as e:
            print(f"⚠️  Failed to cleanup GPU instance: {e}")
            print(f"   Manual cleanup: broker terminate {worker.connection_info['instance_id']}")

async def main():
    """Run sanity check - local or remote based on flag."""
    parser = argparse.ArgumentParser(description="Sanity check with optional remote testing")
    parser.add_argument("--remote", action="store_true", 
                       help="Test on actual remote GPU instance (incurs costs)")
    parser.add_argument("--local", action="store_true", default=True,
                       help="Test locally using OpenAI API (default)")
    
    args = parser.parse_args()
    
    if args.remote:
        return await test_remote_pipeline()
    else:
        return await test_local_pipeline()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)