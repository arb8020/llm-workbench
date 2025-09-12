#!/usr/bin/env python3
"""
Full End-to-End Sanity Check with Real API

Tests the complete worker pipeline with actual API calls using a single GSM8K sample.
This catches integration issues before expensive multi-GPU deployment.

Usage:
    python sanity_check_full.py
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Import worker logic
from worker_experiment import process_job, Job
from launch_experiment import PROMPT_VARIANTS

# Import rollouts for real evaluation
from rollouts.evaluation import evaluate_sample, EvalSample
from rollouts.dtypes import Message, Endpoint

def load_environment():
    """Load API keys from .env file."""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
    else:
        print("⚠️  .env file not found, using system environment variables")

def create_test_endpoint() -> Endpoint:
    """Create a test endpoint using OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    return Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=api_key,
        api_base="https://api.openai.com/v1"
    )

def create_test_gsm8k_sample() -> dict:
    """Create a simple GSM8K-style test sample."""
    return {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "18",  # 16 - 3 - 4 = 9 eggs left, 9 * $2 = $18
        "sample_id": "sanity_check_001"
    }

async def test_full_pipeline():
    """Test the complete pipeline with real API calls."""
    print("🔥 Starting FULL end-to-end sanity check with real API calls...")
    print()
    
    # Load environment
    load_environment()
    
    # Create real endpoint
    print("1️⃣ Creating OpenAI endpoint...")
    try:
        endpoint = create_test_endpoint()
        print(f"✅ Endpoint created: {endpoint.model} via {endpoint.provider}")
    except Exception as e:
        print(f"❌ Failed to create endpoint: {e}")
        return 1
    print()
    
    # Create test sample
    print("2️⃣ Creating test GSM8K sample...")
    sample_data = create_test_gsm8k_sample()
    print(f"✅ Test question: {sample_data['question'][:50]}...")
    print(f"✅ Expected answer: {sample_data['answer']}")
    print()
    
    # Test different variants
    variants_to_test = ["control", "frustration", "collaborative"]
    
    for variant_name in variants_to_test:
        print(f"3️⃣ Testing variant: {variant_name}")
        
        # Create job
        variant_info = PROMPT_VARIANTS[variant_name]
        job = Job(
            sample_id=f"sanity_{variant_name}",
            sample_data=sample_data,
            variant_name=variant_name,
            transform_name=variant_info.transform_name
        )
        
        # Process with temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            worker_id = "sanity_worker"
            
            try:
                print(f"   🤖 Processing job with real API call...")
                result = await process_job(job, endpoint, output_dir, worker_id)
                
                print(f"   ✅ Job completed successfully!")
                print(f"   📊 Metrics: {result.metrics}")
                print(f"   🔄 Turns used: {result.metadata.get('turns_used', 'unknown')}")
                
                # Verify files were created
                variant_dir = output_dir / variant_name / job.sample_id
                trajectory_file = variant_dir / "trajectory.jsonl"
                agent_state_file = variant_dir / "agent_state.json"
                sample_file = variant_dir / "sample.json"
                
                assert trajectory_file.exists(), "Trajectory file missing"
                assert agent_state_file.exists(), "Agent state file missing"  
                assert sample_file.exists(), "Sample file missing"
                
                # Verify JSON content
                with open(trajectory_file) as f:
                    trajectory_lines = f.readlines()
                    print(f"   📝 Trajectory has {len(trajectory_lines)} messages")
                    
                    # Show first and last message for verification
                    if trajectory_lines:
                        first_msg = json.loads(trajectory_lines[0])
                        print(f"   👤 First message role: {first_msg.get('role')}")
                        
                        if len(trajectory_lines) > 1:
                            last_msg = json.loads(trajectory_lines[-1])
                            print(f"   🤖 Last message role: {last_msg.get('role')}")
                            if last_msg.get('content'):
                                content_preview = last_msg['content'][:100]
                                print(f"   💬 Response preview: {content_preview}...")
                
                with open(agent_state_file) as f:
                    agent_state = json.load(f)
                    print(f"   🧠 Agent state saved successfully")
                
                with open(sample_file) as f:
                    sample_result = json.load(f)
                    print(f"   📋 Sample data: variant={sample_result.get('variant')}")
                
                print(f"   ✅ All files validated for {variant_name}")
                
            except Exception as e:
                print(f"   ❌ Failed processing {variant_name}: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        print()
    
    print("🎉 FULL END-TO-END SANITY CHECK PASSED!")
    print("🚀 Pipeline is ready for multi-GPU deployment")
    return 0

async def main():
    """Run the full sanity check."""
    try:
        return await test_full_pipeline()
    except KeyboardInterrupt:
        print("\n⚠️  Sanity check interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Sanity check failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)