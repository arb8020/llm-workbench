#!/usr/bin/env python3
"""
Local test script for worker_experiment.py logic

Tests the core functionality without deploying to remote GPU machines.
This catches serialization and data structure issues early.

Usage:
    python test_worker_locally.py
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Import worker logic
from worker_experiment import process_job, Job
from launch_experiment import PROMPT_VARIANTS

# Import rollouts for testing
from rollouts.evaluation import evaluate_sample, EvalSample
from rollouts.dtypes import Message, Endpoint, AgentState, Trajectory

def create_mock_eval_sample() -> EvalSample:
    """Create a mock EvalSample for testing serialization."""
    # Create mock messages
    messages = [
        Message(role="user", content="What is 2 + 2?"),
        Message(role="assistant", content="The answer is 4.")
    ]
    
    # Create mock trajectory
    trajectory = Trajectory(messages=messages)
    
    # Create mock agent states
    from rollouts.dtypes import Actor
    from worker_experiment import NoToolsEnvironment
    
    mock_actor = MagicMock()
    mock_actor.trajectory = trajectory
    mock_environment = NoToolsEnvironment()
    
    mock_agent_state = AgentState(
        actor=mock_actor,
        environment=mock_environment,
        max_turns=10,
        turn_idx=1
    )
    
    # Create the eval sample
    return EvalSample(
        sample_id="test_sample",
        input_data={"question": "What is 2 + 2?", "answer": "4"},
        trajectory=trajectory,
        agent_states=[mock_agent_state],  # This is the key fix - it's agent_states not agent_state
        metrics={"accuracy": 1.0, "reward": 1.0},
        metadata={"turns_used": 1}
    )

async def test_process_job_serialization():
    """Test that process_job can serialize EvalSample correctly."""
    
    print("🧪 Testing process_job serialization...")
    
    # Create a mock job
    job = Job(
        sample_id="test_sample",
        sample_data={"question": "What is 2 + 2?", "answer": "4"},
        variant_name="control",
        transform_name="identity_transform"
    )
    
    # Create a mock endpoint 
    endpoint = MagicMock()
    
    # Create a mock eval sample
    mock_result = create_mock_eval_sample()
    
    # Mock the evaluate_sample function
    original_evaluate = evaluate_sample
    async def mock_evaluate_sample(*args, **kwargs):
        return mock_result
    
    # Patch evaluate_sample
    import worker_experiment
    worker_experiment.evaluate_sample = mock_evaluate_sample
    
    try:
        # Test with a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            worker_id = "test_worker"
            
            # This should not crash
            await process_job(job, endpoint, output_dir, worker_id)
            
            # Verify files were created
            sample_dir = output_dir / "control" / "test_sample"
            assert sample_dir.exists(), "Sample directory should exist"
            
            trajectory_file = sample_dir / "trajectory.jsonl"
            assert trajectory_file.exists(), "Trajectory file should exist"
            
            agent_state_file = sample_dir / "agent_state.json"
            assert agent_state_file.exists(), "Agent state file should exist"
            
            sample_file = sample_dir / "sample.json"
            assert sample_file.exists(), "Sample file should exist"
            
            print("✅ All files created successfully")
            
            # Test that files contain valid JSON
            with open(trajectory_file) as f:
                for line in f:
                    json.loads(line.strip())  # Should not raise
            print("✅ Trajectory JSONL is valid")
            
            with open(agent_state_file) as f:
                json.load(f)  # Should not raise
            print("✅ Agent state JSON is valid")
            
            with open(sample_file) as f:
                json.load(f)  # Should not raise
            print("✅ Sample JSON is valid")
            
    finally:
        # Restore original function
        worker_experiment.evaluate_sample = original_evaluate

def test_prompt_variants():
    """Test that all prompt variants are properly defined."""
    print("🧪 Testing prompt variants...")
    
    required_variants = ["control", "frustration", "impatience", "anxiety", 
                        "collaborative", "patience", "calm"]
    
    for variant in required_variants:
        assert variant in PROMPT_VARIANTS, f"Missing variant: {variant}"
        variant_obj = PROMPT_VARIANTS[variant]
        assert hasattr(variant_obj, 'name'), f"Variant {variant} missing name"
        assert hasattr(variant_obj, 'transform_name'), f"Variant {variant} missing transform_name"
        assert hasattr(variant_obj, 'description'), f"Variant {variant} missing description"
    
    print(f"✅ All {len(required_variants)} prompt variants are properly defined")

async def main():
    """Run all tests."""
    print("🚀 Starting local worker tests...")
    
    try:
        # Test basic data structures
        test_prompt_variants()
        
        # Test serialization logic (the main issue)
        await test_process_job_serialization()
        
        print("\n🎉 All tests passed! Worker logic should work on remote machines.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)