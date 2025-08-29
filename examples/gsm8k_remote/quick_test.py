#!/usr/bin/env python3
"""Quick test of the gsm8k_remote with current running server"""

import asyncio
from rollouts.evaluation import evaluate_sample
from rollouts.dtypes import Message, Endpoint, RunConfig
from rollouts.agents import stdout_handler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from deploy_and_evaluate import (
    NoToolsEnvironment, prepare_gsm8k_messages_no_tools, 
    make_correctness_reward, format_reward, efficiency_reward
)

async def quick_test():
    # Use current running server
    endpoint = Endpoint(
        provider="openai",
        model="willcb/Qwen3-0.6B",
        api_base="https://9709d01gbpcqro-8000.proxy.runpod.net/v1",
        api_key="dummy",
        max_tokens=500,  
        temperature=0.1
    )
    
    # Simple test sample
    sample = {
        "question": "If I have 5 apples and I eat 2, how many apples do I have left?",
        "answer": "3",
        "sample_id": "test_001"
    }
    
    environment = NoToolsEnvironment()
    run_config = RunConfig(on_chunk=stdout_handler)
    
    # Create reward functions
    reward_functions = [
        ("correctness", make_correctness_reward(sample)),
        ("format", format_reward),
        ("efficiency", efficiency_reward),
    ]
    
    print("🧪 Testing single GSM8K sample...")
    print(f"Question: {sample['question']}")
    print(f"Expected Answer: {sample['answer']}")
    print("="*50)
    
    try:
        result = await evaluate_sample(
            sample_data=sample,
            sample_id=sample["sample_id"],
            prepare_messages=prepare_gsm8k_messages_no_tools,
            reward_functions=reward_functions,
            environment=environment,
            endpoint=endpoint,
            run_config=run_config,
            max_turns=1,
            verbose=True
        )
        
        print("="*50)
        print(f"🔍 Results:")
        print(f"Sample ID: {result.sample_id}")
        print(f"Correctness: {result.metrics.get('correctness', 0.0)}")
        print(f"Format: {result.metrics.get('format', 0.0)}")
        print(f"Efficiency: {result.metrics.get('efficiency', 0.0)}")
        
        # Show the actual response
        assistant_messages = [m for m in result.trajectory.messages if m.role == "assistant"]
        if assistant_messages:
            print(f"\n📝 Model Response:")
            print(assistant_messages[-1].content)
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(quick_test())