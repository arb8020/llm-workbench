#!/usr/bin/env python3
"""
Quick test script for GSM8K nnsight remote evaluation.

This script tests the core functionality without provisioning new GPUs.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
import requests

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.gsm8k_nnsight_remote.deploy_and_evaluate import (
    create_nnsight_endpoint, 
    GSMRewardFunction,
    extract_numerical_answer
)

async def test_endpoint_creation():
    """Test that we can create nnsight endpoints correctly."""
    print("🔧 Testing endpoint creation...")
    
    # Test basic endpoint
    endpoint = await create_nnsight_endpoint("http://example.com:8001", collect_activations=False)
    assert endpoint.model == "willcb/Qwen3-0.6B"
    assert endpoint.api_base == "http://example.com:8001/v1"
    print("✅ Basic endpoint creation works")
    
    # Test endpoint with activation collection (simplified)
    endpoint_with_activations = await create_nnsight_endpoint("http://example.com:8001", collect_activations=True)
    assert endpoint_with_activations.model == "willcb/Qwen3-0.6B"
    print("✅ Endpoint with activation collection works")

def test_answer_extraction():
    """Test numerical answer extraction from GSM8K answers."""
    print("🔢 Testing answer extraction...")
    
    test_cases = [
        ("The answer is #### 42", 42),
        ("So the total is $25.50. #### 25.5", 25.5),
        ("#### 100", 100),
        ("The final result: 3,250 items #### 3250", 3250),
        ("No clear answer here", None),
        ("Multiple numbers 10, 20, 30 #### 30", 30),
    ]
    
    for text, expected in test_cases:
        result = extract_numerical_answer(text)
        print(f"  '{text[:30]}...' -> {result} (expected: {expected})")
        if expected is None:
            assert result is None, f"Expected None but got {result}"
        else:
            assert result == expected, f"Expected {expected} but got {result}"
    
    print("✅ Answer extraction works correctly")

async def test_reward_function():
    """Test the GSM8K reward function."""
    print("🎯 Testing reward function...")
    
    from rollouts.dtypes import Trajectory, Message
    
    reward_fn = GSMRewardFunction()
    
    # Test correct answer
    problem = {"answer": "The answer is #### 42"}
    trajectory = Trajectory(messages=[
        Message(role="user", content="What is 6 * 7?"),
        Message(role="assistant", content="6 * 7 = 42. The answer is 42.")
    ])
    
    reward = await reward_fn(trajectory, problem)
    assert reward == 1.0, f"Expected reward 1.0 for correct answer, got {reward}"
    print("✅ Correct answer gives reward 1.0")
    
    # Test incorrect answer
    trajectory_wrong = Trajectory(messages=[
        Message(role="user", content="What is 6 * 7?"),
        Message(role="assistant", content="6 * 7 = 43. The answer is 43.")
    ])
    
    reward_wrong = await reward_fn(trajectory_wrong, problem)
    assert reward_wrong == 0.0, f"Expected reward 0.0 for wrong answer, got {reward_wrong}"
    print("✅ Incorrect answer gives reward 0.0")

def test_server_health(server_url: str):
    """Test if a server is responding to health checks."""
    print(f"🌐 Testing server health at {server_url}...")
    
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy: {health_data}")
            return True
        else:
            print(f"❌ Server unhealthy: {response.status_code} {response.text}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 GSM8K nnsight Remote - Quick Test")
    print("=" * 50)
    
    try:
        # Test core functions
        await test_endpoint_creation()
        test_answer_extraction()
        await test_reward_function()
        
        print("\n🎉 All unit tests passed!")
        
        # Test server connectivity (if provided)
        if len(sys.argv) > 1:
            server_url = sys.argv[1]
            print(f"\n🌐 Testing server connectivity...")
            if test_server_health(server_url):
                print("✅ Server test passed!")
            else:
                print("❌ Server test failed!")
                return False
        else:
            print("\n💡 To test server connectivity, run:")
            print("   python examples/gsm8k_nnsight_remote/quick_test.py http://SERVER_IP:8001")
        
        print("\n✅ Quick test completed successfully!")
        print("\n🚀 Ready to run full evaluation:")
        print("   python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py --samples 3 --collect-activations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)