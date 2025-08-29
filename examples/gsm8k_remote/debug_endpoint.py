#!/usr/bin/env python3
"""Debug script to test endpoint configuration directly"""

import asyncio
import os
from rollouts.dtypes import Endpoint, Message, Trajectory, Actor, AgentState, RunConfig
from rollouts.agents import rollout_openai, stdout_handler

async def test_endpoint():
    """Test the endpoint configuration with a simple request"""
    
    # For this debug, I'll use the URL from the previous test
    # In real usage, this would come from deploy_qwen_vllm_server()
    test_url = "https://5l9xhug5x1tfs9-8000.proxy.runpod.net"  # This instance is terminated, just for illustration
    
    print(f"Testing endpoint: {test_url}")
    
    # Create endpoint
    endpoint = Endpoint(
        provider="openai",
        model="willcb/Qwen3-0.6B", 
        api_base=test_url,
        api_key="dummy",
        max_tokens=50,
        temperature=0.1
    )
    
    print(f"Endpoint config:")
    print(f"  provider: {endpoint.provider}")
    print(f"  model: {endpoint.model}")
    print(f"  api_base: {endpoint.api_base}")
    print(f"  api_key: {endpoint.api_key}")
    
    # Create simple trajectory
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is 2+2?")
    ]
    trajectory = Trajectory(messages=messages)
    
    # Create actor
    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=[]
    )
    
    print(f"\nTesting rollout_openai function...")
    try:
        # This will fail because the instance is terminated, but we can see the exact error
        result = await rollout_openai(actor, stdout_handler)
        print(f"✅ Success! Response: {result.trajectory.messages[-1].content[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        print(f"This helps us debug the exact issue")

if __name__ == "__main__":
    asyncio.run(test_endpoint())