#!/usr/bin/env python3
"""
Test script to debug rollouts routing logic.
"""
import asyncio
from rollouts.dtypes import Endpoint, Actor, Trajectory, Message
from rollouts.agents import rollout, stdout_handler

async def test_routing():
    """Test what rollout function is called for different providers."""
    
    # Create test endpoints with different providers
    test_endpoints = [
        ("openai", Endpoint(provider="openai", model="gpt-4", api_key="dummy", api_base="http://localhost:8000/v1")),
        ("vllm", Endpoint(provider="vllm", model="test-model", api_key="dummy", api_base="http://localhost:8000/v1")),
        ("moonshot", Endpoint(provider="moonshot", model="test-model", api_key="dummy", api_base="http://localhost:8000/v1")),
    ]
    
    for provider_name, endpoint in test_endpoints:
        print(f"\n🧪 Testing provider: {provider_name}")
        
        # Create a simple trajectory
        trajectory = Trajectory(messages=[
            Message(role="user", content="Hello, test message")
        ])
        
        # Create actor
        actor = Actor(trajectory=trajectory, endpoint=endpoint)
        
        try:
            # This should print debug info showing which provider is being used
            result = await rollout(actor, stdout_handler)
            print(f"✅ Provider {provider_name} completed successfully")
        except Exception as e:
            print(f"❌ Provider {provider_name} failed: {e}")
            print(f"   Error type: {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(test_routing())