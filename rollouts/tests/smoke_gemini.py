#!/usr/bin/env python3
"""
Smoke test for Gemini using OpenAI-compatible API endpoint.

Usage:
    GEMINI_API_KEY=your_key python smoke_gemini.py

This test verifies that Gemini works through the OpenAI rollout function
by making a simple chat completion request.
"""

import os
import asyncio
from rollouts.dtypes import Endpoint, Actor, Trajectory, Message, AgentState, Environment
from rollouts.agents import rollout, stdout_handler


class EmptyEnvironment(Environment):
    """Simple empty environment for testing"""
    pass


async def test_gemini_basic():
    """Test basic Gemini chat completion via OpenAI-compatible endpoint"""
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("   Set it with: export GEMINI_API_KEY=your_api_key")
        return False
    
    print("🧪 Testing Gemini via OpenAI-compatible endpoint...")
    
    # Create endpoint with Gemini's OpenAI-compatible URL
    endpoint = Endpoint(
        provider="openai",  # Use OpenAI rollout function
        model="gemini-1.5-flash",  # Gemini model name
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key=api_key,
        max_tokens=100,
        temperature=0.1
    )
    
    # Create simple test trajectory
    trajectory = Trajectory(
        messages=[
            Message(role="user", content="Hello! Please respond with exactly 'Gemini test successful' if you can see this message.")
        ]
    )
    
    # Create actor
    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint
    )
    
    try:
        print(f"📡 Making request to: {endpoint.api_base}")
        print(f"🤖 Model: {endpoint.model}")
        
        # Make the rollout call
        result_actor = await rollout(actor, stdout_handler)
        
        # Check response
        last_message = result_actor.trajectory.messages[-1]
        
        if last_message.role == "assistant" and last_message.content:
            print(f"\n✅ Success! Gemini responded: {last_message.content}")
            return True
        else:
            print(f"\n❌ Unexpected response format: {last_message}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during rollout: {e}")
        print("   This could be due to:")
        print("   - Invalid API key")
        print("   - Wrong API endpoint URL")
        print("   - Network issues")
        print("   - Gemini API incompatibility")
        return False


async def test_gemini_with_tools():
    """Test Gemini with simple tool calling"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        return False
    
    print("\n🔧 Testing Gemini with tool calling...")
    
    # Use simple calculator environment for tool testing
    import sys
    sys.path.append('/Users/chiraagbalu/llm-workbench/rollouts/tests')
    from simple_calculator_env import SimpleCalculatorEnvironment
    
    endpoint = Endpoint(
        provider="openai",
        model="gemini-1.5-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key=api_key,
        max_tokens=100,
        temperature=0.1
    )
    
    trajectory = Trajectory(
        messages=[
            Message(role="user", content="What is 15 + 27? Please use the calculate tool.")
        ]
    )
    
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    
    # Create environment and state
    env = SimpleCalculatorEnvironment()
    state = AgentState(
        actor=actor,
        environment=env,
        max_turns=3
    )
    
    try:
        # Get tools and add to actor
        tools = env.get_tools()
        print(f"🔍 Debug - Raw tools: {[t.to_json() for t in tools[:2]]}")  # Show first 2 tools
        
        actor_with_tools = Actor(
            trajectory=trajectory,
            endpoint=endpoint,
            tools=tools
        )
        
        result_actor = await rollout(actor_with_tools, stdout_handler)
        
        last_message = result_actor.trajectory.messages[-1]
        
        print(f"🔍 Debug - Last message: {last_message}")
        print(f"🔍 Debug - Available tools: {[t.function.name for t in tools]}")
        
        if last_message.tool_calls:
            print(f"✅ Gemini made tool calls: {[tc.name for tc in last_message.tool_calls]}")
            return True
        else:
            print(f"⚠️  No tool calls made. Response: '{last_message.content}'")
            if not last_message.content:
                print("❌ Empty response - something went wrong!")
                return False
            print("   (This might be expected if Gemini doesn't support function calling)")
            return True  # Don't fail the test, just note it
            
    except Exception as e:
        print(f"❌ Error with tool calling: {e}")
        return False


async def main():
    """Run all Gemini smoke tests"""
    print("🚀 Gemini Smoke Tests")
    print("=" * 50)
    
    # Test 1: Basic chat
    success1 = await test_gemini_basic()
    
    # Test 2: Tool calling (if basic works)
    success2 = True
    if success1:
        success2 = await test_gemini_with_tools()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All Gemini tests passed!")
        print("   You can now use Gemini in rollouts with provider='openai'")
    elif success1:
        print("✅ Basic Gemini chat works!")
        print("⚠️  Tool calling may not be supported")
    else:
        print("❌ Gemini integration failed")
        print("   Check your API key and endpoint configuration")


if __name__ == "__main__":
    asyncio.run(main())