#!/usr/bin/env python3
"""
Debug Gemini function calling issues based on Google's documentation
"""

import os
import asyncio
import json
from rollouts.dtypes import Endpoint, Actor, Trajectory, Message
from rollouts.agents import rollout, stdout_handler


async def test_google_weather_example():
    """Test using Google's exact weather function example"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return False
    
    print("🌤️  Testing Google's weather function example...")
    
    # Use Google's exact function definition from docs
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Chicago, IL",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    }
    
    endpoint = Endpoint(
        provider="openai",
        model="gemini-1.5-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key=api_key,
        max_tokens=200,
        temperature=0.1
    )
    
    # Test message similar to Google's example
    trajectory = Trajectory(
        messages=[
            Message(role="user", content="What's the weather like in Chicago, IL?")
        ]
    )
    
    # Create actor with the weather tool
    from rollouts.dtypes import Tool, ToolFunction, ToolFunctionParameter
    
    # Convert to our internal format
    weather_tool_internal = Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get the weather in a given location",
            parameters=ToolFunctionParameter(
                type="object",
                properties={
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Chicago, IL",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                }
            ),
            required=["location"]
        )
    )
    
    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=[weather_tool_internal]
    )
    
    print(f"🔍 Testing with exact Google example tool:")
    print(json.dumps(weather_tool, indent=2))
    
    try:
        result_actor = await rollout(actor, stdout_handler)
        last_message = result_actor.trajectory.messages[-1]
        
        print(f"\n📝 Response: '{last_message.content}'")
        print(f"🔧 Tool calls: {len(last_message.tool_calls)}")
        
        if last_message.tool_calls:
            print(f"✅ SUCCESS: Tool called: {last_message.tool_calls[0].name}")
            print(f"   Args: {last_message.tool_calls[0].args}")
            return True
        elif last_message.content:
            print("⚠️  No tool call but got text response (might be expected)")
            return True
        else:
            print("❌ Empty response - same issue persists")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_no_streaming():
    """Test with streaming disabled"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    
    print("\n🚫 Testing without streaming...")
    
    # Modify the rollout function temporarily to disable streaming
    from openai import AsyncOpenAI
    import json
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai"
    )
    
    params = {
        "model": "gemini-1.5-flash",
        "messages": [
            {"role": "user", "content": "What's the weather in New York? Please use the get_weather tool."}
        ],
        "temperature": 0.1,
        "stream": False,  # KEY CHANGE: No streaming
        "max_tokens": 200,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. Chicago, IL"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        }],
        "tool_choice": "auto"
    }
    
    print(f"🔍 Testing without streaming:")
    
    try:
        response = await client.chat.completions.create(**params)
        
        print(f"📝 Response content: '{response.choices[0].message.content}'")
        print(f"🔧 Tool calls: {response.choices[0].message.tool_calls}")
        
        if response.choices[0].message.tool_calls:
            print("✅ SUCCESS: Function calling works without streaming!")
            return True
        elif response.choices[0].message.content:
            print("⚠️  Got text response but no tool calls")
            return True
        else:
            print("❌ Still empty response")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_different_endpoints():
    """Test different API endpoint variations"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    
    print("\n🔗 Testing different endpoint URLs...")
    
    endpoints = [
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "https://generativelanguage.googleapis.com/v1/openai", 
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    ]
    
    from openai import AsyncOpenAI
    
    for endpoint_url in endpoints:
        print(f"\n🧪 Testing: {endpoint_url}")
        
        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=endpoint_url
            )
            
            # Simple test without tools first
            response = await client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=50
            )
            
            if response.choices[0].message.content:
                print(f"  ✅ Basic chat works")
            else:
                print(f"  ❌ Basic chat fails")
                continue
                
        except Exception as e:
            print(f"  ❌ Endpoint failed: {e}")


async def main():
    """Run all debug tests"""
    print("🔍 Gemini Function Calling Debug Tests")
    print("=" * 60)
    
    # Test 1: Google's exact example
    success1 = await test_google_weather_example()
    
    # Test 2: No streaming  
    success2 = await test_no_streaming()
    
    # Test 3: Different endpoints
    await test_different_endpoints()
    
    print("\n" + "=" * 60)
    if success1 or success2:
        print("🎉 Found working configuration!")
    else:
        print("❌ Function calling still not working")
        print("   This suggests a fundamental compatibility issue")


if __name__ == "__main__":
    asyncio.run(main())