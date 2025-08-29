#!/usr/bin/env python3
"""
Test Gemini with exactly two simple, clear tools
"""

import os
import asyncio
from typing import List
from rollouts.dtypes import (
    Tool, ToolFunction, ToolFunctionParameter, 
    ToolCall, ToolResult, Environment, AgentState, RunConfig,
    Endpoint, Actor, Trajectory, Message
)
from rollouts.agents import rollout, stdout_handler


class TwoToolEnvironment(Environment):
    """Environment with exactly two simple, distinct tools"""
    
    def __init__(self):
        pass
    
    async def serialize(self) -> dict:
        return {}
    
    @staticmethod
    async def deserialize(data: dict) -> 'TwoToolEnvironment':
        return TwoToolEnvironment()
    
    def get_tools(self) -> List[Tool]:
        return [
            # Tool 1: Simple greeting
            Tool(
                type="function",
                function=ToolFunction(
                    name="say_hello",
                    description="Say hello to someone",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "name": {
                                "type": "string", 
                                "description": "The name of the person to greet"
                            }
                        }
                    ),
                    required=["name"]
                )
            ),
            # Tool 2: Simple math
            Tool(
                type="function",
                function=ToolFunction(
                    name="add_numbers",
                    description="Add two numbers together",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"}
                        }
                    ),
                    required=["a", "b"]
                )
            )
        ]
    
    async def exec_tool(self, tool_call: ToolCall, current_state: 'AgentState',
                       run_config: 'RunConfig', checkpoint_store=None) -> ToolResult:
        """Execute the tools"""
        
        if tool_call.name == "say_hello":
            name = tool_call.args.get("name", "World")
            return ToolResult(
                call_id=tool_call.id,
                ok=True,
                content=f"Hello, {name}!"
            )
        
        elif tool_call.name == "add_numbers":
            try:
                a = float(tool_call.args.get("a", 0))
                b = float(tool_call.args.get("b", 0))
                result = a + b
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"{a} + {b} = {result}"
                )
            except Exception as e:
                return ToolResult(
                    call_id=tool_call.id,
                    ok=False,
                    error=f"Error adding numbers: {str(e)}"
                )
        
        return ToolResult(
            call_id=tool_call.id,
            ok=False,
            error=f"Unknown tool: {tool_call.name}"
        )


async def test_two_tools(prompt, expected_tool=None):
    """Test with exactly two tools and a specific prompt"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return False
    
    print(f"\n🧪 Testing: {prompt}")
    if expected_tool:
        print(f"   Expected tool: {expected_tool}")
    
    endpoint = Endpoint(
        provider="openai",
        model="gemini-1.5-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key=api_key,
        max_tokens=200,
        temperature=0.1
    )
    
    trajectory = Trajectory(
        messages=[Message(role="user", content=prompt)]
    )
    
    env = TwoToolEnvironment()
    tools = env.get_tools()
    
    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=tools
    )
    
    print(f"   Available tools: {[t.function.name for t in tools]}")
    
    try:
        result_actor = await rollout(actor, stdout_handler)
        last_message = result_actor.trajectory.messages[-1]
        
        print(f"   Response content: '{last_message.content}'")
        print(f"   Tool calls: {len(last_message.tool_calls)}")
        
        if last_message.tool_calls:
            for tc in last_message.tool_calls:
                print(f"     - {tc.name}({tc.args})")
            return True
        elif last_message.content:
            print(f"   ⚠️  Text response instead of tool call")
            return True
        else:
            print(f"   ❌ Empty response!")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def main():
    """Test various scenarios with exactly two tools"""
    
    print("🔬 Two-Tool Environment Tests")
    print("=" * 60)
    
    # Test 1: Math request (should use add_numbers)
    await test_two_tools(
        "What is 15 + 27? Please use the appropriate tool.",
        expected_tool="add_numbers"
    )
    
    # Test 2: Greeting request (should use say_hello)  
    await test_two_tools(
        "Please say hello to Alice using the greeting tool.",
        expected_tool="say_hello"
    )
    
    # Test 3: Ambiguous request (could use either)
    await test_two_tools(
        "Can you help me with something?",
        expected_tool=None
    )
    
    # Test 4: Sequential request (might use both)
    await test_two_tools(
        "First say hello to Bob, then calculate 10 + 5.",
        expected_tool="both"
    )
    
    # Test 5: Simple math without explicit tool mention
    await test_two_tools(
        "15 + 27 = ?",
        expected_tool="add_numbers"
    )
    
    print("\n" + "=" * 60)
    print("🔍 Analysis: Do any of these work with two tools?")


if __name__ == "__main__":
    asyncio.run(main())