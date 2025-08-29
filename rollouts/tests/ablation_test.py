#!/usr/bin/env python3
"""
Ablation test to identify which calculator tool causes Gemini to fail
"""

import os
import asyncio
from rollouts.dtypes import Endpoint, Actor, Trajectory, Message
from rollouts.agents import rollout, stdout_handler
from rollouts.environments.calculator import CalculatorEnvironment


class AblatedCalculatorEnvironment(CalculatorEnvironment):
    """Calculator environment with specific tools removed for testing"""
    
    def __init__(self, exclude_tools=None):
        super().__init__()
        self.exclude_tools = exclude_tools or []
    
    def get_tools(self):
        """Get all tools except excluded ones"""
        all_tools = super().get_tools()
        return [t for t in all_tools if t.function.name not in self.exclude_tools]


async def test_calculator_variant(exclude_tools, test_name):
    """Test calculator with specific tools excluded"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    
    print(f"\n🧪 {test_name}")
    print(f"   Excluding: {exclude_tools}")
    
    endpoint = Endpoint(
        provider="openai",
        model="gemini-1.5-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key=api_key,
        max_tokens=200,
        temperature=0.1
    )
    
    trajectory = Trajectory(
        messages=[
            Message(role="user", content="Please add 15 to the current value, then add 27.")
        ]
    )
    
    # Create ablated environment
    env = AblatedCalculatorEnvironment(exclude_tools=exclude_tools)
    available_tools = [t.function.name for t in env.get_tools()]
    
    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=env.get_tools()
    )
    
    print(f"   Available tools: {available_tools}")
    
    try:
        result_actor = await rollout(actor, stdout_handler)
        last_message = result_actor.trajectory.messages[-1]
        
        if last_message.tool_calls:
            print(f"   ✅ SUCCESS: Tool called: {[tc.name for tc in last_message.tool_calls]}")
            return True
        elif last_message.content:
            print(f"   ⚠️  Got text response: '{last_message.content[:50]}...'")
            return True
        else:
            print(f"   ❌ Empty response")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def main():
    """Run ablation tests to identify problematic tools"""
    
    print("🔬 Calculator Tool Ablation Tests")
    print("=" * 60)
    
    # Test 1: Baseline (all original tools)
    await test_calculator_variant([], "Baseline (all tools)")
    
    # Test 2: Remove clear
    await test_calculator_variant(["clear"], "Remove 'clear' tool")
    
    # Test 3: Remove complete_task  
    await test_calculator_variant(["complete_task"], "Remove 'complete_task' tool")
    
    # Test 4: Remove both clear and complete_task
    await test_calculator_variant(["clear", "complete_task"], "Remove 'clear' AND 'complete_task'")
    
    # Test 5: Keep only add/subtract (remove multiply, divide, clear, complete_task)
    await test_calculator_variant(["multiply", "divide", "clear", "complete_task"], "Keep only 'add' and 'subtract'")
    
    # Test 6: Keep only add (minimal test)
    await test_calculator_variant(["subtract", "multiply", "divide", "clear", "complete_task"], "Keep only 'add' tool")
    
    print("\n" + "=" * 60)
    print("🔍 Analysis complete! Check which variants work vs fail.")


if __name__ == "__main__":
    asyncio.run(main())