#!/usr/bin/env python3
"""
Simple search demo - test search functionality without complex recursion
"""
import asyncio
import os
from rollouts import (
    Endpoint, Actor, AgentState, RunConfig, stdout_handler, 
    Message, Trajectory, CalculatorEnvironment, SearchEnvironment,
    create_search_config, run_agent
)

async def main():
    # Set up API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set ANTHROPIC_API_KEY to use this demo")
        return
    
    # Create initial messages
    sys_msg = Message(
        role="system",
        content="You are a helpful calculator assistant. Use the calculator tools (add, subtract, multiply, divide, clear) to solve math problems. Do NOT use search tools (branch/decompose) - solve problems directly with the calculator.",
    )
    user_msg = Message(
        role="user", 
        content="Calculate 15 + 27, then multiply the result by 3.",
    )
    
    # Create trajectory and endpoint
    trajectory = Trajectory(messages=[sys_msg, user_msg])
    
    endpoint = Endpoint(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        api_base="https://api.anthropic.com"
    )
    
    # Create calculator environment wrapped in search with max_depth=0 (no search tools)
    calculator_env = CalculatorEnvironment()
    search_config = create_search_config(
        context_passer_name="default",
        autonomous_subagents=True,
        max_depth=0,  # No search tools available
    )
    search_env = SearchEnvironment(calculator_env, search_config, depth=0)
    
    # Create actor and state
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    
    initial_state = AgentState(
        actor=actor,
        environment=search_env,
        turn_idx=0,
        max_turns=5
    )
    
    # Create run config
    run_config = RunConfig(
        on_chunk=stdout_handler
    )
    
    print("🚀 Starting simple search demo (no search tools)...")
    print("-" * 50)
    
    # Run the agent
    states = await run_agent(initial_state, run_config)
    
    # Print summary
    final_state = states[-1]
    print("\n" + "="*50)
    print("📊 Simple Search Demo Summary")
    print("="*50)
    print(f"✅ Turns completed: {final_state.turn_idx}")
    print(f"🧮 Final calculator value: {final_state.environment.inner_env.current_value}")
    if final_state.stop:
        print(f"🛑 Stopped because: {final_state.stop.value}")

if __name__ == "__main__":
    asyncio.run(main())