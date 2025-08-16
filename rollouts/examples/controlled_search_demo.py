#!/usr/bin/env python3
"""
Controlled search demo - test search with depth=1 to prevent infinite recursion
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
        content="You are a helpful mathematical assistant. You can break problems down using 'decompose' if needed, but sub-agents should solve problems directly with calculator tools.",
    )
    user_msg = Message(
        role="user", 
        content="Calculate (10 + 5) * 2. Break this into steps if needed.",
    )
    
    # Create trajectory and endpoint
    trajectory = Trajectory(messages=[sys_msg, user_msg])
    
    endpoint = Endpoint(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        api_base="https://api.anthropic.com"
    )
    
    # Create calculator environment wrapped in search with max_depth=1 (one level only)
    calculator_env = CalculatorEnvironment()
    search_config = create_search_config(
        context_passer_name="default",
        autonomous_subagents=True,
        max_depth=1,  # Only one level of search
        timeout_per_branch=30.0  # 30 seconds per branch
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
    
    print("🚀 Starting controlled search demo (max_depth=1)...")
    print("-" * 50)
    
    # Run the agent
    states = await run_agent(initial_state, run_config)
    
    # Print summary
    final_state = states[-1]
    print("\n" + "="*50)
    print("📊 Controlled Search Demo Summary")
    print("="*50)
    print(f"✅ Turns completed: {final_state.turn_idx}")
    print(f"🧮 Final calculator value: {final_state.environment.inner_env.current_value}")
    if final_state.stop:
        print(f"🛑 Stopped because: {final_state.stop.value}")

if __name__ == "__main__":
    asyncio.run(main())