#!/usr/bin/env python3
"""
Search calculator demo - demonstrates search capabilities wrapped around calculator
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
        content="You are a helpful mathematical assistant. Use the available tools to solve complex math problems. You have two key tools: 'decompose' breaks problems into sequential parallel subtasks, and 'branch' explores multiple solution methods simultaneously (like factoring vs quadratic formula). The best solution wins automatically. For quadratic equations, consider using decompose for step-by-step solving or branch to try different solution methods in parallel.",
    )
    user_msg = Message(
        role="user", 
        content="Solve the quadratic equation x² + 5x + 6 = 0. Find all real solutions and verify them. You can use 'decompose' to break this into sequential steps, or 'branch' to try multiple solution methods (factoring, quadratic formula, completing the square) in parallel.",
    )
    
    # Create trajectory and endpoint
    trajectory = Trajectory(messages=[sys_msg, user_msg])
    
    endpoint = Endpoint(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        api_base="https://api.anthropic.com"
    )
    
    # Create calculator environment wrapped in search
    calculator_env = CalculatorEnvironment()
    search_config = create_search_config(
        context_passer_name="default",
        autonomous_subagents=True,
        max_depth=2,
        timeout_per_branch=120.0  # 2 minutes per branch
    )
    search_env = SearchEnvironment(calculator_env, search_config)
    
    # Create actor and state
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    
    initial_state = AgentState(
        actor=actor,
        environment=search_env,
        turn_idx=0,
        max_turns=8
    )
    
    # Create run config
    run_config = RunConfig(
        on_chunk=stdout_handler
    )
    
    print("🚀 Starting quadratic equation solver demo...")
    print("📋 Problem: x² + 5x + 6 = 0")
    print("🔍 The agent should use 'decompose' to break this into logical steps")
    print("-" * 50)
    
    # Run the agent
    states = await run_agent(initial_state, run_config)
    
    # Print summary
    final_state = states[-1]
    print("\n" + "="*50)
    print("📊 Quadratic Solver Demo Summary")
    print("="*50)
    print(f"✅ Turns completed: {final_state.turn_idx}")
    print(f"🧮 Final calculator value: {final_state.environment.inner_env.current_value}")
    if final_state.stop:
        print(f"🛑 Stopped because: {final_state.stop.value}")

if __name__ == "__main__":
    asyncio.run(main())