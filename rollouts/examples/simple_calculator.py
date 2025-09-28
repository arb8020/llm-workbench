#!/usr/bin/env python3
"""
Simple calculator demo using the extracted agent framework
"""
import asyncio
import os
from rollouts import (
    Endpoint, Actor, AgentState, RunConfig, stdout_handler,
    Message, Trajectory, CalculatorEnvironment,
    run_agent
)

async def main():
    # Set up API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY to use this demo")
        return
    
    # Create initial messages
    sys_msg = Message(
        role="system",
        content="You are a helpful calculator assistant. Help the user with their calculations.",
    )
    user_msg = Message(
        role="user", 
        content="Hi! Can you help me calculate 15 + 27 * 3?",
    )
    
    # Create trajectory and endpoint
    trajectory = Trajectory(messages=[sys_msg, user_msg])
    
    if os.getenv("ANTHROPIC_API_KEY"):
        endpoint = Endpoint(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            api_base="https://api.anthropic.com"
        )
    else:
        endpoint = Endpoint(
            provider="openai", 
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            api_base="https://api.openai.com/v1"
        )
    
    # Create actor and environment
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    environment = CalculatorEnvironment()
    
    # Create initial state
    initial_state = AgentState(
        actor=actor,
        environment=environment,
        turn_idx=0,
        max_turns=5
    )
    
    # Create run config
    run_config = RunConfig(
        on_chunk=stdout_handler
    )
    
    print("🚀 Starting calculator demo...")
    print("-" * 40)
    
    # Run the agent
    states = await run_agent(initial_state, run_config)
    
    # Print summary
    final_state = states[-1]
    print("\n" + "="*40)
    print("📊 Demo Summary")
    print("="*40)
    print(f"✅ Turns completed: {final_state.turn_idx}")
    # User handles type narrowing for their specific environment
    final_env = final_state.environment
    if isinstance(final_env, CalculatorEnvironment):
        print(f"🧮 Final calculator value: {final_env.current_value}")
    else:
        print("✅ Task completed")
    if final_state.stop:
        print(f"🛑 Stopped because: {final_state.stop.value}")

if __name__ == "__main__":
    asyncio.run(main())