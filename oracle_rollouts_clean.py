#!/usr/bin/env python3
"""
Clean Oracle script using rollouts system properly - no tools, no path manipulation.
"""

import asyncio
import os
from dotenv import load_dotenv
from rollouts.dtypes import Endpoint, Actor, AgentState, Message, Trajectory, RunConfig
from rollouts.environments import BasicEnvironment
from rollouts.agents import run_agent, stdout_handler

async def oracle_analysis():
    load_dotenv()
    
    # Read analysis
    with open("GPT2_JAX_ANALYSIS.md", "r") as f:
        analysis_content = f.read()
    
    # Create messages
    system_msg = Message(
        role="system",
        content="You are an expert in numerical computing, JAX, PyTorch, and transformer implementations. Provide actionable fixes for GPT-2 precision issues."
    )
    
    user_msg = Message(
        role="user",
        content=f"Analyze this GPT-2 precision issue and provide specific fixes:\n\n{analysis_content}"
    )
    
    # Create rollouts components
    trajectory = Trajectory(messages=[system_msg, user_msg])
    
    endpoint = Endpoint(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        api_base="https://api.anthropic.com",
        temperature=0.1,
        max_tokens=4000
    )
    
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    environment = BasicEnvironment()  # Clean environment, no tools
    
    agent_state = AgentState(
        actor=actor,
        environment=environment,
        max_turns=1
    )
    
    run_config = RunConfig(on_chunk=stdout_handler)
    
    print("🔮 Running Oracle Analysis via Rollouts (Clean)...")
    print("=" * 60)
    
    # Run the agent
    final_states = await run_agent(agent_state, run_config)
    
    if final_states and final_states[-1].actor.trajectory.messages:
        oracle_response = final_states[-1].actor.trajectory.messages[-1].content
        
        print("\n" + "=" * 60)
        print("🔮 ORACLE ANALYSIS COMPLETE")
        print("=" * 60)
        print(oracle_response)
        
        # Save results
        with open("ORACLE_ROLLOUTS_CLEAN.md", "w") as f:
            f.write("# Oracle Analysis via Clean Rollouts System\n\n")
            f.write(oracle_response)
        
        print(f"\n✅ Analysis saved to: ORACLE_ROLLOUTS_CLEAN.md")
    else:
        print("❌ No response from oracle")

if __name__ == "__main__":
    print("🚀 GPT-2 Oracle Analysis via Clean Rollouts")
    asyncio.run(oracle_analysis())