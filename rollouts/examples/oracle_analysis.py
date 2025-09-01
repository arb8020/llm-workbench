#!/usr/bin/env python3
"""
Oracle Analysis Example - Using rollouts for AI-powered analysis tasks.

This example shows how to use rollouts with BasicEnvironment for clean text-based
AI analysis without confusing calculator or search tools. Perfect for:
- Code analysis
- Document review  
- Technical recommendations
- Any single-shot AI conversation

Usage:
    python examples/oracle_analysis.py
"""

import asyncio
import os
from dotenv import load_dotenv
from rollouts import (
    Endpoint, Actor, AgentState, Message, Trajectory, RunConfig,
    BasicEnvironment, run_agent, stdout_handler
)

async def analyze_text(input_text: str, analysis_prompt: str, 
                      model_provider: str = "anthropic", 
                      model_name: str = "claude-3-5-sonnet-20241022") -> str:
    """
    Use rollouts to analyze text with AI - clean and simple.
    
    Args:
        input_text: The text/document to analyze
        analysis_prompt: Instructions for what kind of analysis to perform
        model_provider: "anthropic" or "openai"
        model_name: Model to use
        
    Returns:
        AI analysis response as string
    """
    
    # Set up API credentials
    if model_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        api_base = "https://api.anthropic.com"
    elif model_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") 
        api_base = "https://api.openai.com/v1"
        model_name = "gpt-4"  # Override for OpenAI
    else:
        raise ValueError(f"Unsupported provider: {model_provider}")
    
    if not api_key:
        raise ValueError(f"{model_provider.upper()}_API_KEY not found in environment")
    
    # Create rollouts components
    system_msg = Message(
        role="system",
        content=analysis_prompt
    )
    
    user_msg = Message(
        role="user", 
        content=input_text
    )
    
    trajectory = Trajectory(messages=[system_msg, user_msg])
    
    endpoint = Endpoint(
        provider=model_provider,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=0.1,
        max_tokens=4000
    )
    
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    
    # Use BasicEnvironment - no tools, just clean text conversation
    environment = BasicEnvironment()
    
    agent_state = AgentState(
        actor=actor,
        environment=environment,
        max_turns=1  # Single response
    )
    
    run_config = RunConfig(on_chunk=stdout_handler)
    
    # Run the analysis
    final_states = await run_agent(agent_state, run_config)
    
    if final_states and final_states[-1].actor.trajectory.messages:
        return final_states[-1].actor.trajectory.messages[-1].content
    else:
        raise RuntimeError("No response from AI")

async def main():
    """Example: Analyze a code snippet for potential improvements."""
    
    load_dotenv()
    
    # Example input - a piece of code to analyze
    code_to_analyze = '''
def process_data(data):
    results = []
    for item in data:
        if item != None:
            if len(item) > 0:
                processed = item.upper().strip()
                if processed not in results:
                    results.append(processed)
    return results
'''
    
    analysis_prompt = """You are an expert code reviewer. Analyze the provided code and suggest improvements for:
1. Performance optimization
2. Code readability  
3. Python best practices
4. Edge case handling

Provide specific, actionable recommendations."""

    print("🔮 Running Oracle Analysis via Rollouts...")
    print("=" * 60)
    
    try:
        analysis = await analyze_text(
            input_text=f"Please analyze this Python code:\n\n```python{code_to_analyze}```",
            analysis_prompt=analysis_prompt,
            model_provider="anthropic"
        )
        
        print("\n" + "=" * 60)
        print("🔮 ANALYSIS COMPLETE")
        print("=" * 60)
        print(analysis)
        
        # Optionally save to file
        with open("analysis_output.md", "w") as f:
            f.write("# Code Analysis Report\n\n")
            f.write(analysis)
        
        print(f"\n✅ Analysis saved to: analysis_output.md")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    print("🚀 Oracle Analysis Example")
    asyncio.run(main())