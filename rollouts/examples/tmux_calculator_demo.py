#!/usr/bin/env python3
"""
Tmux Calculator Demo - Interactive agent with tool confirmations via tmux

This demo:
1. Creates a tmux session with a calculator agent
2. Redirects agent I/O to the tmux window 
3. Uses named pipes for tool confirmations
4. Requires confirmation for divide() calls

Usage:
    python tmux_calculator_demo.py
    
Then attach to see the agent: tmux attach -t calc_demo
"""

import asyncio
import os
import uuid
import time
import sys
from pathlib import Path

try:
    import libtmux
    LIBTMUX_AVAILABLE = True
except ImportError:
    LIBTMUX_AVAILABLE = False

# Add rollouts to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts import (
    Endpoint, Actor, AgentState, RunConfig, CalculatorEnvironment,
    Message, Trajectory, run_agent, ToolCall, ToolConfirmResult, ToolResult
)

async def main():
    if not LIBTMUX_AVAILABLE:
        print("❌ libtmux not available. Install with: pip install libtmux")
        return
        
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set ANTHROPIC_API_KEY")
        return
    
    # Create unique session name
    session_name = f"calc_demo_{int(time.time())}"
    pipe_path = f"/tmp/calc_demo_pipe_{uuid.uuid4().hex[:8]}"
    
    print("🚀 Starting tmux calculator demo...")
    print(f"📺 Session: {session_name}")
    print(f"📡 Pipe: {pipe_path}")
    print(f"👀 Attach with: tmux attach -t {session_name}")
    print(f"✋ Respond with: echo 'y' > {pipe_path}")
    
    # Create named pipe for confirmations
    if os.path.exists(pipe_path):
        os.unlink(pipe_path)
    os.mkfifo(pipe_path)
    
    try:
        # Create tmux session
        server = libtmux.Server()
        session = server.new_session(session_name=session_name, detach=True)
        window = session.windows[0]
        pane = window.panes[0]
        
        # Setup tmux output handlers
        def tmux_chunk_handler(chunk):
            if hasattr(chunk, 'content') and chunk.content:
                # Send content to tmux, escaping special characters
                content = chunk.content.replace("'", "'\"'\"'")
                pane.send_keys(f"echo -n '{content}'", enter=False)
        
        def tmux_confirm_tool(tool_call: ToolCall, state: AgentState, run_config: RunConfig):
            # Display tool confirmation in tmux
            pane.send_keys(f"echo '🔧 Tool confirmation: {tool_call.name}({tool_call.args})'")
            pane.send_keys(f"echo '   Respond with: echo y > {pipe_path}'")
            
            # Read response from named pipe (blocking)
            try:
                with open(pipe_path, 'r') as pipe:
                    response = pipe.read().strip()
                    
                if response.lower() in ['y', 'yes']:
                    pane.send_keys("echo '✅ Confirmed'")
                    return state, ToolConfirmResult(proceed=True)
                else:
                    pane.send_keys("echo '❌ Rejected'") 
                    return state, ToolConfirmResult(
                        proceed=False,
                        tool_result=ToolResult(
                            call_id=tool_call.id,
                            ok=False,
                            error="Rejected by user"
                        )
                    )
            except Exception as e:
                pane.send_keys(f"echo '❌ Pipe error: {e}'")
                return state, ToolConfirmResult(
                    proceed=False,
                    tool_result=ToolResult(
                        call_id=tool_call.id,
                        ok=False,
                        error=f"Confirmation error: {e}"
                    )
                )
        
        # Create agent configuration
        sys_msg = Message(
            role="system",
            content="You are a calculator assistant. Help solve math problems step by step."
        )
        user_msg = Message(
            role="user",
            content="Calculate 100 / 5, then multiply by 3. Show your work."
        )
        
        trajectory = Trajectory(messages=[sys_msg, user_msg])
        endpoint = Endpoint(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            api_base="https://api.anthropic.com"
        )
        
        actor = Actor(trajectory=trajectory, endpoint=endpoint)
        environment = CalculatorEnvironment()
        
        initial_state = AgentState(
            actor=actor,
            environment=environment,
            turn_idx=0,
            max_turns=8
        )
        
        # Configure with tmux handlers
        run_config = RunConfig(
            on_chunk=tmux_chunk_handler,
            confirm_tool=tmux_confirm_tool
        )
        
        # Display startup info in tmux
        pane.send_keys("echo '🚀 Calculator Agent Starting...'")
        pane.send_keys("echo '📋 Task: Calculate 100 / 5, then multiply by 3'")
        pane.send_keys("echo ''")
        
        # Run agent
        states = await run_agent(initial_state, run_config)
        
        # Display completion
        final_state = states[-1]
        pane.send_keys("echo ''")
        pane.send_keys("echo '✅ Agent completed!'")
        pane.send_keys(f"echo '🔢 Final result: {final_state.environment.current_value}'")
        pane.send_keys(f"echo '🔄 Turns: {final_state.turn_idx}'")
        
        if final_state.stop:
            pane.send_keys(f"echo '🛑 Stop reason: {final_state.stop.value}'")
        
        pane.send_keys("echo ''")
        pane.send_keys("echo '🎯 Demo complete! Session will remain open for inspection.'")
        pane.send_keys("exec bash")  # Keep session alive
        
        print("✅ Agent completed!")
        print(f"📺 Session '{session_name}' remains open for inspection")
        print(f"👀 View with: tmux attach -t {session_name}")
        print(f"🗑️  Clean up with: tmux kill-session -t {session_name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up pipe
        if os.path.exists(pipe_path):
            os.unlink(pipe_path)

if __name__ == "__main__":
    asyncio.run(main())