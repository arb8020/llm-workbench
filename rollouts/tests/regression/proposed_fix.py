#!/usr/bin/env python3
"""
Proposed fix for Gemini multi-tool parsing issue
"""

import json
import asyncio
from typing import AsyncIterator, Callable, Awaitable, Any
from rollouts.dtypes import StreamChunk, ChatCompletion, Choice, Usage, ToolCall, Message


async def fixed_aggregate_stream(
    stream: AsyncIterator, 
    on_chunk: Callable[[StreamChunk], Awaitable[None]]
) -> ChatCompletion:
    """Fixed version of aggregate_stream that handles Gemini's tool call format"""
    
    # Accumulate the response
    accumulated_content = ""
    finish_reason = None
    response_id = None
    created = None
    
    # Tool call buffer - FIXED: Use auto-incrementing index when None
    call_buf: dict[int, dict[str, Any]] = {}
    next_auto_index = 0  # Auto-assign indexes when tool_call.index is None
    
    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        
        # Store metadata from first chunk
        if response_id is None:
            response_id = chunk.id
            created = chunk.created
        
        # Handle content streaming
        if delta.content:
            accumulated_content += delta.content
            await on_chunk(StreamChunk("token", {"text": delta.content}))
        
        # Handle tool calls with FIXED indexing
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                # FIXED: Handle None index properly
                idx = tool_call.index
                if idx is None:
                    # Gemini sends tool calls with index=None
                    # We need to auto-assign unique indexes
                    idx = next_auto_index
                    next_auto_index += 1
                
                # Initialize if needed
                if idx not in call_buf:
                    call_buf[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""}
                    }
                
                # Update tool call - FIXED: Don't concatenate, replace
                if tool_call.id:
                    call_buf[idx]["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        call_buf[idx]["function"]["name"] = tool_call.function.name
                    if tool_call.function.arguments:
                        # FIXED: For Gemini, arguments come complete in one chunk
                        # Don't concatenate, just set directly
                        call_buf[idx]["function"]["arguments"] = tool_call.function.arguments
            
            # Emit partial tool call event
            await on_chunk(StreamChunk("tool_call_partial", {"calls": list(call_buf.values())}))
        
        # Handle finish reason
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    
    # Parse accumulated tool calls and emit completion events
    tool_calls = []
    for idx, tc in sorted(call_buf.items()):
        if tc["function"]["name"]:  # Only process complete tool calls
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except json.JSONDecodeError:
                # Emit error chunk for malformed JSON
                await on_chunk(StreamChunk("tool_call_error", {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "error": f"Invalid JSON arguments: {tc['function']['arguments']}",
                    "index": idx
                }))
                continue
            
            # Emit completion event for each tool call
            await on_chunk(StreamChunk("tool_call_complete", {
                "id": tc["id"],
                "name": tc["function"]["name"],
                "args": args,
                "raw_arguments": tc["function"]["arguments"],
                "index": idx
            }))
            
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                args=args
            ))
    
    # Create final message
    final_message = Message(
        role="assistant",
        content=accumulated_content if accumulated_content else "",
        tool_calls=tool_calls
    )
    
    # Emit assistant completion event
    await on_chunk(StreamChunk("assistant_complete", {
        "content": accumulated_content,
        "tool_call_count": len(tool_calls),
        "finish_reason": finish_reason
    }))
    
    # Create completion object
    completion = ChatCompletion(
        id=response_id or "unknown",
        object="chat.completion",
        created=created or 0,
        model="",
        usage=Usage(0, 0, 0),
        choices=[Choice(0, final_message, finish_reason or "stop")]
    )
    
    return completion


# Test the fix
async def test_fix():
    """Test the fixed parsing"""
    import os
    import asyncio
    from rollouts.dtypes import Endpoint, Actor, Trajectory, Message
    from rollouts.agents import stdout_handler
    import rollouts.agents
    import sys
    sys.path.append('/Users/chiraagbalu/llm-workbench/rollouts/tests')
    from two_tool_test import TwoToolEnvironment
    
    # Monkey patch with the fix
    rollouts.agents.aggregate_stream = fixed_aggregate_stream
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return
    
    print("🔧 Testing FIXED aggregate_stream:")
    print("   'First say hello to Bob, then calculate 10 + 5'")
    
    endpoint = Endpoint(
        provider="openai",
        model="gemini-1.5-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai", 
        api_key=api_key,
        max_tokens=200,
        temperature=0.1
    )
    
    trajectory = Trajectory(
        messages=[Message(role="user", content="First say hello to Bob, then calculate 10 + 5.")]
    )
    
    env = TwoToolEnvironment()
    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=env.get_tools()
    )
    
    # Import rollout after patching
    from rollouts.agents import rollout
    
    result_actor = await rollout(actor, stdout_handler)
    
    print(f"\n🎉 FIXED RESULT:")
    last_message = result_actor.trajectory.messages[-1]
    print(f"   Content: {repr(last_message.content)}")
    print(f"   Tool calls: {len(last_message.tool_calls)}")
    for i, tc in enumerate(last_message.tool_calls):
        print(f"      {i+1}. {tc.name}({tc.args})")


if __name__ == "__main__":
    asyncio.run(test_fix())