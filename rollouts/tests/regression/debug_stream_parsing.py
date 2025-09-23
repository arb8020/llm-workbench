#!/usr/bin/env python3
"""
Debug Gemini stream parsing by monkey patching aggregate_stream
"""

import os
import asyncio
import json
from typing import AsyncIterator, Callable, Awaitable, Any
from rollouts.dtypes import (
    Endpoint, Actor, Trajectory, Message, StreamChunk, ChatCompletion, 
    Choice, Usage, ToolCall
)
from rollouts.agents import rollout, stdout_handler
import sys
sys.path.append('/Users/chiraagbalu/llm-workbench/rollouts/tests')
from two_tool_test import TwoToolEnvironment


# Store original aggregate_stream
original_aggregate_stream = None


async def debug_aggregate_stream(
    stream: AsyncIterator, 
    on_chunk: Callable[[StreamChunk], Awaitable[None]]
) -> ChatCompletion:
    """Debug version of aggregate_stream that logs everything"""
    
    print(f"\n🔍 DEBUG STREAM PARSING")
    print("=" * 50)
    
    # Accumulate the response
    accumulated_content = ""
    finish_reason = None
    response_id = None
    created = None
    
    # Tool call buffer for partial accumulation
    call_buf: dict[int, dict[str, Any]] = {}
    
    chunk_count = 0
    
    async for chunk in stream:
        chunk_count += 1
        print(f"\n📦 Chunk {chunk_count}:")
        print(f"   ID: {getattr(chunk, 'id', 'N/A')}")
        print(f"   Object: {getattr(chunk, 'object', 'N/A')}")
        print(f"   Choices: {len(getattr(chunk, 'choices', []))}")
        
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            print(f"   Choice index: {getattr(choice, 'index', 'N/A')}")
            print(f"   Finish reason: {getattr(choice, 'finish_reason', 'N/A')}")
            
            if hasattr(choice, 'delta'):
                delta = choice.delta
                print(f"   Delta content: {repr(getattr(delta, 'content', None))}")
                print(f"   Delta tool_calls: {getattr(delta, 'tool_calls', None)}")
                
                # Handle content streaming
                if getattr(delta, 'content', None):
                    accumulated_content += delta.content
                    await on_chunk(StreamChunk("token", {"text": delta.content}))
                
                # Handle tool calls with detailed logging
                if getattr(delta, 'tool_calls', None):
                    print(f"   🔧 TOOL CALLS DETECTED:")
                    for tool_call in delta.tool_calls:
                        idx = getattr(tool_call, 'index', 0)
                        print(f"      Index: {idx}")
                        print(f"      ID: {getattr(tool_call, 'id', 'N/A')}")
                        print(f"      Type: {getattr(tool_call, 'type', 'N/A')}")
                        
                        # Initialize if needed
                        if idx not in call_buf:
                            call_buf[idx] = {
                                "id": "",
                                "type": "function", 
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        # Update tool call details
                        if getattr(tool_call, 'id', None):
                            call_buf[idx]["id"] = tool_call.id
                            print(f"         Updated ID: {tool_call.id}")
                            
                        if hasattr(tool_call, 'function') and tool_call.function:
                            func = tool_call.function
                            if getattr(func, 'name', None):
                                call_buf[idx]["function"]["name"] = func.name
                                print(f"         Function name: {func.name}")
                            if getattr(func, 'arguments', None):
                                call_buf[idx]["function"]["arguments"] += func.arguments
                                print(f"         Arguments chunk: {repr(func.arguments)}")
                                print(f"         Total arguments: {repr(call_buf[idx]['function']['arguments'])}")
                    
                    # Emit partial tool call event
                    await on_chunk(StreamChunk("tool_call_partial", {"calls": list(call_buf.values())}))
            
            # Handle finish reason
            if getattr(choice, 'finish_reason', None):
                finish_reason = choice.finish_reason
                print(f"   🏁 FINISH REASON: {finish_reason}")
        
        # Store metadata from first chunk
        if response_id is None:
            response_id = getattr(chunk, 'id', 'unknown')
            created = getattr(chunk, 'created', 0)
    
    print(f"\n📊 STREAM SUMMARY:")
    print(f"   Total chunks: {chunk_count}")
    print(f"   Final content: {repr(accumulated_content)}")
    print(f"   Tool call buffer: {call_buf}")
    print(f"   Finish reason: {finish_reason}")
    
    # Parse accumulated tool calls and emit completion events
    tool_calls = []
    for idx, tc in sorted(call_buf.items()):
        if tc["function"]["name"]:  # Only process complete tool calls
            print(f"\n🔧 Processing tool call {idx}:")
            print(f"   Name: {tc['function']['name']}")
            print(f"   Raw arguments: {repr(tc['function']['arguments'])}")
            
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                print(f"   Parsed arguments: {args}")
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON DECODE ERROR: {e}")
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
    
    print(f"\n✅ FINAL MESSAGE:")
    print(f"   Content: {repr(final_message.content)}")
    print(f"   Tool calls: {len(final_message.tool_calls)}")
    for tc in final_message.tool_calls:
        print(f"      - {tc.name}({tc.args})")
    
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


async def test_with_debug_parsing():
    """Test the problematic case with detailed stream debugging"""
    
    # Monkey patch the aggregate_stream function
    import rollouts.agents
    global original_aggregate_stream
    original_aggregate_stream = rollouts.agents.aggregate_stream
    rollouts.agents.aggregate_stream = debug_aggregate_stream
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not set")
            return
        
        print("🧪 Testing problematic case with debug parsing:")
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
        
        result_actor = await rollout(actor, stdout_handler)
        
        print(f"\n🏁 FINAL RESULT:")
        last_message = result_actor.trajectory.messages[-1]
        print(f"   Content: {repr(last_message.content)}")
        print(f"   Tool calls: {len(last_message.tool_calls)}")
        
    finally:
        # Restore original function
        if original_aggregate_stream:
            rollouts.agents.aggregate_stream = original_aggregate_stream


if __name__ == "__main__":
    asyncio.run(test_with_debug_parsing())