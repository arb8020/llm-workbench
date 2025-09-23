# Core agent execution framework

import sys
import asyncio
import json
import os
import time
from dataclasses import replace
from typing import (Any, Dict, List, Optional, Tuple, Callable,
                   AsyncIterator, Awaitable)

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types import CompletionUsage

import aiohttp
from dacite import from_dict

import copy

# Environment class and other core types are now imported from dtypes

from .dtypes import (
    Tool, ToolCall, ToolResult, ToolConfirmResult, StopReason, StreamChunk, Message, Usage, Choice, ChatCompletion, Actor, AgentState, RunConfig
)
from .providers import (
    add_cache_control_to_last_content,
    aggregate_anthropic_stream,
    aggregate_stream,
    rollout_anthropic,
    rollout_moonshot,
    rollout_openai,
    rollout_vllm,
    verbose,
)

# ── Core Design Philosophy ────────────────────────────────────────────────────
#
# FULL STATE PASSING: Pass full state everywhere rather than using globals.
# Benefits: testable, checkpointable, parallelizable. Cost: verbose signatures.
# Pattern: Always return new state, never mutate in place.
#
# STATE IMMUTABILITY: All core data structures are frozen dataclasses.
# Benefits: time-travel debugging, safe concurrency, easy rollback.
# Cost: O(n) allocations per turn. Assumption: allocation cheaper than debugging.

# Core types (Endpoint, Actor, AgentState, RunConfig, Environment) are now imported from dtypes

# ── Utility functions ────────────────────────────────────────────────────────

def add_cache_control_to_last_content(
    messages, cache_control={"type": "ephemeral"}, max_cache_controls=4
):
    """Adds cache control to the last content item in messages."""
    if not messages:
        return messages
    new_messages = copy.deepcopy(messages)
    cache_control_count = sum(
        1
        for msg in new_messages
        for content in (
            msg["content"]
            if isinstance(msg.get("content"), list)
            else [msg.get("content")]
        )
        if isinstance(content, dict) and "cache_control" in content
    )
    if cache_control_count >= max_cache_controls:
        return new_messages
    last_message = new_messages[-1]
    if isinstance(last_message.get("content"), list) and last_message["content"]:
        last_content = last_message["content"][-1]
        if (
            isinstance(last_content, dict)
            and "type" in last_content
            and "cache_control" not in last_content
        ):
            last_content["cache_control"] = cache_control
    elif isinstance(last_message.get("content"), dict):
        if "cache_control" not in last_message["content"]:
            last_message["content"]["cache_control"] = cache_control
    return new_messages

def verbose(level: int) -> bool:
    """Simple verbose checker - just return True for now"""
    return True

async def handle_checkpoint_event(state: 'AgentState', event: str, run_config: 'RunConfig', 
                                 session_id: Optional[str] = None) -> None:
    """Handle checkpoint event if configured - stub for now"""
    pass

async def stdout_handler(chunk: StreamChunk):
    """Simple stdout handler for chunks"""
    if chunk.kind == "token":
        print(chunk.data["text"], end='', flush=True)
    elif chunk.kind == "tool_call_complete":
        print(f"\n🔧 Calling {chunk.data['name']}({chunk.data['args']})")
    elif chunk.kind == "tool_result":
        status = "✓" if chunk.data["ok"] else "✗"
        print(f"\n  {status} {chunk.data['content'][:100]}...")
    elif chunk.kind == "thinking":
        print(f"\033[95m{chunk.data['text']}\033[0m", end='', flush=True)

def _message_to_openai(m: Message) -> ChatCompletionMessageParam:
    """Convert our Message to OpenAI format"""
    msg: Dict[str, Any] = {"role": m.role}
    
    if m.content is not None:
        msg["content"] = m.content
    
    # Handle tool calls
    if m.tool_calls and m.role == "assistant":
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args)
                }
            }
            for tc in m.tool_calls
        ]
    
    # Handle tool results
    if m.role == "tool":
        msg["tool_call_id"] = m.tool_call_id  # Use the actual tool_call_id
        msg["content"] = m.content
    
    return msg


def _tool_to_openai(tool: Tool) -> Dict[str, Any]:
    """Convert our Tool to OpenAI function format"""
    return {
        "type": tool.type,
        "function": {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": {
                "type": tool.function.parameters.type,
                "properties": tool.function.parameters.properties,
                "required": tool.function.required
            }
        }
    }

def _parse_usage(u: CompletionUsage) -> Usage:
    return Usage(u.prompt_tokens, u.completion_tokens, u.total_tokens)

def _parse_completion(resp: Any) -> ChatCompletion:
    """Parse OpenAI response into our ChatCompletion"""
    # Extract tool calls from the response
    choices = []
    for c in resp.choices:
        tool_calls = []
        if hasattr(c.message, 'tool_calls') and c.message.tool_calls:
            for tc in c.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments)
                ))
        
        msg = Message(
            role=c.message.role,
            content=c.message.content,
            tool_calls=tool_calls
        )
        choices.append(Choice(c.index, msg, c.finish_reason))
    
    return ChatCompletion(
        id=resp.id,
        object=resp.object,
        created=resp.created,
        model=resp.model,
        usage=_parse_usage(resp.usage),
        choices=choices
    )

# ── Core agent functions ──────────────────────────────────────────────────────
async def aggregate_stream(
    stream: AsyncIterator, 
    on_chunk: Callable[[StreamChunk], Awaitable[None]]
) -> ChatCompletion:
    """Aggregate streaming chunks into a complete ChatCompletion"""
    # Accumulate the response
    accumulated_content = ""
    finish_reason = None
    response_id = None
    created = None
    
    # Tool call buffer for partial accumulation
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
        
        # Handle tool calls with better accumulation
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                # FIXED: Handle None index properly for Gemini compatibility
                idx = tool_call.index
                if idx is None:
                    # Gemini sends tool calls with index=None - auto-assign unique indexes
                    idx = next_auto_index
                    next_auto_index += 1
                
                # Initialize if needed
                if idx not in call_buf:
                    call_buf[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""}
                    }
                
                # Update tool call
                if tool_call.id:
                    call_buf[idx]["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        call_buf[idx]["function"]["name"] = tool_call.function.name
                    if tool_call.function.arguments:
                        call_buf[idx]["function"]["arguments"] += tool_call.function.arguments
            
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
        content=accumulated_content if accumulated_content else "",  # Always string, never None
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
        model="",  # Will be set by caller
        usage=Usage(0, 0, 0),  # Usage not available in streaming
        choices=[Choice(0, final_message, finish_reason or "stop")]
    )
    
    return completion


# TODO: handle api call failures. right now things are assumed to always succeed.
async def rollout_vllm(actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]],
                       max_api_retries = 16, backoff_base = 4) -> Actor:
    """API call to vllm instance"""

    messages = [_message_to_openai(m) for m in actor.trajectory.messages]

    params = {
        "model": actor.endpoint.model,
        "messages": messages,
        "max_tokens": actor.endpoint.max_tokens,
        "temperature": actor.endpoint.temperature,
        "stream": False,
        "logprobs": True,   # needed for RL
        "echo": True,       # needed for RL 
    }
    
    # Add tools if available
    if actor.tools:
        params["tools"] = [_tool_to_openai(t) for t in actor.tools]
        params["tool_choice"] = "auto"
    
    # Add extra_params if available (for custom endpoints like amplified sampling)  
    if hasattr(actor.endpoint, 'extra_params') and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)

    api_base = actor.endpoint.api_base 
    if not api_base.endswith('/chat/completions'):
        if api_base.endswith('/v1'):
            api_base = api_base.rstrip('/') + '/chat/completions'
        else:
            api_base = api_base.rstrip('/') + '/v1/chat/completions'
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    completion = None
    # Add debugging before HTTP call
    print(f"🔥 Making HTTP POST to: {api_base}")
    print(f"🔥 Headers: {headers}")
    print(f"🔥 Request params keys: {list(params.keys())}")
    import sys; sys.stdout.flush()  # Force immediate output
    
    # Stream the response with timeout
    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Stream the response
        for attempt in range(1, max_api_retries + 1):
            try:
                print(f"🔥 Attempt {attempt}: Sending HTTP request...")
                sys.stdout.flush()
                async with session.post(api_base, json=params, headers=headers) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        print(f"❌ Server returned {response.status}: {error_body}")
                        
                        # Parse and provide helpful guidance for common errors
                        if "maximum context length" in error_body.lower():
                            print(f"💡 CONTEXT LENGTH ERROR DETECTED:")
                            print(f"   • This is NOT a server startup failure - server is working correctly")
                            print(f"   • Your max_tokens ({params.get('max_tokens')}) exceeds server's limit")
                            print(f"   • FIX: Reduce max_tokens to a smaller value (try {params.get('max_tokens', 8192) // 2})")
                            print(f"   • OR: Redeploy server with larger --max-model-len")
                            print(f"🛑 Stopping retries - context length errors cannot be fixed by retrying")
                            raise Exception(f"Context length exceeded: {error_body}")
                        elif "not a valid parameter" in error_body.lower():
                            print(f"💡 PARAMETER ERROR DETECTED:")
                            print(f"   • Server doesn't support one of your parameters")
                            print(f"   • Your parameters: {list(params.keys())}")
                            print(f"   • Try removing 'logprobs' or 'echo' parameters")
                        
                        response.raise_for_status()
                    else:
                        completion = await response.json()
                        break
            except Exception as e:
                print(f"llm_call failed with {e}")
                if attempt < max_api_retries:  # back-off & retry
                    if verbose(1): 
                        print(f"timed out request, retrying with {backoff_base * 2 ** (attempt - 1)} seconds")
                    await asyncio.sleep(backoff_base * 2 ** (attempt - 1))
    # TODO: handle api call misses
    assert completion
    message = completion["choices"][0]["message"]
    raw_tool_calls = message.get("tool_calls")

    if raw_tool_calls:
        conformed_tool_calls = []
        for tc in raw_tool_calls:
            conformed_tool_calls.append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "args": json.loads(tc["function"]["arguments"]),
            })
        
        message["tool_calls"] = conformed_tool_calls
    print(completion)
    completion = from_dict(ChatCompletion, completion)
    # Skip logprobs assertion for amplified sampling server - it doesn't provide prompt_logprobs
    if completion.prompt_logprobs:
        print(f"🔥 prompt_logprobs available: {len(completion.prompt_logprobs)} items")
    else:
        print("🔥 No prompt_logprobs (expected for amplified sampling server)")
    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message

    # Create new trajectory
    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion]
    )
    
    return replace(actor, trajectory=new_trajectory)

async def rollout_moonshot(actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]]) -> Actor:
    """Make LLM API call with streaming and return updated actor with new trajectory"""
    client = AsyncOpenAI(
        api_key = actor.endpoint.api_key,
        base_url= actor.endpoint.api_base
    )
    
    assert os.getenv("MOONSHOT_API_KEY")
    messages = [_message_to_openai(m) for m in actor.trajectory.messages]
    params = {
        "model": actor.endpoint.model,
        "messages": messages,
        "temperature": actor.endpoint.temperature,
        "stream": True,
    }

    if actor.endpoint.max_completion_tokens is not None:
        params["max_completion_tokens"] = actor.endpoint.max_completion_tokens
    else:
        params["max_tokens"] = actor.endpoint.max_tokens

    if actor.tools:
        params["tools"] = [_tool_to_openai(t) for t in actor.tools]
        params["tool_choice"] = "auto"
    
    try:
        stream = await client.chat.completions.create(**params)
    except Exception as e:
        print("Exception:", str(e), json.dumps(params, indent=2, default=str))
    
    completion = await aggregate_stream(stream, on_chunk)
    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message

    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion]
    )
    return replace(actor, trajectory=new_trajectory)



async def rollout_openai(actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]]) -> Actor:
    """Make LLM API call with streaming and return updated actor with new trajectory"""
    client = AsyncOpenAI(api_key=actor.endpoint.api_key, base_url=actor.endpoint.api_base)
    
    # Convert messages to OpenAI format
    messages = [_message_to_openai(m) for m in actor.trajectory.messages]
    
    # Prepare API call parameters
    params = {
        "model": actor.endpoint.model,
        "messages": messages,
        "temperature": actor.endpoint.temperature,
        "stream": True,
    }

    if actor.endpoint.max_completion_tokens is not None:
        params["max_completion_tokens"] = actor.endpoint.max_completion_tokens
    else:
        params["max_tokens"] = actor.endpoint.max_tokens

    
    # Add tools if available
    if actor.tools:
        params["tools"] = [_tool_to_openai(t) for t in actor.tools]
        params["tool_choice"] = "auto"

    # TODO: Add reasoning_effort for o-series models
    if actor.endpoint.reasoning_effort is not None:
        params["reasoning_effort"] = actor.endpoint.reasoning_effort
    
    # Add extra_params if available (for custom endpoints like amplified sampling)
    if hasattr(actor.endpoint, 'extra_params') and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)
    
    try:
        # Stream the response
        stream = await client.chat.completions.create(**params)
        completion = await aggregate_stream(stream, on_chunk)
        
    except Exception as e:
        # Dump the exact request on error
        print("\n" + "="*80)
        print("ERROR: Failed to call OpenAI API")
        print("="*80)
        print("Exception:", str(e))
        print("\nExact request sent to OpenAI:")
        print(json.dumps(params, indent=2, default=str))
        print("="*80)
        raise  # Re-raise the exception
    
    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message

    # Create new trajectory
    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion]
    )
    
    return replace(actor, trajectory=new_trajectory)

async def rollout_anthropic(actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]], user_message_for_thinking: Optional[str] = None, turn_idx: int = 0, inline_thinking: Optional[str] = None) -> Actor:
    """Make Anthropic API call with streaming using SDK's stream() method"""
    # Only pass base_url if it's not empty, otherwise use default
    client_kwargs = {"api_key": actor.endpoint.api_key}
    if actor.endpoint.api_base:
        client_kwargs["base_url"] = actor.endpoint.api_base
    client = AsyncAnthropic(**client_kwargs)
    
    # Apply truncation to messages before sending to API (keeps full trajectory intact)
    # truncated_trajectory = truncate_trajectory_simple(actor.trajectory)
    truncated_trajectory = actor.trajectory

    # Handle system message and convert messages
    system_prompt = None
    messages = []
    
    for m in truncated_trajectory.messages:
        if m.role == "system":
            system_prompt = m.content
        elif m.role == "tool":
            # Anthropic expects tool results as user messages
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content
                }]
            })
        else:
            messages.append(_message_to_anthropic(m, inline_thinking))
    
    # Inject thinking message after turn 1 if provided
    if user_message_for_thinking and turn_idx > 0:
        messages.append({"role": "user", "content": user_message_for_thinking})
    
    # Ensure first message is from user
    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "Begin."})
    
    # Apply cache control to messages for prompt caching
    messages_with_cache = add_cache_control_to_last_content(messages)
    
    # Prepare parameters
    params: Dict[str, Any] = {
        "max_tokens": actor.endpoint.max_tokens,
        "messages": messages_with_cache,
        "model": actor.endpoint.model,
        "temperature": actor.endpoint.temperature,
    }
    
    if system_prompt:
        params["system"] = system_prompt
    
    if actor.tools:
        params["tools"] = [_tool_to_anthropic(t) for t in actor.tools]

    if actor.endpoint.thinking is not None:
        params["thinking"] = actor.endpoint.thinking  # Uses the thinking config from Endpoint

    
    # Retry logic for any API errors
    import asyncio
    max_retries = 10
    base_delay = 2  # seconds
    completion = None
    
    for attempt in range(max_retries + 1):
        try:
            # Use SDK's stream() method with prompt caching headers
            async with client.messages.stream(  # type: ignore[missing-argument]  # TODO: Fix Anthropic SDK strict typing with **params
                **params,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            ) as stream:
                completion = await aggregate_anthropic_stream(stream, on_chunk)
                break  # Success, exit retry loop
                
        except Exception as e:
            print(f"🔄 Anthropic API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
            
            if attempt < max_retries:
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                print(f"   Retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            else:
                # Max retries exceeded
                print("\n" + "="*80)
                print("ERROR: Failed to call Anthropic API after all retries")
                print("="*80)
                print("Final Exception:", str(e))
                print("\nExact request sent to Anthropic:")
                print(json.dumps(params, indent=2, default=str))
                print("="*80)
                raise
    
    if completion is None:
        raise Exception("Failed to get completion after all retries")
    
    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message
    
    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion]
    )
    
    
    await client.close()
    return replace(actor, trajectory=new_trajectory)

def _apply_inline_thinking_template(thinking_content: str, content: str, inline_thinking: str) -> str:
    """Apply inline thinking template to combine thinking and content (anthropic only)"""
    assert "{thinking}" in inline_thinking, "inline_thinking template must contain {thinking} field"
    assert "{content}" in inline_thinking, "inline_thinking template must contain {content} field"
    return inline_thinking.format(thinking=thinking_content, content=content)

def _message_to_anthropic(m: Message, inline_thinking: Optional[str] = None) -> Dict[str, Any]:
    """Convert our Message to Anthropic format"""
    msg: Dict[str, Any] = {"role": m.role}
    
    # Handle assistant messages with tool calls
    if m.role == "assistant" and m.tool_calls:
        content_blocks = []
        
        # Handle inline thinking template if provided (anthropic only)
        if (inline_thinking and hasattr(m, 'thinking_content') and m.thinking_content and m.content):
            # Use template to combine thinking and content
            combined_text = _apply_inline_thinking_template(m.thinking_content, m.content, inline_thinking)
            content_blocks.append({
                "type": "text",
                "text": combined_text
            })
        else:
            # Original behavior: Add thinking block first if it exists
            if hasattr(m, 'thinking_content') and m.thinking_content:
                thinking_block = {
                    "type": "thinking",
                    "thinking": m.thinking_content
                }
                # Add signature if available
                if hasattr(m, 'thinking_signature') and m.thinking_signature:
                    thinking_block["signature"] = m.thinking_signature
                content_blocks.append(thinking_block)
            
            # Add text content if present
            if m.content:
                content_blocks.append({
                    "type": "text",
                    "text": m.content
                })
        
        # Add tool use blocks
        for tc in m.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.args
            })
        
        msg["content"] = content_blocks
    else:
        # Simple text content - also check for inline thinking template
        if (inline_thinking and hasattr(m, 'thinking_content') and m.thinking_content and m.content):
            # Use template to combine thinking and content
            combined_text = _apply_inline_thinking_template(m.thinking_content, m.content, inline_thinking)
            msg["content"] = combined_text
        else:
            # Original behavior
            msg["content"] = m.content if m.content else [{
                    "type": "text",
                    "text": "[Response truncated due to time limit]"
                }]
    
    return msg



def _tool_to_anthropic(tool: Tool) -> Dict[str, Any]:
    """Convert our Tool to Anthropic format
    
    Key differences from OpenAI:
    - Uses 'input_schema' instead of 'parameters'
    - No 'type' field at top level
    - Required fields are part of the schema
    """
    return {
        "name": tool.function.name,
        "description": tool.function.description,
        "input_schema": {
            "type": tool.function.parameters.type,
            "properties": tool.function.parameters.properties,
            "required": tool.function.required
        }
    }

async def aggregate_anthropic_stream(
    stream,  # Anthropic's MessageStream object
    on_chunk: Callable[[StreamChunk], Awaitable[None]]
) -> ChatCompletion:
    """Aggregate Anthropic SDK stream into our ChatCompletion format"""
    
    accumulated_content = ""
    thinking_content = ""
    thinking_signature = None
    tool_calls = []
    message_id = None
    created_at = int(time.time())
    finish_reason = "stop"
    
    # For accumulating tool input JSON
    tool_json_accumulator = {}  # index -> accumulated json string
    tool_metadata = {}  # index -> {id, name}
    
    # Process events from the SDK stream
    async for event in stream:
        event_type = event.type
        
        if event_type == "message_start":
            message_id = event.message.id
            created_at = int(time.time())  # SDK doesn't expose created timestamp
            
        elif event_type == "content_block_start":
            block = event.content_block
            index = event.index
            
            if block.type == "text":
                # Text block starting
                pass
                
            elif block.type == "tool_use":
                # Tool use block starting
                tool_metadata[index] = {
                    "id": block.id,
                    "name": block.name
                }
                tool_json_accumulator[index] = ""
                
            elif block.type == "thinking":
                # Extended thinking block - we'll skip for now
                # You could emit thinking events if needed
                pass

            # elif block.type == "server_tool_use":
            #     # TODO: Handle Anthropic's built-in tools
            #     tool_metadata[index] = {
            #         "id": block.id,
            #         "name": block.name,
            #         "is_server_tool": True
            #     }
            # elif block.type == "web_search_tool_result":
            #     # TODO: Handle web search results
            #     await on_chunk(StreamChunk("web_search_result", {
            #         "tool_use_id": block.tool_use_id,
            #         "content": block.content
            #     }))

                
        elif event_type == "content_block_delta":
            delta = event.delta
            index = event.index
            
            if delta.type == "text_delta":
                text = delta.text
                accumulated_content += text
                await on_chunk(StreamChunk("token", {"text": text}))
                
            elif delta.type == "input_json_delta":
                # Accumulate partial JSON for tool input
                partial = delta.partial_json
                tool_json_accumulator[index] += partial
                
                # Emit partial tool call event
                if index in tool_metadata:
                    await on_chunk(StreamChunk("tool_call_partial", {
                        "index": index,
                        "id": tool_metadata[index]["id"],
                        "name": tool_metadata[index]["name"],
                        "partial_json": partial,
                        "accumulated_json": tool_json_accumulator[index]
                    }))
                    
            elif delta.type == "thinking_delta":
                # Could emit thinking events
                thinking_text = delta.thinking
                thinking_content += thinking_text  # TODO: Accumulate thinking
                await on_chunk(StreamChunk("thinking", {"text": thinking_text}))

            elif delta.type == "signature_delta":
                signature = delta.signature
                thinking_signature = signature  # TODO: Store signature
                await on_chunk(StreamChunk("thinking_signature", {
                    "signature": signature,
                    "index": index
                }))

                
        elif event_type == "content_block_stop":
            index = event.index
            
            # If this was a tool use block, parse the complete JSON
            if index in tool_metadata:
                raw = tool_json_accumulator.get(index, "")  # may be ""
                try:
                    # IMPORTANT GOTCHA: Tools with no arguments (like "clear") may generate
                    # empty JSON strings "" instead of empty objects "{}". json.loads("")
                    # fails, but json.loads("{}") works. Default to {} for empty strings.
                    tool_input = json.loads(raw) if raw else {}
                    
                    tool_call = ToolCall(
                        id=tool_metadata[index]["id"],
                        name=tool_metadata[index]["name"],
                        args=tool_input
                    )
                    tool_calls.append(tool_call)
                    
                    # Emit tool call complete event
                    await on_chunk(StreamChunk("tool_call_complete", {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "args": tool_input,
                        "index": index
                    }))
                    
                except json.JSONDecodeError as e:
                    # Emit error for malformed JSON
                    await on_chunk(StreamChunk("tool_call_error", {
                        "index": index,
                        "id": tool_metadata[index]["id"],
                        "name": tool_metadata[index]["name"],
                        "error": f"Invalid JSON: {str(e)}",
                        "partial_json": tool_json_accumulator[index]
                    }))
                    
        elif event_type == "message_delta":
            # Final message metadata
            if hasattr(event, 'delta') and hasattr(event.delta, 'stop_reason'):
                finish_reason = event.delta.stop_reason or "stop"
                
        elif event_type == "message_stop":
            # Message complete
            pass
            
        elif event_type == "ping":
            # Keep-alive ping
            await on_chunk(StreamChunk("ping", {}))
            
        elif event_type == "error":
            # Handle streaming errors
            error_data = {"type": event.error.type, "message": event.error.message}
            await on_chunk(StreamChunk("error", error_data))
            raise Exception(f"Anthropic stream error: {error_data}")
    
    # Create final message
    final_message = Message(
        role="assistant",
        content=accumulated_content if accumulated_content else "",
        thinking_content=thinking_content if thinking_content else None,  # TODO: Include thinking,
        thinking_signature=thinking_signature,
        tool_calls=tool_calls
    )
    
    # Emit completion event
    await on_chunk(StreamChunk("assistant_complete", {
        "content": accumulated_content,
        "tool_call_count": len(tool_calls),
        "finish_reason": finish_reason
    }))
    
    # Get final message from stream for usage stats
    final_anthropic_message = await stream.get_final_message()
    
    # Extract usage information
    usage = Usage(
        prompt_tokens=final_anthropic_message.usage.input_tokens,
        completion_tokens=final_anthropic_message.usage.output_tokens,
        total_tokens=final_anthropic_message.usage.input_tokens + final_anthropic_message.usage.output_tokens
    )
    
    # Create completion object
    completion = ChatCompletion(
        id=message_id or f"msg_{int(time.time() * 1000)}",
        object="chat.completion",
        created=created_at,
        model="",  # Will be set by caller
        usage=usage,
        choices=[Choice(0, final_message, finish_reason)]
    )
    
    return completion

async def confirm_tool_with_feedback(tc: ToolCall, state: AgentState, run_config: 'RunConfig') -> Tuple[AgentState, ToolConfirmResult]:
    """Confirm tool execution, returning state and confirmation result"""
    if not state.environment.requires_confirmation(tc):
        return state, ToolConfirmResult(proceed=True)

    print(f"\n▶️ Execute `{tc.name}({tc.args})`?")
    print("  [y] Yes, execute")
    print("  [n] No, provide feedback")
    print("  [s] No, skip silently")
    
    resp = input("Choice: ").strip().lower()
    
    if resp == 'y':
        return state, ToolConfirmResult(proceed=True)
    
    elif resp == 'n':
        feedback = input("Why not? Provide guidance: \n").strip()
        return state, ToolConfirmResult(
            proceed=False,
            tool_result=ToolResult(
                call_id=tc.id,
                ok=False,
                error="Rejected by user"
            ),
            user_message=feedback
        )
    
    else:  # Skip silently
        return state, ToolConfirmResult(
            proceed=False,
            tool_result=ToolResult(
                call_id=tc.id,
                ok=False,
                error="Skipped by user"
            )
        )

def handle_tool_error(result: ToolResult, state: AgentState) -> AgentState:
    """Handle tool execution errors - currently a no-op"""
    return state

def inject_turn_warning(state: AgentState) -> AgentState:
    """Warn when 2 turns remaining"""
    turns_left = state.max_turns - state.turn_idx
    if turns_left == 2:
        warning = Message(
            role="user",
            content="⚠️ You have 2 turns remaining. Please complete your task quickly.",
            tool_calls=[]
        )
        new_trajectory = replace(
            state.actor.trajectory,
            messages=state.actor.trajectory.messages + [warning]
        )
        return replace(state, actor=replace(state.actor, trajectory=new_trajectory))
    return state

def handle_stop_max_turns(state: AgentState) -> AgentState:
    """Stop when max turns reached"""
    if state.turn_idx >= state.max_turns:
        return replace(state, stop=StopReason.MAX_TURNS)
    return state

async def inject_tool_reminder(state: AgentState, run_config: 'RunConfig') -> AgentState:
    """Remind the agent to use tools"""
    reminder = Message(
        role="user",
        content="Please use the available tools to complete the task. What calculation would you like to perform?",
        tool_calls=[]
    )
    new_trajectory = replace(
        state.actor.trajectory,
        messages=state.actor.trajectory.messages + [reminder]
    )
    return replace(state, actor=replace(state.actor, trajectory=new_trajectory))


FullAuto = RunConfig(
    on_chunk=stdout_handler,
    confirm_tool=confirm_tool_with_feedback, #type: ignore
    handle_tool_error=handle_tool_error,
    on_step_start=inject_turn_warning,
    handle_stop=handle_stop_max_turns,       # Add this
    handle_no_tool=inject_tool_reminder,
)

async def rollout(actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]]=stdout_handler, 
                  user_message_for_thinking: Optional[str] = None, turn_idx: int = 0, inline_thinking: Optional[str] = None) -> Actor:
    provider = actor.endpoint.provider
    if provider == "openai":
        new_actor = await rollout_openai(actor, on_chunk)
    elif provider == "moonshot":
        new_actor = await rollout_moonshot(actor, on_chunk)
    elif provider == "vllm":
        new_actor = await rollout_vllm(actor, on_chunk)
    elif provider == "anthropic":
        new_actor = await rollout_anthropic(actor, on_chunk, user_message_for_thinking, turn_idx, inline_thinking)
    else:
        print(f"Invalid provider {actor.endpoint.provider})")
        sys.exit(0)
    return new_actor

async def run_agent_step(state: AgentState, rcfg: RunConfig) -> AgentState:
    """Execute one complete turn: LLM call → ALL tool executions → next turn.
    
    Turn atomicity: Execute ALL tools before giving control back to LLM.
    This simplifies reasoning but prevents early stopping or parallel execution.
    """

    state = rcfg.handle_stop(state)
    if state.stop:
        return state

    # If we have pending tools, resume processing them
    if state.pending_tool_calls:
        return await process_pending_tools(state, rcfg)

    
    state = rcfg.on_step_start(state)
    
    # Otherwise, do a new rollout
    available_tools = state.environment.get_tools()
    updated_actor = replace(state.actor, tools=available_tools)
    
    # Make LLM call
    next_actor = await rollout(updated_actor, rcfg.on_chunk, rcfg.user_message_for_thinking, state.turn_idx, rcfg.inline_thinking)
    
    # Extract tool calls
    last_message = next_actor.trajectory.messages[-1]
    tool_calls = last_message.tool_calls if last_message.tool_calls else []
    
    # Update state with new actor AND pending tools
    current_state = replace(
        state, 
        actor=next_actor,
        pending_tool_calls=tool_calls,
        next_tool_idx=0
    )
    
    # If no tools, we're done with this turn
    if not tool_calls:
        current_state = await rcfg.handle_no_tool(current_state, rcfg)
        # Check if handler added a stop reason
        if current_state.stop:
            return current_state
        # Otherwise increment turn and continue
        return replace(current_state, 
                      turn_idx=current_state.turn_idx + 1,
                      pending_tool_calls=[])

    
    # Process the pending tools
    return await process_pending_tools(current_state, rcfg)

# TODO: Checkpoint granularity for multi-tool calls
# 
# Current behavior: When LLM returns multiple tool calls in one response,
# we execute ALL tools before creating a checkpoint. This means:
#   - add(10) → multiply(3) → divide(5) all execute, THEN checkpoint
#   - If crash occurs during multiply(), we restart from turn beginning
# 
# This is usually fine because tool execution is fast, but consider finer
# checkpointing if:
#   - Tools make slow external API calls  
#   - Tools have expensive side effects (can't safely re-run)
#   - Running very long tool chains (10+ tools per turn)
#
# Implementation approach: Modify process_pending_tools to yield intermediate
# states after each tool, then checkpoint each yielded state in run_agent.
# See next_tool_idx which already tracks progress within a tool batch.

async def process_pending_tools(state: AgentState, rcfg: RunConfig) -> AgentState:
    """Resume processing tools from next_tool_idx"""
    current_state = state
    
    # SERIALIZE environment state before tool processing
    env_data = await current_state.environment.serialize()
    
    for i in range(state.next_tool_idx, len(state.pending_tool_calls)):
        tool_call = state.pending_tool_calls[i]
        current_state = replace(current_state, next_tool_idx=i)
        
        # Get confirmation result
        current_state, confirm_result = await rcfg.confirm_tool(tool_call, current_state, rcfg)
        
        if confirm_result.proceed:
            # DESERIALIZE fresh environment for each tool call
            fresh_env = await current_state.environment.__class__.deserialize(env_data)
            
            # Execute tool on fresh environment
            tool_result = await fresh_env.exec_tool(tool_call, current_state, rcfg, None)
            
            if tool_result.ok:
                # SERIALIZE the updated environment state
                env_data = await fresh_env.serialize()
                
                # DESERIALIZE again to update current_state
                current_state = replace(
                    current_state,
                    environment=await current_state.environment.__class__.deserialize(env_data)
                )
        else:
            # Use the provided tool result
            tool_result = confirm_result.tool_result
            # TODO: handle None on tool results
        
        # Emit tool result
        assert tool_result
        await rcfg.on_chunk(StreamChunk("tool_result", {
            "call_id": tool_call.id,
            "ok": tool_result.ok,
            "content": tool_result.content,
            "error": tool_result.error
        }))
        
        # Add tool result message
        result_message = Message(
            role="tool",
            content=tool_result.content if tool_result.ok else f"Error: {tool_result.error}",
            tool_call_id=tool_call.id
        )
        
        messages_to_add = [result_message]
        
        # Add user feedback if provided
        if confirm_result.user_message:
            user_msg = Message(
                role="user",
                content=confirm_result.user_message,
                tool_calls=[]
            )
            messages_to_add.append(user_msg)
        
        # Update trajectory with all messages
        updated_trajectory = replace(
            current_state.actor.trajectory,
            messages=current_state.actor.trajectory.messages + messages_to_add
        )
        current_state = replace(
            current_state,
            actor=replace(current_state.actor, trajectory=updated_trajectory)
        )
        
        # Handle tool errors
        current_state = rcfg.handle_tool_error(tool_result, current_state)
        
        # Fire tool_complete event after each tool (if checkpoint strategy supports it)
        if hasattr(rcfg, 'checkpoint_store') and rcfg.checkpoint_store:
            await handle_checkpoint_event(current_state, "tool_complete", rcfg)
        
        # Check if tool requested agent to stop
        if tool_result.stop_reason:
            current_state = replace(current_state, stop=tool_result.stop_reason)
            # Break out of tool processing loop - agent will stop after this turn
            break
    
    # All tools processed
    #print(f"[DEBUG] process_pending_tools done - incrementing turn from {current_state.turn_idx} to {current_state.turn_idx + 1}")
    #print(f"[DEBUG] Stop reason: {current_state.stop}")
    return replace(
        current_state, 
        turn_idx=current_state.turn_idx + 1,
        pending_tool_calls=[],
        next_tool_idx=0
    )


async def run_agent(
    state: AgentState, 
    run_config: RunConfig,
    session_id: Optional[str] = None
) -> List[AgentState]:
    """Run agent until stop condition, checkpointing each state"""
    if run_config.checkpoint_store and not session_id:
        session_id = f"session_{int(time.time() * 1000)}"  # ms timestamp
    
    states = [state]
    current_state = state
    
    try:
        # Debug: Print environment type and available tools
        #print(f"[DEBUG] run_agent called with environment type: {type(current_state.environment).__name__}")
        if hasattr(current_state.environment, 'get_tools'):
            [tool.function.name for tool in current_state.environment.get_tools()]
            #print(f"[DEBUG] Available tools: {tools}")
        
        # Save initial state
        await handle_checkpoint_event(state, "turn_start", run_config, session_id)
        
        while not current_state.stop and current_state.turn_idx < current_state.max_turns:
            #print(f"[DEBUG] Agent loop - Turn {current_state.turn_idx}, Stop: {current_state.stop}, Max turns: {current_state.max_turns}")
            next_state = await run_agent_step(current_state, run_config)
            #print(f"[DEBUG] After step - Turn {next_state.turn_idx}, Stop: {next_state.stop}")
            current_state = next_state
            states.append(current_state)
            
            # Checkpoint after each state transition
            await handle_checkpoint_event(current_state, "turn_end", run_config, session_id)
            
        
        # Set stop reason if we hit max turns
        if current_state.turn_idx >= current_state.max_turns:
            current_state = replace(current_state, stop=StopReason.MAX_TURNS)
            states[-1] = current_state
            
            # Save final state
            await handle_checkpoint_event(current_state, "final", run_config, session_id)
        
        return states
    finally:
        # Placeholder for any cleanup if needed in the future
        pass
