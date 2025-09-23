"""Provider-specific rollout helpers for the agents module."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time
from dataclasses import replace
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

import aiohttp
from anthropic import AsyncAnthropic
from dacite import from_dict
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam

from .dtypes import (
    Actor,
    ChatCompletion,
    Choice,
    Message,
    StreamChunk,
    Tool,
    ToolCall,
    Usage,
)


def add_cache_control_to_last_content(
    messages, cache_control={"type": "ephemeral"}, max_cache_controls: int = 4
):
    """Adds cache control metadata to the final content block if possible."""
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


def _message_to_openai(m: Message) -> ChatCompletionMessageParam:
    """Convert framework `Message` objects to the OpenAI SDK schema."""
    msg: Dict[str, Any] = {"role": m.role}

    if m.content is not None:
        msg["content"] = m.content

    if m.tool_calls and m.role == "assistant":
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args),
                },
            }
            for tc in m.tool_calls
        ]

    if m.role == "tool":
        msg["tool_call_id"] = m.tool_call_id
        msg["content"] = m.content

    return msg


def _tool_to_openai(tool: Tool) -> Dict[str, Any]:
    """Convert a framework `Tool` definition into OpenAI's schema."""
    return {
        "type": tool.type,
        "function": {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": {
                "type": tool.function.parameters.type,
                "properties": tool.function.parameters.properties,
                "required": tool.function.required,
            },
        },
    }


def _parse_usage(u: CompletionUsage) -> Usage:
    return Usage(u.prompt_tokens, u.completion_tokens, u.total_tokens)


def _parse_completion(resp: Any) -> ChatCompletion:
    """Convert an OpenAI SDK response into the framework `ChatCompletion`."""
    choices = []
    for c in resp.choices:
        tool_calls = []
        if hasattr(c.message, "tool_calls") and c.message.tool_calls:
            for tc in c.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=json.loads(tc.function.arguments) if tc.function.arguments else {},
                    )
                )

        msg = Message(
            role=c.message.role,
            content=c.message.content,
            tool_calls=tool_calls,
        )
        choices.append(Choice(c.index, msg, c.finish_reason))

    return ChatCompletion(
        id=resp.id,
        object=resp.object,
        created=resp.created,
        model=resp.model,
        usage=_parse_usage(resp.usage),
        choices=choices,
    )


async def aggregate_stream(
    stream: AsyncIterator,
    on_chunk: Callable[[StreamChunk], Awaitable[None]],
) -> ChatCompletion:
    """Aggregate streaming chunks into a complete `ChatCompletion`."""
    accumulated_content = ""
    finish_reason = None
    response_id = None
    created = None

    call_buf: Dict[int, Dict[str, Any]] = {}
    next_auto_index = 0

    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        if response_id is None:
            response_id = chunk.id
            created = chunk.created

        if delta.content:
            accumulated_content += delta.content
            await on_chunk(StreamChunk("token", {"text": delta.content}))

        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                idx = tool_call.index
                if idx is None:
                    idx = next_auto_index
                    next_auto_index += 1

                if idx not in call_buf:
                    call_buf[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }

                if tool_call.id:
                    call_buf[idx]["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        call_buf[idx]["function"]["name"] = tool_call.function.name
                    if tool_call.function.arguments:
                        call_buf[idx]["function"]["arguments"] += (
                            tool_call.function.arguments
                        )

            await on_chunk(
                StreamChunk("tool_call_partial", {"calls": list(call_buf.values())})
            )

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    tool_calls: List[ToolCall] = []
    for idx, tc in sorted(call_buf.items()):
        if tc["function"]["name"]:
            try:
                args = (
                    json.loads(tc["function"]["arguments"])
                    if tc["function"]["arguments"]
                    else {}
                )
            except json.JSONDecodeError:
                await on_chunk(
                    StreamChunk(
                        "tool_call_error",
                        {
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "error": (
                                "Invalid JSON arguments: "
                                f"{tc['function']['arguments']}"
                            ),
                            "index": idx,
                        },
                    )
                )
                continue

            await on_chunk(
                StreamChunk(
                    "tool_call_complete",
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": args,
                        "raw_arguments": tc["function"]["arguments"],
                        "index": idx,
                    },
                )
            )

            tool_calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], args=args))

    final_message = Message(role="assistant", content=accumulated_content or "", tool_calls=tool_calls)

    await on_chunk(
        StreamChunk(
            "assistant_complete",
            {
                "content": accumulated_content,
                "tool_call_count": len(tool_calls),
                "finish_reason": finish_reason,
            },
        )
    )

    completion = ChatCompletion(
        id=response_id or "unknown",
        object="chat.completion",
        created=created or 0,
        model="",
        usage=Usage(0, 0, 0),
        choices=[Choice(0, final_message, finish_reason or "stop")],
    )

    return completion


def verbose(level: int) -> bool:
    """Simple verbose checker - just return True for now."""
    return True


async def rollout_vllm(
    actor: Actor,
    on_chunk: Callable[[StreamChunk], Awaitable[None]],
    max_api_retries: int = 16,
    backoff_base: int = 4,
) -> Actor:
    """Invoke a vLLM server and return the updated actor."""

    messages = [_message_to_openai(m) for m in actor.trajectory.messages]

    params = {
        "model": actor.endpoint.model,
        "messages": messages,
        "max_tokens": actor.endpoint.max_tokens,
        "temperature": actor.endpoint.temperature,
        "stream": False,
        "logprobs": True,
        "echo": True,
    }

    if actor.tools:
        params["tools"] = [_tool_to_openai(t) for t in actor.tools]
        params["tool_choice"] = "auto"

    if hasattr(actor.endpoint, "extra_params") and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)

    api_base = actor.endpoint.api_base
    if not api_base.endswith("/chat/completions"):
        if api_base.endswith("/v1"):
            api_base = api_base.rstrip("/") + "/chat/completions"
        else:
            api_base = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    completion = None
    print(f"🔥 Making HTTP POST to: {api_base}")
    print(f"🔥 Headers: {headers}")
    print(f"🔥 Request params keys: {list(params.keys())}")
    sys.stdout.flush()

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(1, max_api_retries + 1):
            try:
                print(f"🔥 Attempt {attempt}: Sending HTTP request...")
                sys.stdout.flush()
                async with session.post(api_base, json=params, headers=headers) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        print(f"❌ Server returned {response.status}: {error_body}")

                        if "maximum context length" in error_body.lower():
                            print("💡 CONTEXT LENGTH ERROR DETECTED:")
                            print("   • This is NOT a server startup failure - server is working correctly")
                            print(
                                f"   • Your max_tokens ({params.get('max_tokens')}) exceeds server's limit"
                            )
                            print(
                                "   • FIX: Reduce max_tokens to a smaller value (try "
                                f"{params.get('max_tokens', 8192) // 2})"
                            )
                            print(
                                "   • OR: Redeploy server with larger --max-model-len"
                            )
                            print(
                                "🛑 Stopping retries - context length errors cannot be fixed by retrying"
                            )
                            raise Exception(f"Context length exceeded: {error_body}")
                        elif "not a valid parameter" in error_body.lower():
                            print("💡 PARAMETER ERROR DETECTED:")
                            print(
                                "   • Server doesn't support one of your parameters"
                            )
                            print(
                                f"   • Your parameters: {list(params.keys())}"
                            )
                            print(
                                "   • Try removing 'logprobs' or 'echo' parameters"
                            )

                        response.raise_for_status()
                    else:
                        completion = await response.json()
                        break
            except Exception as e:
                print(f"llm_call failed with {e}")
                if attempt < max_api_retries:
                    if verbose(1):
                        print(
                            "timed out request, retrying with "
                            f"{backoff_base * 2 ** (attempt - 1)} seconds"
                        )
                    await asyncio.sleep(backoff_base * 2 ** (attempt - 1))
    assert completion
    message = completion["choices"][0]["message"]
    raw_tool_calls = message.get("tool_calls")

    if raw_tool_calls:
        conformed_tool_calls = []
        for tc in raw_tool_calls:
            conformed_tool_calls.append(
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {},
                }
            )

        message["tool_calls"] = conformed_tool_calls
    print(completion)
    completion = from_dict(ChatCompletion, completion)
    if completion.prompt_logprobs:
        print(
            f"🔥 prompt_logprobs available: {len(completion.prompt_logprobs)} items"
        )
    else:
        print("🔥 No prompt_logprobs (expected for amplified sampling server)")
    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message

    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion],
    )

    return replace(actor, trajectory=new_trajectory)


async def rollout_moonshot(
    actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]]
) -> Actor:
    """Make an API call to Moonshot with streaming enabled."""
    client = AsyncOpenAI(
        api_key=actor.endpoint.api_key,
        base_url=actor.endpoint.api_base,
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

    if hasattr(actor.endpoint, "extra_params") and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)

    stream = await client.chat.completions.create(**params)
    completion = await aggregate_stream(stream, on_chunk)
    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message

    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion],
    )

    return replace(actor, trajectory=new_trajectory)


async def rollout_openai(
    actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]]
) -> Actor:
    """Make an OpenAI API call with streaming and update the actor."""
    client = AsyncOpenAI(
        api_key=actor.endpoint.api_key,
        base_url=actor.endpoint.api_base,
    )

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

    if actor.endpoint.reasoning_effort is not None:
        params["reasoning_effort"] = actor.endpoint.reasoning_effort

    if hasattr(actor.endpoint, "extra_params") and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)

    try:
        stream = await client.chat.completions.create(**params)
        completion = await aggregate_stream(stream, on_chunk)

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: Failed to call OpenAI API")
        print("=" * 80)
        print("Exception:", str(e))
        print("\nExact request sent to OpenAI:")
        print(json.dumps(params, indent=2, default=str))
        print("=" * 80)
        raise

    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message

    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion],
    )

    return replace(actor, trajectory=new_trajectory)


def _apply_inline_thinking_template(
    thinking_content: str, content: str, inline_thinking: str
) -> str:
    """Apply inline thinking template to combine thinking and content."""
    assert "{thinking}" in inline_thinking
    assert "{content}" in inline_thinking
    return inline_thinking.format(thinking=thinking_content, content=content)


def _message_to_anthropic(
    m: Message, inline_thinking: Optional[str] = None
) -> Dict[str, Any]:
    """Convert a `Message` into Anthropic's streaming-compatible schema."""
    msg: Dict[str, Any] = {"role": m.role}

    if m.role == "assistant" and m.tool_calls:
        content_blocks = []

        if (
            inline_thinking
            and hasattr(m, "thinking_content")
            and m.thinking_content
            and m.content
        ):
            combined_text = _apply_inline_thinking_template(
                m.thinking_content, m.content, inline_thinking
            )
            content_blocks.append({"type": "text", "text": combined_text})
        else:
            if hasattr(m, "thinking_content") and m.thinking_content:
                thinking_block = {"type": "thinking", "thinking": m.thinking_content}
                if hasattr(m, "thinking_signature") and m.thinking_signature:
                    thinking_block["signature"] = m.thinking_signature
                content_blocks.append(thinking_block)

            if m.content:
                content_blocks.append({"type": "text", "text": m.content})

        for tc in m.tool_calls:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.args,
                }
            )

        msg["content"] = content_blocks
    else:
        if (
            inline_thinking
            and hasattr(m, "thinking_content")
            and m.thinking_content
            and m.content
        ):
            combined_text = _apply_inline_thinking_template(
                m.thinking_content, m.content, inline_thinking
            )
            msg["content"] = combined_text
        else:
            msg["content"] = (
                m.content
                if m.content
                else [{"type": "text", "text": "[Response truncated due to time limit]"}]
            )

    return msg


def _tool_to_anthropic(tool: Tool) -> Dict[str, Any]:
    """Convert framework `Tool` definitions into Anthropic's schema."""
    return {
        "name": tool.function.name,
        "description": tool.function.description,
        "input_schema": {
            "type": tool.function.parameters.type,
            "properties": tool.function.parameters.properties,
            "required": tool.function.required,
        },
    }


async def aggregate_anthropic_stream(
    stream, on_chunk: Callable[[StreamChunk], Awaitable[None]]
) -> ChatCompletion:
    """Aggregate Anthropic SDK stream events into a `ChatCompletion`."""

    accumulated_content = ""
    thinking_content = ""
    thinking_signature = None
    tool_calls: List[ToolCall] = []
    message_id = None
    created_at = int(time.time())
    finish_reason = "stop"

    tool_json_accumulator: Dict[int, str] = {}
    tool_metadata: Dict[int, Dict[str, Any]] = {}

    async for event in stream:
        event_type = event.type

        if event_type == "message_start":
            message_id = event.message.id
            created_at = int(time.time())

        elif event_type == "content_block_start":
            block = event.content_block
            index = event.index

            if block.type == "tool_use":
                tool_metadata[index] = {"id": block.id, "name": block.name}
                tool_json_accumulator[index] = ""

        elif event_type == "content_block_delta":
            block = event.delta
            index = event.index

            if block.type == "text_delta":
                text = block.text
                accumulated_content += text
                await on_chunk(StreamChunk("token", {"text": text}))

            elif block.type == "tool_use_delta":
                tool_json_accumulator[index] += block.partial_json

            elif block.type == "thinking_delta":
                thinking_content += block.thinking
                if hasattr(block, "signature") and block.signature:
                    thinking_signature = block.signature
                await on_chunk(StreamChunk("thinking", {"text": block.thinking}))

        elif event_type == "content_block_stop":
            index = event.index
            block = event.content_block

            if block.type == "tool_use":
                raw_json = tool_json_accumulator.get(index, "")
                try:
                    tool_input = json.loads(raw_json) if raw_json else {}
                    tool_call = ToolCall(
                        id=tool_metadata[index]["id"],
                        name=tool_metadata[index]["name"],
                        args=tool_input,
                    )
                    tool_calls.append(tool_call)

                    await on_chunk(
                        StreamChunk(
                            "tool_call_complete",
                            {
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "args": tool_input,
                                "index": index,
                            },
                        )
                    )

                except json.JSONDecodeError as e:
                    await on_chunk(
                        StreamChunk(
                            "tool_call_error",
                            {
                                "index": index,
                                "id": tool_metadata[index]["id"],
                                "name": tool_metadata[index]["name"],
                                "error": f"Invalid JSON: {str(e)}",
                                "partial_json": tool_json_accumulator[index],
                            },
                        )
                    )

        elif event_type == "message_delta":
            if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                finish_reason = event.delta.stop_reason or "stop"

        elif event_type == "ping":
            await on_chunk(StreamChunk("ping", {}))

        elif event_type == "error":
            error_data = {
                "type": event.error.type,
                "message": event.error.message,
            }
            await on_chunk(StreamChunk("error", error_data))
            raise Exception(f"Anthropic stream error: {error_data}")

    final_message = Message(
        role="assistant",
        content=accumulated_content if accumulated_content else "",
        thinking_content=thinking_content or None,
        thinking_signature=thinking_signature,
        tool_calls=tool_calls,
    )

    await on_chunk(
        StreamChunk(
            "assistant_complete",
            {
                "content": accumulated_content,
                "tool_call_count": len(tool_calls),
                "finish_reason": finish_reason,
            },
        )
    )

    final_anthropic_message = await stream.get_final_message()

    usage = Usage(
        prompt_tokens=final_anthropic_message.usage.input_tokens,
        completion_tokens=final_anthropic_message.usage.output_tokens,
        total_tokens=
        final_anthropic_message.usage.input_tokens
        + final_anthropic_message.usage.output_tokens,
    )

    completion = ChatCompletion(
        id=message_id or f"msg_{int(time.time() * 1000)}",
        object="chat.completion",
        created=created_at,
        model="",
        usage=usage,
        choices=[Choice(0, final_message, finish_reason)],
    )

    return completion


async def rollout_anthropic(
    actor: Actor,
    on_chunk: Callable[[StreamChunk], Awaitable[None]],
    user_message_for_thinking: Optional[str] = None,
    turn_idx: int = 0,
    inline_thinking: Optional[str] = None,
) -> Actor:
    """Call Anthropic's API using streaming and update the actor."""
    client_kwargs = {"api_key": actor.endpoint.api_key}
    if actor.endpoint.api_base:
        client_kwargs["base_url"] = actor.endpoint.api_base
    client = AsyncAnthropic(**client_kwargs)

    truncated_trajectory = actor.trajectory

    system_prompt = None
    messages = []

    for m in truncated_trajectory.messages:
        if m.role == "system":
            system_prompt = m.content
        elif m.role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
        else:
            messages.append(_message_to_anthropic(m, inline_thinking))

    if user_message_for_thinking and turn_idx > 0:
        messages.append({"role": "user", "content": user_message_for_thinking})

    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "Begin."})

    messages_with_cache = add_cache_control_to_last_content(messages)

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
        params["thinking"] = actor.endpoint.thinking

    max_retries = 10
    base_delay = 2
    completion = None

    for attempt in range(max_retries + 1):
        try:
            async with client.messages.stream(  # type: ignore[missing-argument]
                **params,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            ) as stream:
                completion = await aggregate_anthropic_stream(stream, on_chunk)
                break

        except Exception as e:
            print(
                f"🔄 Anthropic API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
            )

            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"   Retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            else:
                print("\n" + "=" * 80)
                print("ERROR: Failed to call Anthropic API after all retries")
                print("=" * 80)
                print("Final Exception:", str(e))
                print("\nExact request sent to Anthropic:")
                print(json.dumps(params, indent=2, default=str))
                print("=" * 80)
                raise

    if completion is None:
        raise Exception("Failed to get completion after all retries")

    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message

    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion],
    )

    await client.close()
    return replace(actor, trajectory=new_trajectory)
