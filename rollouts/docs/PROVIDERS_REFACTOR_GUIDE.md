# Providers Refactor Guide

This document explains how to correctly split `agents.py` into `agents.py` + `providers.py` without breaking function calling.

## Overview

The goal is to move provider-specific LLM API code to `providers.py` while keeping agent orchestration logic in `agents.py`.

## What Goes Where

### `providers.py` (LLM API Layer)
Move these functions to handle LLM API calls:

**Provider Functions:**
- `rollout_openai()` - OpenAI API calls
- `rollout_anthropic()` - Anthropic API calls
- `rollout_moonshot()` - Moonshot API calls
- `rollout_vllm()` - vLLM API calls

**Conversion Helpers:**
- `_message_to_openai()` - Convert messages to OpenAI format
- `_message_to_anthropic()` - Convert messages to Anthropic format
- `_tool_to_openai()` - Convert tools to OpenAI format
- `_tool_to_anthropic()` - Convert tools to Anthropic format

**Stream Processing:**
- `aggregate_stream()` - Process OpenAI-style streams
- `aggregate_anthropic_stream()` - Process Anthropic streams

**Utilities:**
- `add_cache_control_to_last_content()` - Anthropic caching
- `verbose()` - Logging helper
- `_parse_usage()`, `_parse_completion()` - Response parsing

### `agents.py` (Agent Orchestration Layer)
Keep these functions for agent logic:

**Core Dispatch:**
- `rollout()` - **CRITICAL**: Provider dispatch function
- `run_agent_step()` - Turn execution logic
- `run_agent()` - Main agent loop

**Tool Handling:**
- `confirm_tool_with_feedback()` - Tool confirmation
- `process_pending_tools()` - Tool execution
- `handle_tool_error()` - Error handling

**Agent Lifecycle:**
- `inject_turn_warning()` - Turn warnings
- `inject_tool_reminder()` - Tool reminders
- `handle_stop_max_turns()` - Stop conditions

**Streaming:**
- `stdout_handler()` - Default output handler

**Checkpoints:**
- `handle_checkpoint_event()` - Checkpoint handling

## Critical Requirements

### 1. Maintain Exact Function Signatures
```python
# WRONG - changing parameters breaks callers
async def rollout_anthropic(actor, on_chunk, extra_param):

# RIGHT - keep exact same signature
async def rollout_anthropic(actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]],
                          user_message_for_thinking: Optional[str] = None, turn_idx: int = 0,
                          inline_thinking: Optional[str] = None) -> Actor:
```

### 2. Preserve JSON Parsing Logic
```python
# CRITICAL: Handle empty JSON strings correctly
args = json.loads(tc.function.arguments) if tc.function.arguments else {}

# NOT: This will fail on empty strings
args = json.loads(tc.function.arguments)
```

### 3. Don't Forget the Dispatch Function
The `rollout()` function in `agents.py` is **essential**:

```python
async def rollout(actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]]=stdout_handler,
                  user_message_for_thinking: Optional[str] = None, turn_idx: int = 0,
                  inline_thinking: Optional[str] = None) -> Actor:
    """Dispatch to the appropriate provider rollout function."""
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
        print(f"Invalid provider {actor.endpoint.provider}")
        sys.exit(0)
    return new_actor
```

### 4. Update Imports Correctly
In `agents.py`:
```python
from .providers import (
    rollout_anthropic,
    rollout_moonshot,
    rollout_openai,
    rollout_vllm,
    # ... other functions
)
```

## Testing the Refactor

1. **Run simple calculator demo** - Tests basic function calling
2. **Run search calculator demo** - Tests complex search tools
3. **Verify LLM calls functions with proper parameters** - Not `add({})` but `add({'value': 27})`

## Common Mistakes

❌ **Moving the `rollout()` dispatch function** - This breaks the call chain
❌ **Changing function signatures** - Breaks existing callers
❌ **Missing JSON parsing safety checks** - Causes empty parameter calls
❌ **Forgetting to update imports** - Causes import errors
❌ **Not testing thoroughly** - Function calling can silently break

## Why This Refactor Failed Originally

The previous attempt moved provider functions to `providers.py` but:
1. **Missing `rollout()` dispatch** - No way to route to providers
2. **Incomplete imports** - Functions weren't properly accessible
3. **JSON parsing differences** - Subtle bugs in argument handling
4. **Import path issues** - Python was using cached/built versions

By following this guide, the refactor can be done safely while maintaining all functionality.