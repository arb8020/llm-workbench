# Gemini OpenAI-Compatible API Debug Analysis

## Problem Statement

When using Gemini through its OpenAI-compatible endpoint with function calling, the API returns completely empty responses (`content: ''`) instead of either:
1. Using the provided tools correctly, OR  
2. Falling back to text-only responses

This breaks the conversation flow entirely.

## Test Results Summary

✅ **Basic chat works**: Gemini responds correctly to simple text prompts  
❌ **Function calling fails**: Empty response when tools are provided  

## Exact API Parameters Sent

### Working Case (No Tools)
```json
{
  "model": "gemini-1.5-flash",
  "messages": [
    {
      "role": "user",
      "content": "Hello! Please respond with exactly 'Gemini test successful' if you can see this message."
    }
  ],
  "temperature": 0.1,
  "stream": true,
  "max_tokens": 100
}
```
**Result**: `"Gemini test successful"` ✅

### Failing Case (With Tools)
```json
{
  "model": "gemini-1.5-flash", 
  "messages": [
    {
      "role": "user",
      "content": "What is 15 + 27? Please use the calculator tool."
    }
  ],
  "temperature": 0.1,
  "stream": true,
  "max_tokens": 100,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "add",
        "description": "Add a number to the current value",
        "parameters": {
          "type": "object",
          "properties": {
            "value": {
              "type": "number", 
              "description": "Number to add"
            }
          },
          "required": ["value"]
        }
      }
    },
    // ... 5 more similar tools
  ],
  "tool_choice": "auto"
}
```
**Result**: `""` (completely empty) ❌

## Technical Implementation Details

### Rollout Architecture
- Uses `rollout_openai()` function with `provider="openai"`
- Endpoint: `https://generativelanguage.googleapis.com/v1beta/openai`
- Model: `gemini-1.5-flash`
- Client: OpenAI AsyncClient with custom base_url

### Tool Conversion Logic
Tools are converted from internal format to OpenAI format via `_tool_to_openai()`:

```python
def _tool_to_openai(tool: Tool) -> Dict[str, Any]:
    return {
        "type": tool.type,  # "function"
        "function": {
            "name": tool.function.name,
            "description": tool.function.description, 
            "parameters": {
                "type": tool.function.parameters.type,  # "object"
                "properties": tool.function.parameters.properties,
                "required": tool.function.required
            }
        }
    }
```

This produces **valid OpenAI function calling format**.

### Stream Processing
- Uses `aggregate_stream()` to collect streaming chunks
- Returns `Message(role='assistant', content='', tool_calls=[])` when tools present
- No streaming chunks received (empty content suggests no response at all)

## Hypotheses

### Primary Hypothesis: Function Calling Not Supported
Gemini's OpenAI-compatible endpoint likely doesn't support the `tools` and `tool_choice` parameters at all, causing it to fail silently rather than:
- Ignoring unsupported parameters gracefully
- Returning an error message
- Falling back to text-only response

### Secondary Hypotheses

1. **Parameter Incompatibility**: Gemini expects different tool schema format
2. **API Version Issue**: Using wrong API version (`/v1beta/openai` vs `/v1/openai`)  
3. **Model Limitation**: `gemini-1.5-flash` specifically doesn't support function calling
4. **Authentication Issue**: Function calling requires different auth/headers
5. **Streaming Conflict**: Function calling incompatible with `stream: true`

## Source Code Locations

### Key Files
- `rollouts/agents.py:408-464` - `rollout_openai()` function
- `rollouts/agents.py:125-138` - `_tool_to_openai()` converter  
- `rollouts/agents.py:96-122` - `_message_to_openai()` converter
- `rollouts/environments/calculator.py` - Tool definitions

### Test File
- `rollouts/tests/smoke_gemini.py` - Reproduction test case

## Questions for Analysis

1. **Is the OpenAI function calling format we're using correct?**
2. **Does Gemini's OpenAI endpoint support function calling at all?**
3. **Should we try different parameters (remove `tool_choice`, disable streaming, etc.)?**
4. **Is there a different API endpoint or model that supports tools?**
5. **What's the proper way to handle this gracefully in the rollouts framework?**

## Potential Solutions

1. **Try without streaming**: Test with `stream: false`
2. **Remove tool_choice**: Try without the `tool_choice: "auto"` parameter  
3. **Different model**: Test with `gemini-1.5-pro` or other variants
4. **Fallback logic**: Detect empty responses and retry without tools
5. **Native Gemini SDK**: Implement separate `rollout_gemini()` using Google AI SDK
6. **API endpoint variants**: Try different endpoint URLs

## Expected Expert Analysis

Please analyze:
- Whether our OpenAI function calling format is correct
- If Gemini's OpenAI compatibility has known limitations with function calling
- Recommended approaches to either fix the integration or implement proper fallbacks
- Best practices for handling API compatibility issues like this

The goal is to either make Gemini work with tools through the OpenAI endpoint, or gracefully fall back to text-only responses instead of returning empty content.