# NNsight Activation Capture - Debugging Context

## Goal
Get activation capture working in `server_singlepass.py` for the GSM8K nnsight remote example.

## What We're Trying To Do
1. Run a FastAPI server that loads a language model using nnsight
2. Accept chat completion requests via OpenAI-compatible API
3. Capture neural network activations during generation
4. Save activation tensors to disk on remote GPU
5. Return activation metadata in the API response

## Current Status
- ✅ Server starts successfully on port 8011
- ✅ Health endpoint works: `/health` returns 200
- ✅ Model loading works: `/models/load` returns success
- ✅ Chat completions work: `/v1/chat/completions` returns 200
- ❌ Activation capture fails: returns empty `activations_meta: {}`

## Environment
- **Remote GPU**: RunPod instance with RTX A5000
- **Python**: 3.10.12
- **nnsight version**: 0.5.0+ (has breaking API changes)
- **Model**: willcb/Qwen3-0.6B
- **Server file**: `examples/gsm8k_nnsight_remote/server_singlepass.py`

## Server Code (Current State)

### File: `examples/gsm8k_nnsight_remote/server_singlepass.py`

#### Lines 1-32: Imports and Setup
```python
#!/usr/bin/env python3
"""
Single-pass NNsight FastAPI server (OpenAI-style chat + activation capture).
Key behavior:
- POST /models/load loads a HF model into NNsight with optional savepoints.
- POST /v1/chat/completions runs one-pass generate while saving activations.
- Saves activations to disk under /tmp/nnsight_activations as .pt files.
- GET /health reports model status.

This adapts the server design from nnsight_server.txt to file-backed storage for
quick remote debugging (existence checks, SSH inspection).
"""

import json
import os
import re
import time
import uuid
import threading
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# NNsight
from nnsight import LanguageModel
import nnsight

app = FastAPI(title="OpenAI-style Chat + NNsight (single pass)")
```

#### Lines 86-93: ManagedModel Class
```python
class ManagedModel:
    def __init__(self, lm: LanguageModel, tokenizer: AutoTokenizer, savepoints: List[SavePointSpec]):
        self.lm = lm
        self.tokenizer = tokenizer
        self.savepoints = savepoints
        self.lock = threading.Lock()
        self.model_id = getattr(tokenizer, "name_or_path", "unknown")
```

#### Lines 304-358: The Failing Activation Capture Code
```python
# Line 304: Chat endpoint starts
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded. POST /models/load first.")
    mm = MODELS[req.model]

    # Prepare prompt (string) using chat template if available
    messages = [m.model_dump() for m in req.messages]
    prompt_text = _render_prompt_text(mm.tokenizer, messages)

    # Approx usage accounting for prompt
    prompt_ids = mm.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_tokens = int(prompt_ids.numel())

    # One-pass: generate AND trace in the same context
    activation_proxies: Dict[str, Any] = {}
    generated_output = None
    with mm.lock:
        try:
            gen_kwargs = dict(
                max_new_tokens=req.max_tokens,
                temperature=max(req.temperature, 1e-5),
                top_p=req.top_p,
            )

            # Let NNsight/Accelerate handle device placement; keep inputs on CPU
            # Ensure special tokens are properly configured
            if mm.lm.model.config.pad_token_id is None:
                if mm.tokenizer.pad_token_id is None and mm.tokenizer.eos_token_id is not None:
                    mm.tokenizer.pad_token = mm.tokenizer.eos_token
                mm.lm.model.config.pad_token_id = mm.tokenizer.pad_token_id

            # WRAP the exact working test endpoint logic
            print(f"DEBUG: Using working test endpoint logic inside chat endpoint")

            # Multi-token activation capture using correct NNsight pattern
            try:
                print("DEBUG: Using multi-token activation capture pattern")

                # Use modern nnsight pattern (v0.5.0+): regular Python list for accumulation
                logits_list = []

                # Use tracer.all() to capture activations across ALL generated tokens
                with mm.lm.generate(prompt_text, max_new_tokens=req.max_tokens) as tracer:
                    with tracer.all():
                        # Save each token's logits
                        saved_logits = mm.lm.lm_head.output.save()
                        logits_list.append(saved_logits)

                activation_proxies["_logits"] = logits_list
                generated_output = tracer.output  # Capture generated output from tracer
                print(f"DEBUG: SUCCESS! Multi-token logits captured: {len(logits_list)} tokens")

                # TODO: Fix custom savepoints - they cause OutOfOrderError when called outside generate context
                # Save custom savepoints using same working pattern
                # for sp in mm.savepoints:
                #     try:
                #         node = _safe_eval_selector(mm.lm, sp.selector)
                #         saved_activation = node.save()  # ← THIS CAUSES OutOfOrderError!
                #         activation_proxies[sp.name] = saved_activation
                #     except Exception as e:
                #         activation_proxies[sp.name] = {"error": f"Could not save '{sp.selector}': {e}"}

            except Exception as e:
                print(f"DEBUG: Even wrapped test endpoint logic failed: {e}")
                # Create minimal response to avoid total failure
                activation_proxies = {}
                generated_output = None

            print(f"DEBUG: Final activation count: {len(activation_proxies)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation/tracing failed: {e}")
```

## Error History (Chronological)

### Error 1: Dependency Name Mismatch (FIXED)
**Line 43 of remote output (pyproject.toml issue)**
```
error: Extra `examples-gsm8k-nnsight-remote` is not defined in the project's `optional-dependencies` table
```
**Cause**: UV normalizes underscores to hyphens in extra names. Script used `examples_gsm8k_nnsight_remote` but pyproject only had `examples_gsm8k_remote_nnsight`.

**Fix**: Added alias in pyproject.toml lines 122-139

### Error 2: Server File Not Found (FIXED)
**Server log output**:
```
/root/.bifrost/workspace/.venv/bin/python3: can't open file '/root/.bifrost/workspace/examples/gsm8k_nnsight_remote/server_singlepass.py': [Errno 2] No such file or directory
```
**Cause**: The `examples/gsm8k_nnsight_remote/` directory wasn't committed to git, so bifrost couldn't push it.

**Fix**: Committed the directory with `git add examples/gsm8k_nnsight_remote/`

### Error 3: nginx 405 Not Allowed (WORKAROUND)
**Remote output lines 74-80**:
```html
<html>
<head><title>405 Not Allowed</title></head>
<body>
<center><h1>405 Not Allowed</h1></center>
<hr><center>nginx/1.18.0 (Ubuntu)</center>
</body>
</html>
```
**Cause**: RunPod's nginx proxy on port 8001 blocks POST requests for security reasons.

**Workaround**: Changed to port 8011 (nginx doesn't intercept non-standard ports)

### Error 4: Deprecated nnsight.list() API (ATTEMPTED FIX)
**Server log (tail -30 ~/nnsight_singlepass.log)**:
```
/root/.bifrost/workspace/.venv/lib/python3.10/site-packages/nnsight/__init__.py:80: UserWarning: builtins.list is deprecated as of v0.5.0 and will be removed in a future version.
Use the standard `list()` instead.
  warnings.warn(deprecation_message)
DEBUG: Using working test endpoint logic inside chat endpoint
DEBUG: Using multi-token activation capture pattern
DEBUG: Even wrapped test endpoint logic failed: 'list' object has no attribute 'save'
DEBUG: Final activation count: 0
```
**Original code (server_singlepass.py line 328)**:
```python
logits_list = nnsight.list().save()
```

**Attempted fix**:
```python
logits_list = []
# ...
with mm.lm.generate(prompt_text, max_new_tokens=req.max_tokens) as tracer:
    with tracer.all():
        saved_logits = mm.lm.lm_head.output.save()
        logits_list.append(saved_logits)
```

### Error 5: tracer.output Attribute Missing (CURRENT)
**Server log (current state)**:
```
DEBUG: Using working test endpoint logic inside chat endpoint
DEBUG: Using multi-token activation capture pattern
DEBUG: Even wrapped test endpoint logic failed: 'RemoteInterleavingTracer' object has no attribute 'output'
DEBUG: Final activation count: 0
DEBUG: generated_output is None
```
**Failing code (server_singlepass.py line 338)**:
```python
generated_output = tracer.output  # This line fails
```

**Error details**: The `tracer` object returned by `mm.lm.generate()` is of type `RemoteInterleavingTracer` and does not have an `output` attribute in nnsight v0.5.0+.

## Working Pattern from Test File

### File: `examples/gsm8k_nnsight_remote/archive/test_nnsight_tutorial.py` lines 36-43
```python
# Test generate pattern
print("\n--- Testing generate pattern ---")
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke(prompt):
        logits_gen = model.lm_head.output.save()

print(f"✅ Generate pattern successful!")
print(f"Generated logits shape: {logits_gen.shape}")
print(f"Generated text: {tokenizer.decode(tracer.output[0])}")
```

**Note**: This test file uses `tracer.invoke(prompt)` instead of passing prompt to `generate()`, and it shows `tracer.output[0]` working.

## API Test Results

### Health Check (WORKS)
```bash
$ curl -s http://localhost:8011/health
{"status":"ok","loaded_models":[]}
```

### Model Loading (WORKS)
```bash
$ curl -s -X POST http://localhost:8011/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "willcb/Qwen3-0.6B", "device_map": "auto", "savepoints": []}'
{"ok":true,"model":"willcb/Qwen3-0.6B","savepoints":[]}
```

### Chat Completion with Activation Capture (FAILS)
```bash
$ curl -s -X POST http://localhost:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "willcb/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Count to 3"}],
    "max_tokens": 10,
    "store_activations": true
  }'
```

**Response**:
```json
{
    "id": "chatcmpl-3d4a00eb-4ffc-40b7-a782-04da68b23cbb",
    "object": "chat.completion",
    "created": 1759203483,
    "model": "willcb/Qwen3-0.6B",
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": "[Generated output not captured]",
            "run_id": "3d4a00eb-4ffc-40b7-a782-04da68b23cbb",
            "session_id": "3d4a00eb",
            "activation_files": {}
        }
    }],
    "usage": {
        "prompt_tokens": 33,
        "completion_tokens": 6,
        "total_tokens": 39
    },
    "activations_meta": {}
}
```

**Problem**: `"activations_meta": {}` is empty, should contain activation tensor metadata.

## Related Code Files

### server_composition.py (Also Has Issues)
**File**: `examples/gsm8k_nnsight_remote/server_composition.py` lines 40-69

Has same problems with deprecated API:
```python
def capture_activations(self, prompt: str, max_new_tokens: int = 3) -> Dict[str, Any]:
    """Multi-token NNsight activation capture using correct pattern"""
    with self.lock:
        # Initialize NNsight list to accumulate activations across all generated tokens
        logits_list = nnsight.list().save()  # ← DEPRECATED

        # Use proper multi-token generation pattern
        with self.lm.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
            with tracer.all():
                logits_list.append(self.lm.lm_head.output)

        # Extract generated text from tracer output
        try:
            if hasattr(tracer, 'output') and tracer.output is not None:  # ← FAILS
                generated_text = self.tokenizer.decode(tracer.output[0], skip_special_tokens=True)
```

## Key Questions

1. What is the correct nnsight v0.5.0+ pattern for capturing activations during `generate()`?
2. How to access generated tokens from the `RemoteInterleavingTracer` object?
3. How to accumulate activations across multiple generated tokens (not just first token)?
4. Why does `tracer.output` work in `test_nnsight_tutorial.py` but not in the server code?

## GPU Access
- **SSH**: `root@69.30.85.244:22111`
- **Server logs**: `tail -f ~/nnsight_singlepass.log`
- **Tmux session**: `tmux attach -t nnsight-singlepass`
- **GPU ID**: `j1l1otewhlo5nj`
- **Port**: 8011

## Repository Info
- **Repo**: llm-workbench
- **Branch**: main
- **Latest commit**: e53e913 (Fix nnsight v0.5.0+ compatibility)
- **Location**: `/Users/chiraagbalu/llm-workbench`