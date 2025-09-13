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

app = FastAPI(title="OpenAI-style Chat + NNsight (single pass)")

# ----------------------------
# Constants
# ----------------------------
STORE_DIR = "/tmp/nnsight_activations"
os.makedirs(STORE_DIR, exist_ok=True)

# ----------------------------
# Models & storage
# ----------------------------
class SavePointSpec(BaseModel):
    name: str
    # NNsight selector relative to the LanguageModel object
    selector: str = Field(..., examples=["model.layers[-1].input_layernorm.output", "lm_head.output"]) 

class LoadModelRequest(BaseModel):
    model_id: str = Field(..., examples=["openai-community/gpt2", "willcb/Qwen3-0.6B"])
    device_map: Union[str, Dict[str, int]] = "auto"
    trust_remote_code: bool = False
    savepoints: List[SavePointSpec] = []
    max_seq_len: Optional[int] = None  # reserved; not used directly here

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    # storage controls
    store_activations: bool = True
    session_id: Optional[str] = None

class ChatChoice(BaseModel):
    index: int
    finish_reason: Optional[str]
    message: Dict[str, Any]

class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]

class ManagedModel:
    def __init__(self, lm: LanguageModel, tokenizer: AutoTokenizer, savepoints: List[SavePointSpec]):
        self.lm = lm
        self.tokenizer = tokenizer
        self.savepoints = savepoints
        self.lock = threading.Lock()
        self.model_id = getattr(tokenizer, "name_or_path", "unknown")

MODELS: Dict[str, ManagedModel] = {}

SAFE_SELECTOR_REGEX = re.compile(r"^[a-zA-Z0-9_\.\[\]\-]+$")  # attribute/index path only

# ----------------------------
# Helpers
# ----------------------------
def _now() -> int:
    return int(time.time())

def _safe_eval_selector(lm: LanguageModel, selector: str):
    """
    Enhanced selector that supports both string parsing and direct layer access.
    Uses patterns proven to work in outlier_features_moe.
    """
    # Handle common patterns directly (more reliable)
    if selector == "output.logits":
        return lm.output.logits
    elif selector.startswith("model.layers[") and ".input_layernorm.output" in selector:
        # Extract layer number from selector like "model.layers[0].input_layernorm.output"
        import re
        match = re.search(r"model\.layers\[(\d+)\]\.input_layernorm\.output", selector)
        if match:
            layer_idx = int(match.group(1))
            return lm.model.layers[layer_idx].input_layernorm.output
    elif selector.startswith("model.layers[") and ".post_attention_layernorm.output" in selector:
        # Extract layer number from selector like "model.layers[0].post_attention_layernorm.output"  
        import re
        match = re.search(r"model\.layers\[(\d+)\]\.post_attention_layernorm\.output", selector)
        if match:
            layer_idx = int(match.group(1))
            return lm.model.layers[layer_idx].post_attention_layernorm.output
    
    # Fallback to original string parsing for other cases
    if not SAFE_SELECTOR_REGEX.match(selector):
        raise HTTPException(status_code=400, detail=f"Illegal selector: {selector}")
    obj = lm
    tokens = re.split(r"\.", selector)
    for tok in tokens:
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)((?:\[[^\]]+\])*)$", tok)
        if not m:
            raise HTTPException(status_code=400, detail=f"Bad selector token: {tok}")
        attr, brackets = m.group(1), m.group(2)
        obj = getattr(obj, attr)
        while brackets:
            m2 = re.match(r"^\[([^\]]+)\](.*)$", brackets)
            if not m2:
                raise HTTPException(status_code=400, detail=f"Bad index in selector near: {brackets}")
            idx_str = m2.group(1)
            # Handle negative indices like [-1]
            if idx_str.startswith('-'):
                idx = int(idx_str)
            else:
                idx = int(idx_str)
            brackets = m2.group(2)
            obj = obj[idx]
    return obj

def _tensor_like_to_tensor(x):
    """Accept tensor or list of tensors from NNsight proxy.value and return a tensor."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)) and x and all(isinstance(t, torch.Tensor) for t in x):
        try:
            return torch.stack([t.detach().cpu() for t in x], dim=0)
        except Exception:
            return torch.cat([t.detach().cpu().reshape(1, *t.shape) for t in x], dim=0)
    try:
        return torch.tensor(x)
    except Exception:
        return None

def _tensor_to_small_json(t: torch.Tensor) -> Dict[str, Any]:
    """Return tensor metadata without the actual data (for performance)."""
    t = t.detach().cpu()
    info = {
        "dtype": str(t.dtype), 
        "shape": list(t.shape), 
        "numel": t.numel(),
        "size_mb": t.numel() * t.element_size() / (1024 * 1024)
    }
    # Only include actual tensor data for very small tensors (< 100 elements)
    if t.numel() <= 100:
        info["data"] = t.tolist()
        info["data_included"] = True
    else:
        info["data_included"] = False
        info["note"] = "Tensor data saved to disk - too large for JSON response"
    return info

def _render_prompt_text(tokenizer: AutoTokenizer, messages: List[Dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback naive chat stitch
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        text += f"{role.upper()}: {content}\n"
    text += "ASSISTANT:"
    return text

def _extract_generated_text(lm: LanguageModel, tokenizer: AutoTokenizer, prompt_text: str, max_new_tokens: int) -> str:
    ids = None
    # Try common NNsight patterns
    try:
        val = lm.generator.output.value
        if isinstance(val, torch.Tensor):
            ids = val[0].tolist()
        elif isinstance(val, (list, tuple)):
            ids = list(val[0])
    except Exception:
        pass
    if ids is None:
        try:
            val = lm.output.ids.value
            if isinstance(val, torch.Tensor):
                ids = val[0].tolist()
            elif isinstance(val, (list, tuple)):
                ids = list(val[0])
        except Exception:
            pass
    if ids is None:
        return ""
    text = tokenizer.decode(ids, skip_special_tokens=True)
    if text.startswith(prompt_text):
        return text[len(prompt_text):]
    return tokenizer.decode(ids[-max_new_tokens:], skip_special_tokens=True)

def _save_tensor_pt(t: torch.Tensor, path: str) -> None:
    # Keep size modest if enormous: sample hidden dim modestly
    t_cpu = t.detach().cpu()
    try:
        if t_cpu.dim() >= 1 and t_cpu.numel() > 5_000_000:
            # Heuristic: sample last dim up to 128 features
            sl = [slice(None)] * (t_cpu.dim() - 1) + [slice(0, min(128, t_cpu.shape[-1]))]
            t_cpu = t_cpu[tuple(sl)]
    except Exception:
        pass
    torch.save(t_cpu, path)

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(MODELS.keys())}

@app.post("/models/load")
def load_model(req: LoadModelRequest):
    try:
        tokenizer = AutoTokenizer.from_pretrained(req.model_id, trust_remote_code=req.trust_remote_code)
        lm = LanguageModel(req.model_id, device_map=req.device_map)
        MODELS[req.model_id] = ManagedModel(lm, tokenizer, req.savepoints)
        return {"ok": True, "model": req.model_id, "savepoints": [sp.model_dump() for sp in req.savepoints]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

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
    with mm.lock:
        try:
            gen_kwargs = dict(
                max_new_tokens=req.max_tokens,
                temperature=max(req.temperature, 1e-5),
                top_p=req.top_p,
            )
            # Use trace-then-generate pattern to avoid "Envoy out of order" issues
            with mm.lm.trace() as tracer:
                # Register all savepoints BEFORE any compute starts
                try:
                    activation_proxies["_logits"] = mm.lm.output.logits.save()
                except Exception:
                    pass
                for sp in mm.savepoints:
                    try:
                        node = _safe_eval_selector(mm.lm, sp.selector)
                        activation_proxies[sp.name] = node.save()
                    except Exception as e:
                        activation_proxies[sp.name] = {"error": f"Could not save '{sp.selector}': {e}"}
                
                # Now actually run generation (after hooks are armed)
                _ = mm.lm.generate(prompt_text, **gen_kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation/tracing failed: {e}")

    # Decode completion from the SAME run
    reply_text = _extract_generated_text(mm.lm, mm.tokenizer, prompt_text, req.max_tokens)
    if reply_text is None:
        reply_text = ""

    # Gather + optionally store activations
    run_id = str(uuid.uuid4())
    session_id = req.session_id or run_id[:8]
    saved_files: Dict[str, str] = {}
    small_json: Dict[str, Any] = {}
    for k, proxy in activation_proxies.items():
        if isinstance(proxy, dict) and "error" in proxy:
            small_json[k] = proxy
            continue
        try:
            raw_val = proxy.detach()  # Use correct NNsight API
            t = _tensor_like_to_tensor(raw_val)
            if t is None:
                small_json[k] = {"warning": "Could not convert activation to tensor", "type": str(type(raw_val))}
                continue
            # Save to disk if asked
            if req.store_activations:
                fname = f"activations_{session_id}_{run_id}_{k}.pt"
                fpath = os.path.join(STORE_DIR, fname)
                _save_tensor_pt(t, fpath)
                saved_files[k] = fpath
            # Always include small JSON metadata
            small_json[k] = _tensor_to_small_json(t)
        except Exception as e:
            small_json[k] = {"error": f"Activation not available: {e}"}

    # Usage (approx)
    completion_ids = mm.tokenizer(reply_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    completion_tokens = int(completion_ids.numel())
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    resp = ChatResponse(
        id=f"chatcmpl-{run_id}",
        object="chat.completion",
        created=_now(),
        model=mm.model_id,
        choices=[ChatChoice(
            index=0,
            finish_reason="stop",
            message={"role": "assistant", "content": reply_text, "run_id": run_id, "session_id": session_id, "activation_files": saved_files},
        )],
        usage=usage,
    )

    if req.stream:
        # Minimal SSE: single final delta. (You can expand to token-by-token later.)
        def _iter():
            chunk = {
                "id": resp.id,
                "object": "chat.completion.chunk",
                "created": resp.created,
                "model": resp.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_iter(), media_type="text/event-stream")

    # Attach small activation metadata to response body for debugging
    out = json.loads(resp.model_dump_json())
    out["activations_meta"] = small_json
    return JSONResponse(content=out)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="NNSight single-pass server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

