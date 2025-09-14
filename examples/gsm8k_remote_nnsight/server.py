#!/usr/bin/env python3
"""Nano NDIF: OpenAI chat-completions wrapper using NNsight.

Endpoints:
- GET /v1/models
- POST /v1/chat/completions
- POST /v1/interventions  (configure capture; persists in-process)

Notes:
- Default model: willcb/Qwen3-0.6B
- Uses HF chat template when available; falls back to a simple role-prefix format
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
import threading

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from nnsight import LanguageModel

from .config import InterventionConfig, SUPPORTED_HOOK_POINTS
from .activation_capture import write_activation_set, new_request_id


DEFAULT_MODEL_ID = "willcb/Qwen3-0.6B"


# ---------- Pydantic Schemas ----------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL_ID)
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    # Optional per-call NDIF overrides
    ndif: Optional[Dict[str, Any]] = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    usage: Usage
    choices: List[Choice]
    # Breadcrumb for where activations were written (if enabled)
    ndif: Optional[Dict[str, Any]] = None


class InterventionsRequest(BaseModel):
    enabled: bool = True
    layers: List[int] = []
    hook_points: List[str] = list(SUPPORTED_HOOK_POINTS)
    mode: str = "trace"  # or "generate"
    save_dir: Optional[str] = None
    per_request_subdir: bool = True
    sample_hidden: Optional[int] = 64
    save_format: str = "pt"  # or "npy"
    include_metadata_json: bool = True


class InterventionsResponse(BaseModel):
    ok: bool
    config: Dict[str, Any]


# ---------- App State ----------

app = FastAPI(title="nano-ndif", version="0.1.0")
llm: LanguageModel | None = None
MODEL_ID: str = DEFAULT_MODEL_ID
DEVICE_MAP: str = "auto"
cfg_lock = threading.Lock()
cfg = InterventionConfig()  # mutable in-process config
_hf_model = None  # lazy HF fallback for generation


# ---------- Prompt helpers ----------

def to_hf_chat(messages: List[Message]) -> List[Dict[str, str]]:
    # Map OpenAI-like roles to HF roles
    role_map = {"system": "system", "user": "user", "assistant": "assistant"}
    chat: List[Dict[str, str]] = []
    for m in messages:
        r = role_map.get(m.role, m.role)
        chat.append({"role": r, "content": m.content})
    return chat


def render_prompt(messages: List[Message]) -> Dict[str, Any]:
    assert llm is not None
    chat = to_hf_chat(messages)
    tok = llm.tokenizer

    # Prefer HF chat template if present
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        try:
            # Produce text via chat template, then tokenize to tensors
            prompt_text = tok.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            return tok(prompt_text, return_tensors="pt")
        except Exception:
            pass

    # Fallback: simple role-tagged text
    text = []
    for m in messages:
        if m.role == "system":
            text.append(f"System: {m.content}")
        elif m.role == "user":
            text.append(f"Human: {m.content}")
        elif m.role == "assistant":
            text.append(f"Assistant: {m.content}")
        else:
            text.append(f"{m.role}: {m.content}")
    text.append("Assistant:")
    rendered = "\n".join(text)
    return llm.tokenizer(rendered, return_tensors="pt")


def render_prompt_text(messages: List[Message]) -> str:
    assert llm is not None
    chat = to_hf_chat(messages)
    tok = llm.tokenizer
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        try:
            return tok.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    # Fallback
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"System: {m.content}")
        elif m.role == "user":
            parts.append(f"Human: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
        else:
            parts.append(f"{m.role}: {m.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


# ---------- Capture helpers ----------

def _collect_targets(layers: List[int], hook_points: List[str]):
    assert llm is not None
    targets = []
    for layer_idx in layers:
        for hp in hook_points:
            if hp == "input_layernorm.output":
                targets.append((f"layer_{layer_idx}_input_layernorm_output", llm.model.layers[layer_idx].input_layernorm.output))
            elif hp == "post_attention_layernorm.output":
                targets.append((f"layer_{layer_idx}_post_attention_layernorm_output", llm.model.layers[layer_idx].post_attention_layernorm.output))
    return targets


def _extract_proxy_tensor(proxy) -> torch.Tensor:
    # NNsight proxies expose .value after context exit
    t = None
    if hasattr(proxy, "value"):
        t = proxy.value
    else:
        # Fallback: best effort detach
        try:
            t = proxy.detach()
        except Exception:
            pass
    if not isinstance(t, torch.Tensor):
        raise RuntimeError("Could not resolve proxy to tensor")
    return t


# ---------- Routes ----------

def _load_model(model_id: str, device_map: str) -> LanguageModel:
    return LanguageModel(model_id, device_map=device_map)


@app.on_event("startup")
async def _startup():
    global llm
    llm = _load_model(MODEL_ID, DEVICE_MAP)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 1677649963,
                "owned_by": "nano-ndif",
            }
        ],
    }


@app.get("/health")
async def health():
    return {
        "ok": llm is not None,
        "model": MODEL_ID,
        "device": str(getattr(llm, "device", None)) if llm is not None else None,
    }


class ModelLoadRequest(BaseModel):
    model: str
    device_map: Optional[str] = None


@app.post("/v1/model")
async def load_model(req: ModelLoadRequest):
    """Hot-reload the underlying model and device map."""
    global llm, MODEL_ID, DEVICE_MAP, _hf_model
    model_id = req.model
    device_map = req.device_map or DEVICE_MAP
    try:
        new_llm = _load_model(model_id, device_map)
        old_llm = llm
        llm = new_llm
        MODEL_ID = model_id
        DEVICE_MAP = device_map
        _hf_model = None  # reset HF fallback cache
        # Free old model
        del old_llm
        try:
            import gc, torch as _torch
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        # Re-validate interventions against new model depth
        valid = True
        note = None
        try:
            n_layers = len(llm.model.layers)
        except Exception:
            n_layers = None
        if n_layers is not None:
            with cfg_lock:
                try:
                    cfg.validate(n_layers=n_layers)
                except Exception:
                    filtered = [i for i in cfg.layers if 0 <= i < n_layers]
                    cfg.layers = filtered
                    if not filtered:
                        cfg.enabled = False
                    valid = False
                    note = f"Adjusted interventions for new model; enabled={cfg.enabled}, layers={cfg.layers}"
        return {"ok": True, "model": MODEL_ID, "device_map": DEVICE_MAP, "interventions_valid": valid, "interventions_note": note}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {e}")


@app.post("/v1/interventions", response_model=InterventionsResponse)
async def set_interventions(body: InterventionsRequest):
    global cfg
    # Build new config from request
    new_cfg = InterventionConfig(
        enabled=body.enabled,
        layers=body.layers or cfg.layers,
        hook_points=body.hook_points or cfg.hook_points,
        mode=body.mode or cfg.mode,
        save_dir=Path(body.save_dir) if body.save_dir else cfg.save_dir,
        per_request_subdir=body.per_request_subdir,
        sample_hidden=body.sample_hidden,
        save_format=body.save_format,
        include_metadata_json=body.include_metadata_json,
    )
    # Validate including layer index bounds
    try:
        n_layers = len(llm.model.layers) if llm is not None else None
    except Exception:
        n_layers = None
    new_cfg.validate(n_layers=n_layers)
    with cfg_lock:
        cfg = new_cfg
    return {"ok": True, "config": new_cfg.to_dict()}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Snapshot current interventions config (thread-safe)
    with cfg_lock:
        local_cfg = cfg.clone()

    # Render prompt inputs (tensors) and prompt text for generate()
    enc = render_prompt(req.messages)
    prompt_text = render_prompt_text(req.messages)
    input_ids = enc["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    # Move to device only if it's a real device (avoid 'meta')
    dev = getattr(llm, "device", None)
    dev_str = str(dev) if dev is not None else None
    if dev_str and dev_str != "meta":
        input_ids = input_ids.to(dev)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None and not isinstance(attn_mask, torch.Tensor):
        attn_mask = torch.tensor(attn_mask)
    if attn_mask is not None and dev_str and dev_str != "meta":
        attn_mask = attn_mask.to(dev)

    # Per-call overrides for NDIF (without mutating global cfg)
    run_dir_override: Optional[str] = None
    per_request_subdir_override: Optional[bool] = None
    ndif_breadcrumb: Optional[Dict[str, Any]] = None
    request_id = new_request_id()
    if req.ndif:
        run_dir_override = req.ndif.get("save_dir")
        if run_dir_override is not None:
            run_dir_override = str(run_dir_override)
        per_request_subdir_override = req.ndif.get("per_request_subdir")

    def _gen_args():
        return {
            "max_new_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
        }

    # Generation + capture
    generated_ids = None
    if local_cfg.enabled and local_cfg.mode == "generate":
        proxies: Dict[str, Any] = {}
        with llm.generate(prompt_text, **_gen_args()) as tracer:
            targets = _collect_targets(local_cfg.layers, local_cfg.hook_points)
            for name, node in targets:
                proxies[name] = node.save()

        generated_ids = getattr(tracer, "generator", None)
        generated_ids = getattr(generated_ids, "output", None) or getattr(tracer, "output", None)

        # Some wrappers return (batch, seq); ensure indexing consistent
        if isinstance(generated_ids, (list, tuple)):
            generated_ids = generated_ids[0]
        # Write activations to disk
        if run_dir_override is not None:
            base_dir = Path(run_dir_override)
        else:
            base_dir = local_cfg.save_dir
        use_subdir = per_request_subdir_override if per_request_subdir_override is not None else local_cfg.per_request_subdir
        if use_subdir:
            from time import strftime
            run_dir = base_dir / f"run-{strftime('%Y%m%d-%H%M%S')}"
        else:
            run_dir = base_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        tensors: Dict[str, torch.Tensor] = {k: _extract_proxy_tensor(v) for k, v in proxies.items()}
        ndif_breadcrumb = write_activation_set(run_dir, request_id, tensors, local_cfg)
    else:
        # Plain generation using nnsight's generate() wrapper; fallback to HF generate if needed
        try:
            with llm.generate(
                prompt_text,
                **_gen_args(),
            ) as tracer:
                pass
            generated_ids = getattr(tracer, "generator", None)
            generated_ids = getattr(generated_ids, "output", None) or getattr(tracer, "output", None)
            if isinstance(generated_ids, (list, tuple)):
                generated_ids = generated_ids[0]
        except Exception:
            # Fallback: HF generate
            from transformers import AutoModelForCausalLM
            global _hf_model
            if _hf_model is None:
                _hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
            toks = llm.tokenizer(prompt_text, return_tensors="pt")
            input_ids_hf = toks["input_ids"].to(_hf_model.device)
            out = _hf_model.generate(
                input_ids_hf,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=True,
                pad_token_id=llm.tokenizer.eos_token_id,
            )
            generated_ids = out

        # Optionally run a secondary trace over prompt to collect activations
        if local_cfg.enabled and local_cfg.mode == "trace":
            proxies: Dict[str, Any] = {}
            with llm.trace(prompt_text) as tracer:
                targets = _collect_targets(local_cfg.layers, local_cfg.hook_points)
                for name, node in targets:
                    proxies[name] = node.save()

            if run_dir_override is not None:
                base_dir = Path(run_dir_override)
            else:
                base_dir = local_cfg.save_dir
            use_subdir = per_request_subdir_override if per_request_subdir_override is not None else local_cfg.per_request_subdir
            if use_subdir:
                from time import strftime
                run_dir = base_dir / f"run-{strftime('%Y%m%d-%H%M%S')}"
            else:
                run_dir = base_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            tensors: Dict[str, torch.Tensor] = {k: _extract_proxy_tensor(v) for k, v in proxies.items()}
            ndif_breadcrumb = write_activation_set(run_dir, request_id, tensors, local_cfg)

    # Decode text
    text = llm.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Heuristic: drop prompt text if template fallback used
    try:
        rendered_text = llm.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if text.startswith(rendered_text):
            text = text[len(rendered_text):].strip()
    except Exception:
        pass

    # Build response
    prompt_tokens = int(input_ids.numel())
    completion_tokens = int(generated_ids.shape[-1] - input_ids.shape[-1]) if generated_ids is not None else 0
    created = int(time.time())

    resp = ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=created,
        model=req.model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        choices=[
            Choice(index=0, message=Message(role="assistant", content=text), finish_reason="stop")
        ],
        ndif=ndif_breadcrumb,
    )

    return resp


def main():
    import argparse
    global MODEL_ID, DEVICE_MAP

    parser = argparse.ArgumentParser(description="nano-ndif server")
    parser.add_argument("--model", default=MODEL_ID, help="HF model id")
    parser.add_argument("--device-map", default=DEVICE_MAP, help="device map (auto/cpu/cuda/…)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    MODEL_ID = args.model
    DEVICE_MAP = args.device_map
    host = args.host
    port = args.port

    print(f"nano-ndif serving on http://{host}:{port} (model={MODEL_ID})")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

