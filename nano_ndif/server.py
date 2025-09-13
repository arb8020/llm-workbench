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

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from nnsight import LanguageModel

from .config import InterventionConfig, SUPPORTED_HOOK_POINTS
from .activation_capture import write_activation_set, new_request_id


DEFAULT_MODEL_ID = os.environ.get("NANO_NDIF_MODEL", "willcb/Qwen3-0.6B")


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
cfg = InterventionConfig()  # mutable in-process config


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

@app.on_event("startup")
async def _startup():
    global llm
    device_map = os.environ.get("NANO_NDIF_DEVICE_MAP", "auto")
    llm = LanguageModel(MODEL_ID, device_map=device_map)


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
    new_cfg.validate()
    cfg = new_cfg
    return {"ok": True, "config": cfg.to_dict()}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Render prompt inputs
    enc = render_prompt(req.messages)
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

    # Per-call overrides for NDIF
    run_cfg = cfg
    run_dir_override: Optional[str] = None
    if req.ndif:
        # Allow overriding only safe, per-call options
        if "save_dir" in req.ndif:
            run_dir_override = str(req.ndif["save_dir"])  # use for this request only
        if "per_request_subdir" in req.ndif:
            run_cfg.per_request_subdir = bool(req.ndif["per_request_subdir"])

    # Generate (optionally inside nnsight generate() context)
    generated_ids = None
    request_id = new_request_id()
    ndif_breadcrumb: Dict[str, Any] | None = None

    def _gen_args() -> Dict[str, Any]:
        return dict(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=True,
            pad_token_id=llm.tokenizer.eos_token_id,
        )

    if cfg.enabled and cfg.mode == "generate":
        with llm.generate(
            {"input_ids": input_ids, "attention_mask": attn_mask} if attn_mask is not None else {"input_ids": input_ids},
            **_gen_args(),
        ) as tracer:
            # Register targets to save
            targets = _collect_targets(cfg.layers, cfg.hook_points)
            proxies: Dict[str, Any] = {}
            for name, node in targets:
                proxies[name] = node.save()

        # After context exits, model has generated outputs and proxies have values
        # NNsight exposes generator output at tracer.generator.output; fall back to tracer.output
        gen_out = getattr(tracer, "generator", None)
        if gen_out is not None and hasattr(gen_out, "output"):
            generated_ids = gen_out.output
        else:
            generated_ids = getattr(tracer, "output")
        # Some wrappers return (batch, seq); ensure indexing consistent
        if isinstance(generated_ids, (list, tuple)):
            generated_ids = generated_ids[0]
        # Write activations to disk
        run_dir = Path(run_dir_override) if run_dir_override else run_cfg.ensure_dirs()
        tensors: Dict[str, torch.Tensor] = {k: _extract_proxy_tensor(v) for k, v in proxies.items()}
        ndif_breadcrumb = write_activation_set(run_dir, request_id, tensors, run_cfg)
    else:
        # Plain generation using nnsight's generate() wrapper
        with llm.generate(
            {"input_ids": input_ids, "attention_mask": attn_mask} if attn_mask is not None else {"input_ids": input_ids},
            **_gen_args(),
        ) as tracer:
            pass
        gen_out = getattr(tracer, "generator", None)
        if gen_out is not None and hasattr(gen_out, "output"):
            generated_ids = gen_out.output
        else:
            generated_ids = getattr(tracer, "output")
        if isinstance(generated_ids, (list, tuple)):
            generated_ids = generated_ids[0]

        # Optionally run a secondary trace over prompt to collect activations
        if cfg.enabled and cfg.mode == "trace":
            with llm.trace({"input_ids": input_ids, "attention_mask": attn_mask} if attn_mask is not None else {"input_ids": input_ids}) as tracer:
                targets = _collect_targets(cfg.layers, cfg.hook_points)
                proxies: Dict[str, Any] = {}
                for name, node in targets:
                    proxies[name] = node.save()

            run_dir = Path(run_dir_override) if run_dir_override else run_cfg.ensure_dirs()
            tensors: Dict[str, torch.Tensor] = {k: _extract_proxy_tensor(v) for k, v in proxies.items()}
            ndif_breadcrumb = write_activation_set(run_dir, request_id, tensors, run_cfg)

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
    host = os.environ.get("NANO_NDIF_HOST", "0.0.0.0")
    port = int(os.environ.get("NANO_NDIF_PORT", "8002"))
    print(f"nano-ndif serving on http://{host}:{port} (model={MODEL_ID})")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
