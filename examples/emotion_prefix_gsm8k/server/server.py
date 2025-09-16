#!/usr/bin/env python3
"""OpenAI chat wrapper using NNsight (copied for emotion_prefix_gsm8k).

Endpoints:
- GET /v1/models
- GET /v1/model/structure
- POST /v1/interventions
- POST /v1/chat/completions (non-streaming, allows unknown fields)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pathlib import Path
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
try:
    from pydantic import ConfigDict  # pydantic v2
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore
import uvicorn

from nnsight import LanguageModel

from .config import InterventionConfig, SUPPORTED_HOOK_POINTS
from .activation_capture import write_activation_set, new_request_id


DEFAULT_MODEL_ID = "willcb/Qwen3-0.6B"

app = FastAPI(title="emotion-prefix-ndif", version="0.1.0")
llm: LanguageModel | None = None
MODEL_ID: str = DEFAULT_MODEL_ID
DEVICE_MAP: str = "auto"
cfg_lock = threading.Lock()
cfg = InterventionConfig()
_hf_model = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL_ID)
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    ndif: Optional[Dict[str, Any]] = None  # per-request overrides

    # Allow unknown fields (rollouts may pass extras like logprobs/echo)
    if ConfigDict is not None:  # pydantic v2
        model_config = ConfigDict(extra="allow")  # type: ignore

    class Config:  # pydantic v1 fallback
        extra = "allow"


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
    ndif: Optional[Dict[str, Any]] = None


class InterventionsRequest(BaseModel):
    enabled: bool = True
    layers: Optional[Any] = None  # List[int] | "all"
    hook_points: List[str] = list(SUPPORTED_HOOK_POINTS)
    mode: str = "trace"
    save_dir: Optional[str] = None
    per_request_subdir: bool = True
    sample_hidden: Optional[int] = None
    save_format: str = "pt"
    include_metadata_json: bool = True


class InterventionsResponse(BaseModel):
    ok: bool
    config: Dict[str, Any]


def _load_model(model_id: str, device_map: str) -> LanguageModel:
    return LanguageModel(model_id, device_map=device_map)


@app.on_event("startup")
async def _startup():
    global llm
    llm = _load_model(MODEL_ID, DEVICE_MAP)
    # Log compact structure
    try:
        nl = len(llm.model.layers)
        print(f"[server] model={MODEL_ID} layers={nl} device_map={DEVICE_MAP}")
    except Exception:
        print(f"[server] model={MODEL_ID} device_map={DEVICE_MAP}")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 1677649963,
                "owned_by": "emotion-prefix-ndif",
            }
        ],
    }


@app.get("/v1/model/structure")
async def model_structure():
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        n_layers = len(llm.model.layers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read layers: {e}")
    return {
        "model": MODEL_ID,
        "device_map": DEVICE_MAP,
        "num_layers": n_layers,
        "hook_points": list(SUPPORTED_HOOK_POINTS),
    }


@app.get("/health")
async def health():
    return {"ok": llm is not None, "model": MODEL_ID}


class ModelLoadRequest(BaseModel):
    model: str
    device_map: Optional[str] = None


@app.post("/v1/model")
async def load_model(req: ModelLoadRequest):
    global llm, MODEL_ID, DEVICE_MAP, _hf_model
    model_id = req.model
    device_map = req.device_map or DEVICE_MAP
    try:
        new_llm = _load_model(model_id, device_map)
        old_llm = llm
        llm = new_llm
        MODEL_ID = model_id
        DEVICE_MAP = device_map
        _hf_model = None
        del old_llm
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # Validate interventions wrt new depth
        valid = True
        note = None
        try:
            n_layers = len(llm.model.layers)
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
        except Exception:
            pass
        return {"ok": True, "model": MODEL_ID, "device_map": DEVICE_MAP, "interventions_valid": valid, "interventions_note": note}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {e}")


@app.post("/v1/interventions")
async def set_interventions(body: InterventionsRequest):
    global cfg
    # Resolve layers
    resolved_layers: List[int] = []
    try:
        n_layers = len(llm.model.layers) if llm is not None else None
    except Exception:
        n_layers = None
    if isinstance(body.layers, str) and body.layers.lower() == "all":
        if n_layers is None:
            raise HTTPException(status_code=500, detail="Cannot resolve 'all' layers without model depth")
        resolved_layers = list(range(n_layers))
    elif isinstance(body.layers, list) and all(isinstance(x, int) for x in body.layers or []):
        resolved_layers = body.layers or []
    else:
        # Keep previous if not provided
        resolved_layers = cfg.layers

    new_cfg = InterventionConfig(
        enabled=body.enabled,
        layers=resolved_layers,
        hook_points=body.hook_points or cfg.hook_points,
        mode=body.mode or cfg.mode,
        save_dir=Path(body.save_dir) if body.save_dir else cfg.save_dir,
        per_request_subdir=body.per_request_subdir,
        sample_hidden=body.sample_hidden,
        save_format=body.save_format,
        include_metadata_json=body.include_metadata_json,
    )
    try:
        new_cfg.validate(n_layers=n_layers)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    with cfg_lock:
        cfg = new_cfg
    return {"ok": True, "config": new_cfg.to_dict()}


def to_hf_chat(messages: List[Message]) -> List[Dict[str, str]]:
    role_map = {"system": "system", "user": "user", "assistant": "assistant"}
    return [{"role": role_map.get(m.role, m.role), "content": m.content} for m in messages]


def render_prompt(messages: List[Message]) -> Dict[str, Any]:
    assert llm is not None
    chat = to_hf_chat(messages)
    tok = llm.tokenizer
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        try:
            prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt_text, return_tensors="pt")
            return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask")}
        except Exception:
            pass
    # fallback: join with role prefixes
    def _render(m: Dict[str, str]) -> str:
        return f"{m['role']}: {m['content']}"
    prompt_text = "\n".join(_render(m) for m in chat) + "\nassistant:"
    enc = llm.tokenizer(prompt_text, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask")}


def render_prompt_text(messages: List[Message]) -> str:
    chat = to_hf_chat(messages)
    if hasattr(llm.tokenizer, "apply_chat_template") and getattr(llm.tokenizer, "chat_template", None):
        try:
            return llm.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return "\n".join(f"{m['role']}: {m['content']}" for m in chat) + "\nassistant:"


def _collect_targets(layers: List[int], hook_points: List[str]):
    targets = []
    for i in layers:
        for p in hook_points:
            try:
                node = getattr(llm.model.layers[i], p.replace(".", "_"))
                targets.append((f"layer_{i}_{p}", node))
            except Exception:
                pass
    return targets


def _extract_proxy_tensor(proxy) -> torch.Tensor:
    t = None
    if hasattr(proxy, "value"):
        t = proxy.value
    else:
        try:
            t = proxy.detach()
        except Exception:
            pass
    if not isinstance(t, torch.Tensor):
        raise RuntimeError("Could not resolve proxy to tensor")
    return t


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with cfg_lock:
        local_cfg = cfg.clone()

    enc = render_prompt(req.messages)
    prompt_text = render_prompt_text(req.messages)
    input_ids = enc["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None and not isinstance(attn_mask, torch.Tensor):
        attn_mask = torch.tensor(attn_mask)
    dev = getattr(llm, "device", None)
    dev_str = str(dev) if dev is not None else None
    if dev_str and dev_str != "meta":
        input_ids = input_ids.to(dev)
        if attn_mask is not None:
            attn_mask = attn_mask.to(dev)

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
        do_sample = True
        temperature = req.temperature
        if temperature is None or temperature <= 0:
            do_sample = False
            temperature = 1.0
        return {"max_new_tokens": req.max_tokens, "temperature": temperature, "top_p": req.top_p, "do_sample": do_sample}

    generated_ids = None
    # Plain generate via HF, trace via NNsight when enabled
    from transformers import AutoModelForCausalLM
    global _hf_model
    if _hf_model is None:
        _hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    toks = llm.tokenizer(prompt_text, return_tensors="pt")
    input_ids_hf = toks["input_ids"].to(_hf_model.device)
    do_sample = True
    temperature = req.temperature
    if temperature is None or temperature <= 0:
        do_sample = False
        temperature = 1.0
    out = _hf_model.generate(
        input_ids_hf,
        max_new_tokens=req.max_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=llm.tokenizer.eos_token_id,
    )
    generated_ids = out

    if local_cfg.enabled and local_cfg.mode in {"trace", "generate"}:
        proxies: Dict[str, Any] = {}
        with llm.trace(prompt_text) as tracer:
            targets = _collect_targets(local_cfg.layers, local_cfg.hook_points)
            for name, node in targets:
                proxies[name] = node.save()
        base_dir = Path(run_dir_override) if run_dir_override is not None else local_cfg.save_dir
        use_subdir = per_request_subdir_override if per_request_subdir_override is not None else local_cfg.per_request_subdir
        if use_subdir:
            from time import strftime
            run_dir = base_dir / f"run-{strftime('%Y%m%d-%H%M%S')}"
        else:
            run_dir = base_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        import torch as _t
        tensors: Dict[str, _t.Tensor] = {k: _extract_proxy_tensor(v) for k, v in proxies.items()}
        ndif_breadcrumb = write_activation_set(run_dir, request_id, tensors, local_cfg)

    text = llm.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    try:
        rendered_text = llm.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if text.startswith(rendered_text):
            text = text[len(rendered_text):].strip()
    except Exception:
        pass

    prompt_tokens = int(input_ids.numel())
    completion_tokens = int(generated_ids.shape[-1] - input_ids.shape[-1]) if generated_ids is not None else 0
    created = int(time.time())

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=created,
        model=req.model,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens),
        choices=[Choice(index=0, message=Message(role="assistant", content=text), finish_reason="stop")],
        ndif=ndif_breadcrumb,
    )


def main():
    import argparse
    global MODEL_ID, DEVICE_MAP
    ap = argparse.ArgumentParser(description="emotion_prefix_gsm8k server")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device-map", default=DEVICE_MAP)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    MODEL_ID = args.model
    DEVICE_MAP = args.device_map
    print(f"emotion-prefix-ndif serving on http://{args.host}:{args.port} (model={MODEL_ID})")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

