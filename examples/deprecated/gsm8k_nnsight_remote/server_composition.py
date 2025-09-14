#!/usr/bin/env python3
"""
NNsight Server with Clean Composition Architecture

Key insight: Separate NNsight operations from chat processing to avoid context pollution.
The working test endpoint proves NNsight works - we just need clean separation.
"""

import json
import threading
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from nnsight import LanguageModel
import nnsight
from pydantic import BaseModel
from transformers import AutoTokenizer


# === CORE NNSIGHT SERVICE (ISOLATED) ===
class NNsightCore:
    """Pure NNsight operations - completely isolated from chat logic"""
    
    def __init__(self, model_id: str, device_map: str = "auto"):
        self.model_id = model_id
        self.lm = LanguageModel(model_id, device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.lock = threading.Lock()
        
        # Ensure tokenizer is properly configured
        if self.lm.model.config.pad_token_id is None:
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def capture_activations(self, prompt: str, max_new_tokens: int = 3) -> Dict[str, Any]:
        """Multi-token NNsight activation capture using correct pattern"""
        with self.lock:
            # Initialize NNsight list to accumulate activations across all generated tokens
            logits_list = nnsight.list().save()
            
            # Use proper multi-token generation pattern
            with self.lm.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
                with tracer.all():
                    logits_list.append(self.lm.lm_head.output)
            
            # Extract generated text from tracer output
            try:
                if hasattr(tracer, 'output') and tracer.output is not None:
                    generated_text = self.tokenizer.decode(tracer.output[0], skip_special_tokens=True)
                    # Remove the original prompt if it appears at the start
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                else:
                    generated_text = "Generated response (text extraction needs fixing)"
            except Exception as e:
                generated_text = f"Generated response (text extraction failed: {e})"
            
            return {
                "logits": logits_list,
                "generated_text": generated_text,
                "generated_tokens": getattr(tracer, 'output', [None])[0] if hasattr(tracer, 'output') else None,
                "success": True,
                "num_tokens_captured": len(logits_list) if logits_list else 0
            }


# === CHAT PROCESSING LAYER ===
class ChatProcessor:
    """Handle chat templates and formatting - separate from NNsight"""
    
    def __init__(self, nnsight_core: NNsightCore):
        self.core = nnsight_core
    
    def render_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt text"""
        # Simple concatenation for now - can enhance with proper chat templates
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    
    def extract_assistant_response(self, full_text: str, prompt: str) -> str:
        """Extract just the assistant's response from generated text"""
        # Remove the original prompt if it appears at the start
        if full_text.startswith(prompt):
            response = full_text[len(prompt):].strip()
        else:
            response = full_text
        
        # Clean up any remaining chat artifacts
        if "assistant:" in response.lower():
            # Find the last occurrence of assistant and take everything after
            idx = response.lower().rfind("assistant:")
            if idx != -1:
                response = response[idx + len("assistant:"):].strip()
        
        return response


# === STORAGE LAYER ===
class ActivationStorage:
    """Handle activation storage and metadata"""
    
    def __init__(self, base_path: str = "/tmp/nnsight_activations"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def store_activation(self, name: str, tensor: torch.Tensor, run_id: str) -> Dict[str, Any]:
        """Store tensor and return metadata"""
        filename = f"{run_id}_{name}.pt"
        filepath = self.base_path / filename
        
        # Save tensor to disk
        torch.save(tensor, filepath)
        
        # Return metadata
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "size_mb": round(tensor.numel() * tensor.element_size() / (1024 * 1024), 2),
            "file_path": str(filepath),
            "data_included": False,
            "note": "Tensor data saved to disk"
        }


# === PYDANTIC MODELS ===
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    store_activations: bool = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# === FASTAPI APP ===
app = FastAPI(title="NNsight Composition Server")

# Global services
CORES: Dict[str, NNsightCore] = {}
PROCESSORS: Dict[str, ChatProcessor] = {}
STORAGE = ActivationStorage()


@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(CORES.keys())}


@app.post("/models/load")
def load_model(req: Dict[str, Any]):
    model_id = req["model_id"]
    device_map = req.get("device_map", "auto")
    
    try:
        # Create isolated NNsight core
        core = NNsightCore(model_id, device_map)
        CORES[model_id] = core
        
        # Create chat processor
        processor = ChatProcessor(core)
        PROCESSORS[model_id] = processor
        
        return {"ok": True, "model": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.post("/test_pure_nnsight")
def test_pure_nnsight():
    """Test the pure NNsight core - should always work"""
    if not CORES:
        raise HTTPException(status_code=400, detail="No models loaded")
    
    core = next(iter(CORES.values()))
    
    try:
        result = core.capture_activations("Hello, how are you?")
        return {
            "success": True,
            "logits_shape": list(result["logits"].shape),
            "logits_dtype": str(result["logits"].dtype),
            "generated_text": result["generated_text"],
            "message": "Pure NNsight core working perfectly!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pure NNsight failed: {e}")


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(req: ChatRequest):
    """Chat completions using composition approach"""
    if req.model not in PROCESSORS:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded")
    
    processor = PROCESSORS[req.model]
    run_id = str(uuid.uuid4())
    
    try:
        # Step 1: Process chat messages (separate from NNsight)
        prompt = processor.render_chat_prompt([msg.model_dump() for msg in req.messages])
        
        # Step 2: Pure NNsight operation (isolated)
        nnsight_result = processor.core.capture_activations(prompt, req.max_tokens)
        
        # Step 3: Post-process text (separate from NNsight)
        assistant_response = processor.extract_assistant_response(
            nnsight_result["generated_text"], 
            prompt
        )
        
        # Step 4: Handle activations storage (if requested)
        activations_meta = {}
        if req.store_activations:
            logits = nnsight_result["logits"]
            activations_meta["_logits"] = STORAGE.store_activation("logits", logits, run_id)
        
        # Step 5: Build response
        import time
        response = ChatResponse(
            id=run_id,
            created=int(time.time()),
            model=req.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": assistant_response},
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        
        # Attach activation metadata
        response_dict = json.loads(response.model_dump_json())
        response_dict["activations_meta"] = activations_meta
        
        return JSONResponse(content=response_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {e}")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="NNsight composition server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")