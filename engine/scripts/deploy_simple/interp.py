#!/usr/bin/env python3
"""FastAPI server with nnsight.VLLM integration for interpretability features."""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

# FastAPI and response types
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# nnsight for interpretability
try:
    from nnsight.modeling.vllm import VLLM
except ImportError:
    print("❌ nnsight not installed or incompatible vLLM version. Install with: pip install 'nnsight>=0.4' 'vllm==0.9.2'")
    sys.exit(1)


# ── Request/Response Models ──────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ActivationCollectionRequest(BaseModel):
    layers: List[int] = field(default_factory=list)
    hook_points: List[str] = field(default_factory=lambda: ["output"])
    positions: List[int] = field(default_factory=lambda: [-1])  # -1 = last token
    return_activations: bool = True

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 50
    temperature: float = 0.1
    top_p: float = 1.0
    collect_activations: Optional[ActivationCollectionRequest] = None

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
    activations: Optional[Dict[str, Any]] = None  # Interpretability extension


# ── Global Model Instance ───────────────────────────────────────────────────
app = FastAPI(title="Interpretability-Enabled vLLM Server", version="0.1.0")
model = None


def initialize_model():
    """Initialize nnsight VLLM model."""
    global model
    print("🧠 Loading model with nnsight.VLLM integration...")
    print("   Model: openai-community/gpt2")
    print("   Features: activation collection, interventions")
    
    try:
        model = VLLM("openai-community/gpt2", device="auto", dispatch=True)
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


# ── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "openai-community/gpt2",
                "object": "model",
                "created": 1677649963,
                "owned_by": "openai-community"
            }
        ]
    }

@app.get("/v1/capabilities")
async def get_capabilities():
    """Interpretability-specific capabilities endpoint."""
    return {
        "activation_collection": True,
        "activation_patching": True,
        "supported_models": ["openai-community/gpt2"],
        "supported_layers": list(range(12)),  # GPT-2 has 12 layers
        "supported_hook_points": ["input", "output", "attn.output", "mlp.output"],
        "max_tokens": 512,
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions with optional activation collection."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract prompt from messages (simple concatenation for now)
        prompt = ""
        for message in request.messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n"
            elif message.role == "user":
                prompt += f"Human: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        
        if not prompt.endswith("Assistant: "):
            prompt += "Assistant: "
        
        # Standard inference path (no activation collection)
        if not request.collect_activations:
            # Use proper nnsight VLLM API for multi-token text generation
            all_logits = []
            with model.trace(
                [prompt],
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            ) as tracer:
                # Collect logits for all generated tokens
                with tracer.iter[0:request.max_tokens]:
                    all_logits.append(model.logits.output.save())
            
            # Decode all generated tokens to text
            try:
                # Get the generated token IDs for all steps
                generated_token_ids = []
                for logits in all_logits:
                    token_id = logits.argmax(dim=-1)
                    # Convert to Python int for tokenizer
                    generated_token_ids.append(int(token_id.item()))
                
                # Decode the generated tokens using the token IDs list
                if generated_token_ids:
                    generated_text = model.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                else:
                    generated_text = ""
                    
                print(f"🔍 Debug - Generated {len(generated_token_ids)} tokens: {generated_token_ids} -> '{generated_text}'")
                
            except Exception as e:
                print(f"❌ Error decoding generated text: {e}")
                # Add more debug info
                print(f"🔍 Debug - all_logits length: {len(all_logits)}")
                if all_logits:
                    print(f"🔍 Debug - first logits shape: {all_logits[0].shape if hasattr(all_logits[0], 'shape') else 'no shape'}")
                raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
            
            # Clean up the response (remove prompt)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{hash(prompt) % 100000}",
                created=1677649963,
                model=request.model,
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(generated_text.split()),
                    total_tokens=len(prompt.split()) + len(generated_text.split())
                ),
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=generated_text),
                        finish_reason="stop"
                    )
                ],
                activations={}  # Empty dict when no activations collected
            )
            
            return response
        
        # Enhanced path with activation collection
        else:
            layers = request.collect_activations.layers
            hook_points = request.collect_activations.hook_points
            
            collected_activations = {}
            
            # First collect activations using nnsight trace
            with model.trace() as tracer:
                with tracer.invoke(prompt):
                    # Collect requested activations
                    for layer_idx in layers:
                        for hook_point in hook_points:
                            key = f"layer_{layer_idx}_{hook_point}"
                            
                            try:
                                if hook_point == "output":
                                    # Try different model paths based on nnsight-vLLM structure
                                    if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
                                        collected_activations[key] = model.model.transformer.h[layer_idx].output.save()
                                    elif hasattr(model, 'transformer'):
                                        collected_activations[key] = model.transformer.h[layer_idx].output.save()
                                    else:
                                        # Try to find the layers through model structure
                                        collected_activations[key] = model.model.layers[layer_idx].output.save()
                                elif hook_point == "input":
                                    if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
                                        collected_activations[key] = model.model.transformer.h[layer_idx].input.save()
                                    elif hasattr(model, 'transformer'):
                                        collected_activations[key] = model.transformer.h[layer_idx].input.save()
                                    else:
                                        collected_activations[key] = model.model.layers[layer_idx].input.save()
                                # Add more hook points as needed
                            except Exception as e:
                                print(f"⚠️  Failed to collect {key}: {e}")
            
            # Generate text using proper nnsight VLLM API for multi-token generation
            with model.trace(
                [prompt],
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            ) as tracer:
                # Collect logits for all generated tokens
                all_logits = []
                with tracer.iter[0:request.max_tokens]:
                    all_logits.append(model.logits.output.save())
            
            # Decode all generated tokens to text
            try:
                # Get the generated token IDs for all steps
                generated_token_ids = []
                for logits in all_logits:
                    token_id = logits.argmax(dim=-1)
                    # Convert to Python int for tokenizer
                    generated_token_ids.append(int(token_id.item()))
                
                # Decode the generated tokens using the token IDs list
                if generated_token_ids:
                    generated_text = model.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                else:
                    generated_text = ""
                    
                print(f"🔍 Debug - Generated {len(generated_token_ids)} tokens (with activations): {generated_token_ids} -> '{generated_text}'")
                
            except Exception as e:
                print(f"❌ Error decoding generated text (with activations): {e}")
                # Add more debug info
                print(f"🔍 Debug - all_logits length (with activations): {len(all_logits)}")
                if all_logits:
                    print(f"🔍 Debug - first logits shape (with activations): {all_logits[0].shape if hasattr(all_logits[0], 'shape') else 'no shape'}")
                raise HTTPException(status_code=500, detail=f"Text generation with activations failed: {str(e)}")
            
            # Clean up the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Process collected activations for JSON serialization
            serialized_activations = {}
            for key, activation_tensor in collected_activations.items():
                if activation_tensor is not None:
                    # Convert to list for JSON serialization (limit size for demo)
                    if hasattr(activation_tensor, 'tolist'):
                        # Limit to first 10 elements to avoid huge JSON responses
                        activation_list = activation_tensor.tolist()
                        if isinstance(activation_list, list) and len(activation_list) > 0:
                            # If it's a nested list, take first 10 of the last dimension
                            if isinstance(activation_list[0], list):
                                serialized_activations[key] = [row[:10] if isinstance(row, list) else row 
                                                             for row in activation_list]
                            else:
                                serialized_activations[key] = activation_list[:10]
                        else:
                            serialized_activations[key] = activation_list
                    else:
                        serialized_activations[key] = f"tensor_shape_{list(activation_tensor.shape) if hasattr(activation_tensor, 'shape') else 'unknown'}"
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{hash(prompt) % 100000}",
                created=1677649963,
                model=request.model,
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(generated_text.split()),
                    total_tokens=len(prompt.split()) + len(generated_text.split())
                ),
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=generated_text),
                        finish_reason="stop"
                    )
                ],
                activations=serialized_activations if serialized_activations else None  # Include activation data
            )
            
            return response
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    success = initialize_model()
    if not success:
        print("❌ Failed to initialize model - server will not function properly")


def main():
    """Start the interpretability server."""
    print("🚀 Starting Interpretability-Enabled vLLM Server")
    print("🧠 Features: activation collection, OpenAI-compatible API")
    print("📡 Server will be available at: http://0.0.0.0:8000")
    print("📚 Documentation: http://0.0.0.0:8000/docs")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️  No GPU detected - model will run on CPU (very slow)")
    except ImportError:
        print("⚠️  PyTorch not available - cannot check GPU status")
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()