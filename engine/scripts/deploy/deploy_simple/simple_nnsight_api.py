#!/usr/bin/env python3
"""Simple OpenAI-compatible API server using direct nnsight model loading.

Bypasses vLLM version conflicts by using nnsight.LanguageModel directly.
Provides activation collection via the existing ActivationCollector backend.
"""

import os
import sys
import argparse
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# FastAPI and response types
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Use existing activation collection backend
try:
    from engine.backends.interpretability.activation_collection import (
        ActivationCollector, ActivationConfig, ActivationRequest
    )
except ImportError:
    print("❌ Cannot import ActivationCollector from engine backend")
    sys.exit(1)

# ── Request/Response Models ──────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ActivationCollectionRequest(BaseModel):
    layers: List[int] = field(default_factory=list)
    hook_points: List[str] = field(default_factory=lambda: ["output"])
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
    activations: Optional[Dict[str, Any]] = None

# ── Global Model Instance ───────────────────────────────────────────────────
app = FastAPI(title="Simple nnsight API Server", version="0.1.0")
activation_collector = None
MODEL_NAME = "openai-community/gpt2"

def initialize_model():
    """Initialize ActivationCollector with direct nnsight model."""
    global activation_collector
    print(f"🧠 Loading model with direct nnsight: {MODEL_NAME}")
    
    try:
        config = ActivationConfig(
            model_name=MODEL_NAME,
            target_layers=[6, 8, 10],  # Some middle layers for GPT-2
            device_map="auto"
        )
        activation_collector = ActivationCollector(config)
        activation_collector.load_model()
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

# ── Helper Functions ─────────────────────────────────────────────────────────
def messages_to_prompt(messages: List[Message]) -> str:
    """Convert chat messages to a simple prompt."""
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"System: {message.content}\n"
        elif message.role == "user":
            prompt += f"Human: {message.content}\n"
        elif message.role == "assistant":
            prompt += f"Assistant: {message.content}\n"
    
    if not prompt.endswith("Assistant: "):
        prompt += "Assistant: "
    
    return prompt

# ── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1677649963,
                "owned_by": "openai-community"
            }
        ]
    }

@app.get("/v1/capabilities")
async def get_capabilities():
    """Activation collection capabilities."""
    if activation_collector is None:
        return {"activation_collection": False}
    
    capabilities = activation_collector.get_capabilities()
    return {
        "activation_collection": True,
        "supported_models": [MODEL_NAME],
        "supported_layers": capabilities.get("supported_layers", []),
        "supported_hook_points": capabilities.get("supported_hook_points", []),
        "max_tokens": 512,
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    """Simple health check."""
    if activation_collector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions with optional activation collection."""
    if activation_collector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert messages to prompt
        prompt = messages_to_prompt(request.messages)
        
        # Standard generation (no activation collection)
        if not request.collect_activations:
            generated_text = activation_collector.generate_standard(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
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
                activations=None
            )
            
            return response
        
        # Generation with activation collection
        else:
            activation_request = ActivationRequest(
                layers=request.collect_activations.layers or [6, 8, 10],
                hook_points=request.collect_activations.hook_points
            )
            
            result = activation_collector.generate_with_activations(
                prompt=prompt,
                activation_request=activation_request,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            generated_text = result["generated_text"]
            activations = result["activations"]
            
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
                activations=activations
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
    """Start the simple nnsight API server."""
    global MODEL_NAME
    
    parser = argparse.ArgumentParser(description="Simple nnsight OpenAI API Server")
    parser.add_argument("--model", default="openai-community/gpt2", 
                       help="Model to load (default: openai-community/gpt2)")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to run on (default: 8000)")
    
    args = parser.parse_args()
    MODEL_NAME = args.model
    
    print("🚀 Starting Simple nnsight OpenAI API Server")
    print("🧠 Features: activation collection, OpenAI-compatible API")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"🎯 Model: {MODEL_NAME}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  No GPU detected - model will run on CPU (very slow)")
    except ImportError:
        print("⚠️  PyTorch not available - cannot check GPU status")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()