#!/usr/bin/env python3
"""Proper nnsight OpenAI API server with Qwen3 and residual stream extraction."""

import os
import sys
import argparse

# Add current directory to Python path (for imports)
sys.path.insert(0, os.path.dirname(__file__))

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import torch

# FastAPI and response types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import nnsight directly for better control
from nnsight import LanguageModel

# Request/Response Models 
class Message(BaseModel):
    role: str
    content: str

class ActivationCollectionRequest(BaseModel):
    layers: List[int] = field(default_factory=list)
    hook_points: List[str] = field(default_factory=lambda: ["input_layernorm.output", "post_attention_layernorm.output"])
    return_activations: bool = False  # Default to remote storage

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

# Global Model Instance
app = FastAPI(title="Proper nnsight API Server", version="0.1.0")
llm_model = None
MODEL_NAME = "willcb/Qwen3-0.6B"  # Use the same model as activation_collection backend

def initialize_model():
    """Initialize nnsight LanguageModel directly."""
    global llm_model
    print(f"🧠 Loading Qwen3 model: {MODEL_NAME}")
    
    try:
        llm_model = LanguageModel(MODEL_NAME, device_map="auto")
        print("✅ Qwen3 model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

def messages_to_prompt(messages: List[Message]) -> str:
    """Convert chat messages to a proper Qwen3 prompt."""
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

def extract_residual_stream_activations(prompt: str, layers: List[int], hook_points: List[str], max_tokens: int, return_activations: bool = True, session_id: str = None) -> tuple[str, Dict[str, Any]]:
    """Extract residual stream activations using proper nnsight patterns."""
    
    # Generate tokens first to get the text using nnsight's generate method
    with torch.inference_mode():
        # Use nnsight's generate method instead of the underlying model's generate
        generated_text = llm_model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=llm_model.tokenizer.eos_token_id
        )
        
        # nnsight.generate() returns the full text including prompt, so remove it
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
    
    # Now extract activations using the outlier_features_moe pattern
    activations = {}
    texts = [prompt]  # Use the original prompt for activation extraction
    
    with torch.inference_mode():
        with llm_model.trace(texts) as tracer:
            for layer_idx in layers:
                for hook_point in hook_points:
                    key = f"layer_{layer_idx}_{hook_point.replace('.', '_')}"
                    
                    # NOTE: Hook point paths are currently HARDCODED for standard transformer architecture
                    # This assumes: llm_model.model.layers[i] structure with .input_layernorm/.post_attention_layernorm
                    # For multi-architecture support, see examples/outlier_features_moe/layer_analysis_results.json
                    # which provides dynamic path detection for different model types (Qwen3, OLMoE, GPT-OSS, etc.)
                    if hook_point == "input_layernorm.output":
                        # Pre-attention residual stream  
                        activation_proxy = llm_model.model.layers[layer_idx].input_layernorm.output.save()
                        activations[key] = activation_proxy
                    elif hook_point == "post_attention_layernorm.output":
                        # Pre-MLP residual stream
                        activation_proxy = llm_model.model.layers[layer_idx].post_attention_layernorm.output.save()
                        activations[key] = activation_proxy
                    else:
                        print(f"⚠️ Unknown hook point: {hook_point}")
                        print("💡 To support new architectures, see examples/outlier_features_moe for dynamic path detection")
    
    # Handle activations based on return_activations flag
    if not return_activations:
        # Store activations on remote GPU/disk instead of returning them
        import uuid
        import os
        
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        # Create remote activations directory
        activations_dir = "/tmp/nnsight_activations"
        os.makedirs(activations_dir, exist_ok=True)
        
        # Store raw tensors (keep on GPU for faster access)
        stored_activations = {}
        for key, activation_proxy in activations.items():
            try:
                tensor = activation_proxy.detach()  # Keep on GPU if possible
                stored_path = f"{activations_dir}/activations_{session_id}_{key}.pt"
                
                # Save tensor to disk but also keep metadata
                torch.save(tensor, stored_path)
                stored_activations[key] = {
                    "format": "stored_tensor",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "stored_path": stored_path,
                    "session_id": session_id
                }
                print(f"✅ Stored {key}: shape={tuple(tensor.shape)} at {stored_path}")
                
            except Exception as e:
                print(f"❌ Error storing {key}: {e}")
                stored_activations[key] = {
                    "format": "error",
                    "error": str(e)
                }
        
        return generated_text, {
            "activation_storage": {
                "session_id": session_id,
                "storage_dir": activations_dir,
                "stored_activations": stored_activations,
                "total_activations": len(stored_activations)
            }
        }
    
    else:
        # Original behavior: serialize and return activations
        serialized_activations = {}
        for key, activation_proxy in activations.items():
            try:
                tensor = activation_proxy.detach().cpu()
                
                # Convert to regular Python lists for JSON serialization
                if tensor.dim() == 3:  # [batch, seq_len, hidden_dim]
                    # For large tensors, take first 50 elements along hidden dimension
                    if tensor.shape[-1] > 50:
                        tensor_sample = tensor[:, :, :50]  # Sample first 50 features
                        serialized_activations[key] = {
                            "format": "sampled_tensor",
                            "shape": list(tensor.shape), 
                            "dtype": str(tensor.dtype),
                            "data": tensor_sample.tolist(),
                            "sample_info": "first_50_features"
                        }
                    else:
                        serialized_activations[key] = {
                            "format": "complete_tensor",
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "data": tensor.tolist()
                        }
                else:
                    serialized_activations[key] = {
                        "format": "complete_tensor", 
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "data": tensor.tolist()
                    }
                    
                print(f"✅ Extracted {key}: shape={tuple(tensor.shape)}")
                
            except Exception as e:
                print(f"❌ Error processing {key}: {e}")
                serialized_activations[key] = {
                    "format": "error",
                    "error": str(e)
                }
        
        return generated_text, serialized_activations

# API Endpoints
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
                "owned_by": "willcb"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Simple health check."""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True, "model_name": MODEL_NAME}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions with residual stream activation collection."""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert messages to prompt
        prompt = messages_to_prompt(request.messages)
        print(f"🔍 Processing prompt: {prompt[:100]}...")
        
        # Standard generation (no activation collection)
        if not request.collect_activations:
            with torch.inference_mode():
                # Use nnsight's generate method for consistency
                generated_text = llm_model.generate(
                    prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    pad_token_id=llm_model.tokenizer.eos_token_id
                )
                
                # Remove prompt from generated text
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
                activations=None
            )
            
            return response
        
        # Generation with residual stream activation collection
        else:
            layers = request.collect_activations.layers or [8, 12, 16]  # Default middle layers for Qwen3-0.6B
            hook_points = request.collect_activations.hook_points
            return_activations = request.collect_activations.return_activations
            
            # Generate session ID for activation storage
            import uuid
            session_id = str(uuid.uuid4())[:8]
            
            generated_text, activations = extract_residual_stream_activations(
                prompt, layers, hook_points, request.max_tokens, 
                return_activations, session_id
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
                activations=activations
            )
            
            return response
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/v1/activations/sessions")
async def list_activation_sessions():
    """List all stored activation sessions."""
    import os
    import glob
    
    activations_dir = "/tmp/nnsight_activations"
    if not os.path.exists(activations_dir):
        return {"sessions": []}
    
    # Find all unique session IDs from stored files
    pattern = f"{activations_dir}/activations_*_*.pt"
    files = glob.glob(pattern)
    
    sessions = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract session_id from filename: activations_{session_id}_{key}.pt
        parts = filename.replace("activations_", "").replace(".pt", "").split("_")
        if len(parts) >= 2:
            session_id = parts[0]
            key = "_".join(parts[1:])
            
            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "activation_files": [],
                    "file_count": 0,
                    "total_size_mb": 0
                }
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            sessions[session_id]["activation_files"].append({
                "key": key,
                "file_path": file_path,
                "size_mb": round(file_size, 2)
            })
            sessions[session_id]["file_count"] += 1
            sessions[session_id]["total_size_mb"] += file_size
    
    for session in sessions.values():
        session["total_size_mb"] = round(session["total_size_mb"], 2)
    
    return {"sessions": list(sessions.values())}

@app.get("/v1/activations/session/{session_id}")
async def get_activation_session(session_id: str):
    """Get detailed info about a specific activation session."""
    import os
    import glob
    
    activations_dir = "/tmp/nnsight_activations"
    pattern = f"{activations_dir}/activations_{session_id}_*.pt"
    files = glob.glob(pattern)
    
    if not files:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session_info = {
        "session_id": session_id,
        "storage_dir": activations_dir,
        "activations": []
    }
    
    for file_path in files:
        filename = os.path.basename(file_path)
        key = filename.replace(f"activations_{session_id}_", "").replace(".pt", "")
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        # Try to load tensor metadata without loading full tensor
        try:
            tensor = torch.load(file_path, map_location='cpu', weights_only=True)
            session_info["activations"].append({
                "key": key,
                "file_path": file_path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "size_mb": round(file_size, 2)
            })
        except Exception as e:
            session_info["activations"].append({
                "key": key,
                "file_path": file_path,
                "size_mb": round(file_size, 2),
                "error": str(e)
            })
    
    return session_info

@app.delete("/v1/activations/session/{session_id}")
async def delete_activation_session(session_id: str):
    """Delete all activation files for a session."""
    import os
    import glob
    
    activations_dir = "/tmp/nnsight_activations"
    pattern = f"{activations_dir}/activations_{session_id}_*.pt"
    files = glob.glob(pattern)
    
    if not files:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    deleted_files = []
    for file_path in files:
        try:
            os.remove(file_path)
            deleted_files.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return {
        "session_id": session_id,
        "deleted_files": deleted_files,
        "total_deleted": len(deleted_files)
    }

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    success = initialize_model()
    if not success:
        print("❌ Failed to initialize model - server will not function properly")

def main():
    """Start the proper nnsight API server."""
    global MODEL_NAME
    
    parser = argparse.ArgumentParser(description="Proper nnsight OpenAI API Server")
    parser.add_argument("--model", default="willcb/Qwen3-0.6B", 
                       help="Model to load (default: willcb/Qwen3-0.6B)")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, 
                       help="Port to run on (default: 8001)")
    
    args = parser.parse_args()
    MODEL_NAME = args.model
    
    print("🚀 Starting Proper nnsight OpenAI API Server")
    print("🧠 Features: Qwen3 model, residual stream activation collection")
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