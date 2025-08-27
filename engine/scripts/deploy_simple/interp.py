#!/usr/bin/env python3
"""FastAPI server with nnsight.VLLM integration for interpretability features."""

import os
import sys
import asyncio
import argparse
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


# ── Activation Collection Validation ────────────────────────────────────────────
class ModelStructureError(Exception):
    """Raised when model structure doesn't match expected patterns."""
    pass

class ActivationCollectionError(Exception):
    """Raised when activation collection fails."""
    pass

def introspect_model_structure(model) -> Dict[str, Any]:
    """Introspect model structure to understand available paths."""
    structure = {
        "has_model_attr": hasattr(model, 'model'),
        "has_transformer_attr": hasattr(model, 'transformer'),
        "model_type": type(model).__name__,
        "available_attributes": [],
        "layer_structure": None,
        "max_layers": None
    }
    
    # Get all available attributes
    structure["available_attributes"] = [attr for attr in dir(model) if not attr.startswith('_')]
    
    # Try to determine layer structure
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            if hasattr(model.model.transformer, 'h'):
                structure["layer_structure"] = "model.model.transformer.h"
                structure["max_layers"] = len(model.model.transformer.h)
        elif hasattr(model, 'transformer'):
            if hasattr(model.transformer, 'h'):
                structure["layer_structure"] = "model.transformer.h"  
                structure["max_layers"] = len(model.transformer.h)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            structure["layer_structure"] = "model.model.layers"
            structure["max_layers"] = len(model.model.layers)
    except Exception as e:
        structure["introspection_error"] = str(e)
    
    return structure


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

class ActivationCollectionStatus(BaseModel):
    """Metadata about activation collection success/failure."""
    requested: bool = False
    successful: bool = False
    requested_layers: List[int] = field(default_factory=list)
    requested_hooks: List[str] = field(default_factory=list)
    collected_keys: List[str] = field(default_factory=list)
    failed_keys: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    model_structure: Optional[Dict[str, Any]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    usage: Usage
    choices: List[Choice]
    activations: Optional[Dict[str, Any]] = None  # Interpretability extension
    activation_collection_status: Optional[ActivationCollectionStatus] = None  # Collection metadata



# ── Activation Collection Validation Functions ─────────────────────────────
def validate_activation_request(request: ActivationCollectionRequest, model) -> None:
    """Validate activation collection request against model capabilities."""
    errors = []
    
    # Introspect model structure
    structure = introspect_model_structure(model)
    
    if structure["max_layers"] is None:
        errors.append(f"Could not determine model layer structure. Available attributes: {structure['available_attributes'][:10]}")
    
    # Validate layer indices
    if structure["max_layers"]:
        for layer_idx in request.layers:
            if layer_idx < 0 or layer_idx >= structure["max_layers"]:
                errors.append(f"Layer {layer_idx} is out of range. Model has {structure['max_layers']} layers (0-{structure['max_layers']-1})")
    
    # Validate hook points
    supported_hooks = ["input", "output", "attn.output", "mlp.output"]
    for hook in request.hook_points:
        if hook not in supported_hooks:
            errors.append(f"Hook point '{hook}' not supported. Supported: {supported_hooks}")
    
    if errors:
        error_msg = "Activation collection validation failed:\n" + "\n".join(f"  • {err}" for err in errors)
        error_msg += f"\n\nModel structure:\n  • Type: {structure['model_type']}\n  • Layer structure: {structure['layer_structure']}\n  • Max layers: {structure['max_layers']}"
        raise ActivationCollectionError(error_msg)

def get_activation_tensor(model, layer_idx: int, hook_point: str):
    """Get activation tensor for specific layer and hook point with explicit error handling."""
    structure = introspect_model_structure(model)
    
    try:
        if hook_point == "output":
            if structure["layer_structure"] == "model.model.transformer.h":
                return model.model.transformer.h[layer_idx].output.save()
            elif structure["layer_structure"] == "model.transformer.h":
                return model.transformer.h[layer_idx].output.save()
            elif structure["layer_structure"] == "model.model.layers":
                return model.model.layers[layer_idx].output.save()
            else:
                raise ModelStructureError(f"Unknown layer structure: {structure['layer_structure']}")
                
        elif hook_point == "input":
            if structure["layer_structure"] == "model.model.transformer.h":
                return model.model.transformer.h[layer_idx].input.save()
            elif structure["layer_structure"] == "model.transformer.h":
                return model.transformer.h[layer_idx].input.save()
            elif structure["layer_structure"] == "model.model.layers":
                return model.model.layers[layer_idx].input.save()
            else:
                raise ModelStructureError(f"Unknown layer structure: {structure['layer_structure']}")
        else:
            raise ActivationCollectionError(f"Hook point '{hook_point}' not yet implemented")
            
    except IndexError as e:
        raise ActivationCollectionError(f"Layer {layer_idx} not accessible: {e}")
    except AttributeError as e:
        raise ActivationCollectionError(f"Attribute error accessing layer {layer_idx} {hook_point}: {e}")
    except Exception as e:
        raise ActivationCollectionError(f"Unexpected error collecting layer {layer_idx} {hook_point}: {type(e).__name__}: {e}")


# ── Global Model Instance ───────────────────────────────────────────────────
app = FastAPI(title="Interpretability-Enabled vLLM Server", version="0.1.0")
model = None
MODEL_NAME = "openai-community/gpt2"  # Default model, can be overridden by command line


def initialize_model():
    """Initialize nnsight VLLM model."""
    global model
    print("🧠 Loading model with nnsight.VLLM integration...")
    print(f"   Model: {MODEL_NAME}")
    print("   Features: activation collection, interventions")
    
    try:
        model = VLLM(MODEL_NAME, device="auto", dispatch=True)
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
                "id": MODEL_NAME,
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
        "supported_models": [MODEL_NAME],
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
                activations={},  # Empty dict when no activations collected
                activation_collection_status=ActivationCollectionStatus(requested=False)  # Not requested
            )
            
            return response
        
        # Enhanced path with activation collection
        else:
            # Initialize activation collection status tracking
            activation_status = ActivationCollectionStatus(
                requested=True,
                requested_layers=request.collect_activations.layers,
                requested_hooks=request.collect_activations.hook_points,
                model_structure=introspect_model_structure(model)
            )
            
            try:
                # Validate activation collection request first
                validate_activation_request(request.collect_activations, model)
                print(f"✅ Activation collection validation passed for layers {request.collect_activations.layers} hooks {request.collect_activations.hook_points}")
                
                layers = request.collect_activations.layers
                hook_points = request.collect_activations.hook_points
                
                collected_activations = {}
                failed_keys = []
                
                # First collect activations using nnsight trace
                with model.trace() as tracer:
                    with tracer.invoke(prompt):
                        # Collect requested activations with tuple handling (based on working example)
                        saved_activation_proxies = {}
                        
                        # Set up .save() calls inside trace context for nnsight
                        for layer_idx in layers:
                            for hook_point in hook_points:
                                key = f"layer_{layer_idx}_{hook_point}"
                                
                                try:
                                    # Get the activation proxy and call .save() INSIDE trace context
                                    if hook_point == "output":
                                        activation_proxy = model.model.layers[layer_idx].output.save()
                                    elif hook_point == "input":
                                        activation_proxy = model.model.layers[layer_idx].input.save()
                                    else:
                                        raise Exception(f"Hook point '{hook_point}' not yet implemented")
                                    
                                    saved_activation_proxies[key] = activation_proxy
                                    print(f"✅ Set up activation collection for {key}")
                                    
                                except Exception as e:
                                    activation_status.failed_keys.append(key)
                                    activation_status.error_message = str(e)
                                    print(f"❌ Failed to set up collection for {key}: {e}")
                                    raise HTTPException(status_code=400, detail=f"Activation collection setup failed for {key}: {str(e)}")
                        
                        # After trace context, extract actual tensor values from proxies
                        for key, activation_proxy in saved_activation_proxies.items():
                            try:
                                # Extract the value - handle different possible structures (TUPLE FIX)
                                if hasattr(activation_proxy, 'value'):
                                    value = activation_proxy.value
                                    # If it's a tuple (hidden_states, attention_weights, etc), take the first element
                                    if isinstance(value, tuple) and len(value) > 0:
                                        # For transformer outputs, first element is usually the hidden states tensor
                                        collected_activations[key] = value[0]
                                        print(f"✅ Extracted tensor from tuple for {key}: shape={getattr(value[0], 'shape', 'unknown')}")
                                    else:
                                        collected_activations[key] = value
                                        print(f"✅ Extracted value for {key}: type={type(value)}")
                                elif isinstance(activation_proxy, tuple) and len(activation_proxy) > 0:
                                    # nnsight+vLLM returns activation proxy as direct tuple
                                    print(f"🔍 TUPLE FIX {key}: tuple len={len(activation_proxy)}, item types={[type(item).__name__ for item in activation_proxy[:3]]}")
                                    tensor_item = activation_proxy[0]  # First item is usually hidden states
                                    collected_activations[key] = tensor_item
                                    print(f"✅ Extracted tensor from tuple for {key}: shape={getattr(tensor_item, 'shape', 'unknown')}, type={type(tensor_item)}")
                                else:
                                    # Fallback: use the proxy itself
                                    collected_activations[key] = activation_proxy  
                                    print(f"⚠️  Using proxy directly for {key}")
                                    
                                activation_status.collected_keys.append(key)
                                
                            except Exception as e:
                                activation_status.failed_keys.append(key)
                                activation_status.error_message = str(e)
                                print(f"❌ Failed to extract activation data for {key}: {e}"
                                    )
                
                # Check if we collected any activations
                if not collected_activations:
                    activation_status.successful = False
                    activation_status.error_message = "No activations were successfully collected"
                    raise HTTPException(
                        status_code=500,
                        detail="Activation collection completely failed - no activations collected"
                    )
                elif failed_keys:
                    # Partial failure - some activations collected but others failed
                    activation_status.successful = False
                    activation_status.error_message = f"Partial activation collection failure: {len(failed_keys)} failed, {len(collected_activations)} succeeded"
                    raise HTTPException(
                        status_code=400,
                        detail=activation_status.error_message
                    )
                else:
                    # Complete success
                    activation_status.successful = True
                    print(f"🎉 All activation collection successful: {len(collected_activations)} activations collected")
                    
            except ActivationCollectionError as e:
                activation_status.successful = False
                activation_status.error_message = str(e)
                print(f"❌ Activation collection validation failed: {e}")
                raise HTTPException(status_code=400, detail=f"Activation collection error: {str(e)}")
            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                activation_status.successful = False
                activation_status.error_message = f"Unexpected error: {str(e)}"
                print(f"❌ Unexpected activation collection error: {e}")
                raise HTTPException(status_code=500, detail=f"Unexpected activation collection error: {str(e)}")
            
            # Generate text using proper nnsight VLLM API for multi-token generation
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
            
            # Robust tensor serialization with comprehensive metadata (based on working example)
            import numpy as np
            import time
            serialized_activations = {}
            
            for key, activation_tensor in collected_activations.items():
                try:
                    print(f"🔍 Processing {key}: type={type(activation_tensor)}")
                    
                    # Extract the actual tensor from nnsight wrapper
                    actual_tensor = None
                    tensor_source = "unknown"
                    
                    # Try multiple ways to extract the underlying tensor
                    if hasattr(activation_tensor, 'value'):
                        actual_tensor = activation_tensor.value
                        tensor_source = "nnsight.value"
                        print(f"🔍 Extracted from nnsight.value: {type(actual_tensor)}")
                        
                        if hasattr(actual_tensor, 'detach'):
                            actual_tensor = actual_tensor.detach()
                            tensor_source += ".detach()"
                        if hasattr(actual_tensor, 'cpu'):
                            actual_tensor = actual_tensor.cpu()
                            tensor_source += ".cpu()"
                            
                    elif hasattr(activation_tensor, 'detach'):
                        actual_tensor = activation_tensor.detach().cpu()
                        tensor_source = "pytorch.detach().cpu()"
                    elif hasattr(activation_tensor, 'cpu'):
                        actual_tensor = activation_tensor.cpu()  
                        tensor_source = "pytorch.cpu()"
                    else:
                        actual_tensor = activation_tensor
                        tensor_source = "direct"
                    
                    print(f"🔍 Tensor source: {tensor_source}")
                    
                    # Now serialize the tensor with comprehensive metadata
                    if actual_tensor is not None and hasattr(actual_tensor, 'numpy'):
                        # Convert to numpy for serialization
                        import torch
                        if hasattr(actual_tensor, 'dtype') and actual_tensor.dtype == torch.bfloat16:
                            print(f"🔄 Converting BFloat16 to Float32 for {key} (numpy compatibility)")
                            numpy_array = actual_tensor.float().numpy()
                        else:
                            numpy_array = actual_tensor.numpy()
                        
                        # Comprehensive metadata
                        original_dtype = str(actual_tensor.dtype) if hasattr(actual_tensor, 'dtype') else "unknown"
                        shape = list(numpy_array.shape)
                        dtype_str = str(numpy_array.dtype)
                        total_size = int(numpy_array.size)
                        memory_mb = (numpy_array.nbytes / 1024 / 1024)
                        
                        # Statistical summary for large tensors
                        stats = None
                        if total_size > 100:
                            stats = {
                                "mean": float(numpy_array.mean()),
                                "std": float(numpy_array.std()),
                                "min": float(numpy_array.min()),
                                "max": float(numpy_array.max()),
                                "norm": float(np.linalg.norm(numpy_array.flatten()))
                            }
                        
                        # For TalkTuner demo, provide manageable amount of data
                        if total_size > 1000:
                            # Sample first 100 elements for demo
                            flat = numpy_array.flatten()
                            sample_data = flat[:100].tolist()
                            
                            serialized_activations[key] = {
                                "format": "sampled_tensor",
                                "shape": shape,
                                "dtype": dtype_str,
                                "original_dtype": original_dtype,
                                "total_elements": total_size,
                                "memory_mb": round(memory_mb, 3),
                                "statistics": stats,
                                "sample_data": sample_data,
                                "sample_info": "first_100_elements",
                                "tensor_source": tensor_source,
                                "blob_storage_ready": True
                            }
                        else:
                            # Small tensor - include all data
                            serialized_activations[key] = {
                                "format": "complete_tensor", 
                                "shape": shape,
                                "dtype": dtype_str,
                                "original_dtype": original_dtype,
                                "total_elements": total_size,
                                "memory_mb": round(memory_mb, 3),
                                "statistics": stats,
                                "data": numpy_array.tolist(),
                                "tensor_source": tensor_source,
                                "blob_storage_ready": True
                            }
                        
                        print(f"✅ Serialized {key}: {serialized_activations[key]['format']}, shape={shape}, size={total_size}, mem={memory_mb:.1f}MB")
                        
                    else:
                        # Fallback for non-numpy convertible tensors
                        shape = list(actual_tensor.shape) if hasattr(actual_tensor, 'shape') else "unknown"
                        
                        serialized_activations[key] = {
                            "format": "metadata_only",
                            "shape": shape,
                            "type": str(type(actual_tensor)),
                            "tensor_source": tensor_source,
                            "error": "Could not convert to numpy array for serialization",
                            "blob_storage_ready": False
                        }
                        print(f"⚠️  Could not serialize {key}: no numpy conversion available")
                        
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Error serializing {key}: {error_msg}")
                    serialized_activations[key] = {
                        "format": "error",
                        "error": f"Serialization failed: {error_msg}",
                        "type": str(type(activation_tensor)) if activation_tensor is not None else "None",
                        "blob_storage_ready": False
                    }
            
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
                activations=serialized_activations,  # Always include (may be empty dict)
                activation_collection_status=activation_status  # Include detailed status
            )
            
            return response
            
    except HTTPException:
        # Re-raise HTTP exceptions (like our 400 validation errors) as-is
        raise
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
    global MODEL_NAME
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Interpretability-Enabled vLLM Server")
    parser.add_argument("--model", default="openai-community/gpt2", 
                       help="Model to load (default: openai-community/gpt2)")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to run on (default: 8000)")
    
    args = parser.parse_args()
    MODEL_NAME = args.model
    
    print("🚀 Starting Interpretability-Enabled vLLM Server")
    print("🧠 Features: activation collection, OpenAI-compatible API")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    print(f"🎯 Model: {MODEL_NAME}")
    
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
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()