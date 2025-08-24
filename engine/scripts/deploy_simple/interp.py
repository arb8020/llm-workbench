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
                        # Collect requested activations with explicit error handling
                        for layer_idx in layers:
                            for hook_point in hook_points:
                                key = f"layer_{layer_idx}_{hook_point}"
                                
                                try:
                                    activation_tensor = get_activation_tensor(model, layer_idx, hook_point)
                                    collected_activations[key] = activation_tensor
                                    activation_status.collected_keys.append(key)
                                    print(f"✅ Successfully collected {key}")
                                except (ActivationCollectionError, ModelStructureError) as e:
                                    failed_keys.append(key)
                                    activation_status.failed_keys.append(key)
                                    error_msg = f"❌ Failed to collect {key}: {str(e)}"
                                    print(error_msg)
                                    # For any activation collection failure, raise HTTP error
                                    activation_status.error_message = str(e)
                                    raise HTTPException(
                                        status_code=400, 
                                        detail=f"Activation collection failed for {key}: {str(e)}"
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
            
            # Process collected activations for JSON serialization with explicit error handling
            serialized_activations = {}
            serialization_errors = []
            
            for key, activation_tensor in collected_activations.items():
                try:
                    if activation_tensor is None:
                        serialization_errors.append(f"Activation tensor for {key} is None")
                        continue
                        
                    # Convert to list for JSON serialization (limit size for demo)
                    if hasattr(activation_tensor, 'tolist'):
                        try:
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
                            print(f"✅ Successfully serialized {key} (shape: {getattr(activation_tensor, 'shape', 'unknown')})")
                        except Exception as e:
                            serialization_errors.append(f"Failed to convert {key} to list: {e}")
                            # Fallback to shape information
                            serialized_activations[key] = {
                                "error": f"Serialization failed: {str(e)}",
                                "shape": list(activation_tensor.shape) if hasattr(activation_tensor, 'shape') else "unknown",
                                "type": str(type(activation_tensor))
                            }
                    else:
                        serialization_errors.append(f"Activation tensor for {key} has no tolist() method")
                        serialized_activations[key] = {
                            "error": "No tolist() method available",
                            "shape": list(activation_tensor.shape) if hasattr(activation_tensor, 'shape') else "unknown",
                            "type": str(type(activation_tensor))
                        }
                        
                except Exception as e:
                    serialization_errors.append(f"Unexpected error serializing {key}: {e}")
                    serialized_activations[key] = {
                        "error": f"Serialization error: {str(e)}",
                        "type": str(type(activation_tensor)) if activation_tensor is not None else "None"
                    }
            
            # Report serialization errors but don't fail the request
            if serialization_errors:
                print(f"⚠️  Activation serialization warnings: {len(serialization_errors)} issues")
                for error in serialization_errors[:5]:  # Show first 5 errors
                    print(f"   • {error}")
                if len(serialization_errors) > 5:
                    print(f"   • ... and {len(serialization_errors) - 5} more errors")
            
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