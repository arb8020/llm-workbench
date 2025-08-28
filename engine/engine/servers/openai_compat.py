"""OpenAI API compatibility layer for custom inference servers."""

import time
from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# OpenAI API Request Models (Base - extend for custom parameters)
class BaseCompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False


class ChatMessage(BaseModel):
    role: str
    content: str


class BaseChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False


# Convenience aliases for standard usage
CompletionRequest = BaseCompletionRequest
ChatCompletionRequest = BaseChatCompletionRequest


# OpenAI API Response Models
class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict] = None
    finish_reason: str


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int
    logprobs: Optional[Dict] = None
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


def setup_openai_routes(
    app: FastAPI, 
    model_name: str, 
    generate_fn: Callable[[str, Dict[str, Any]], str],
    owner: str = "custom"
) -> None:
    """
    Setup standard OpenAI-compatible routes on FastAPI app.
    
    Args:
        app: FastAPI application instance
        model_name: Model identifier to return in responses
        generate_fn: Function that takes (prompt, params) and returns generated text
        owner: Model owner identifier for /v1/models endpoint
    """
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model", 
                    "created": int(time.time()),
                    "owned_by": owner
                }
            ]
        }
    
    @app.post("/v1/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest):
        try:
            # Extract generation parameters
            gen_params = {
                "max_tokens": request.max_tokens or 100,
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
                "n": request.n or 1
            }
            
            # Generate text using provided function (handle both sync and async)
            import asyncio
            if asyncio.iscoroutinefunction(generate_fn):
                generated_text = await generate_fn(request.prompt, gen_params)
            else:
                generated_text = generate_fn(request.prompt, gen_params)
            
            # Create response
            return create_completion_response(
                text=generated_text,
                model=model_name,
                prompt_tokens=len(request.prompt.split()),  # Rough estimate
                completion_tokens=len(generated_text.split())  # Rough estimate
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        try:
            # Convert chat messages to single prompt
            prompt_parts = []
            for msg in request.messages:
                if msg.role == "system":
                    prompt_parts.append(f"System: {msg.content}")
                elif msg.role == "user":
                    prompt_parts.append(f"User: {msg.content}")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}")
            
            full_prompt = "\n".join(prompt_parts) + "\nAssistant:"
            
            # Extract generation parameters
            gen_params = {
                "max_tokens": request.max_tokens or 100,
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
                "n": request.n or 1
            }
            
            # Generate text using provided function (handle both sync and async)
            import asyncio
            if asyncio.iscoroutinefunction(generate_fn):
                generated_text = await generate_fn(full_prompt, gen_params)
            else:
                generated_text = generate_fn(full_prompt, gen_params)
            
            # Create response
            return create_chat_completion_response(
                text=generated_text,
                model=model_name,
                prompt_tokens=len(full_prompt.split()),  # Rough estimate
                completion_tokens=len(generated_text.split())  # Rough estimate
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def create_completion_response(text: str, model: str, prompt_tokens: int, completion_tokens: int) -> CompletionResponse:
    """Create OpenAI-compatible completion response."""
    return CompletionResponse(
        id=f"cmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=model,
        choices=[
            CompletionChoice(
                text=text,
                index=0,
                logprobs=None,
                finish_reason="stop"
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )


def create_chat_completion_response(text: str, model: str, prompt_tokens: int, completion_tokens: int) -> ChatCompletionResponse:
    """Create OpenAI-compatible chat completion response.""" 
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=text),
                index=0,
                logprobs=None,
                finish_reason="stop"
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )