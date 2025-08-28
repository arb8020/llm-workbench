"""Custom inference server utilities."""

from .openai_compat import (
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequest, 
    ChatCompletionResponse,
    setup_openai_routes,
    create_completion_response,
    create_chat_completion_response
)

__all__ = [
    "CompletionRequest",
    "CompletionResponse", 
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "setup_openai_routes",
    "create_completion_response", 
    "create_chat_completion_response"
]