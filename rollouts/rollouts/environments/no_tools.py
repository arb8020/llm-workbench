"""
BasicEnvironment - Simple environment with no tools for clean AI conversations.

This is useful for single-shot analysis tasks where you want the AI to provide
text responses without access to calculator, search, or other tools that might
confuse the conversation.

Example usage:
    from rollouts.environments import BasicEnvironment
    environment = BasicEnvironment()
"""

from typing import List
from ..dtypes import Tool, Environment

class BasicEnvironment(Environment):
    """
    Simple environment with no tools - just for clean AI responses.
    
    This environment provides no tools to the AI agent, making it suitable for:
    - Analysis tasks
    - Text generation
    - Single-shot conversations
    - Any scenario where tools would be distracting
    
    The AI will receive prompts and generate text responses without access to
    external tools or functions.
    """
    
    def get_tools(self) -> List[Tool]:
        """Return empty tool list - no tools available."""
        return []
    
    async def serialize(self) -> dict:
        """Serialize environment state (empty for this simple environment)."""
        return {}
    
    @staticmethod
    async def deserialize(data: dict) -> 'BasicEnvironment':
        """Deserialize environment state."""
        return BasicEnvironment()

# Backward compatibility alias
NoToolsEnvironment = BasicEnvironment