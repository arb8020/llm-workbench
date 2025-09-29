# Checkpoint storage for agent states

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Protocol

from dacite import from_dict, Config
from .dtypes import StopReason, AgentState, Actor, Environment


class CheckpointStore(Protocol):
    async def save(self, checkpoint_id: str, state: AgentState) -> None: ...
    async def load(self, checkpoint_id: str) -> AgentState: ...
    async def list(self) -> List[str]: ...


async def serialize_agent_state(state: AgentState) -> Dict[str, Any]:
    """Convert AgentState to JSON-serializable dict"""
    from dataclasses import asdict
    return {
        "actor": asdict(state.actor),  # asdict handles nested dataclasses!
        "environment": {
            "class_name": state.environment.__class__.__name__,
            "data": await state.environment.serialize()
        },
        "turn_idx": state.turn_idx,
        "max_turns": state.max_turns,
        "stop": state.stop.value if state.stop else None,
        "pending_tool_calls": [asdict(tc) for tc in state.pending_tool_calls],
        "next_tool_idx": state.next_tool_idx,
    }


async def deserialize_agent_state(
        data: Dict[str, Any],
        environment_registry: Dict[str, type[Environment]],
    ) -> AgentState:
    """Reconstruct AgentState from JSON-serializable dict"""
    # Handle environment separately
    env_class_name = data["environment"]["class_name"]
    env_class = environment_registry[env_class_name]
    environment = await env_class.deserialize(data["environment"]["data"])
    
    # Use dacite for the rest
    state_data = data.copy()
    state_data["environment"] = environment  # Replace with actual object
    state_data["actor"] = from_dict(Actor, data["actor"], Config(check_types=False))
    state_data["stop"] = StopReason(data["stop"]) if data["stop"] else None
    
    return from_dict(AgentState, state_data, Config(check_types=False))


class FileCheckpointStore:
    """File-based checkpoint storage for agent states"""

    def __init__(
        self,
        environment_registry: Dict[str, type[Environment]],
        directory: str = "/tmp/rollouts-agent-checkpoints",
    ):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True, parents=True)
        # Registry of environment classes for deserialization
        self.environment_registry = environment_registry
    
    async def save(self, checkpoint_id: str, state: AgentState) -> None:
        """Save state to JSON file"""
        data = await serialize_agent_state(state)
        
        # Add metadata
        data["_metadata"] = {
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "iso_time": datetime.now().isoformat(),
        }
        
        path = self.directory / f"{checkpoint_id}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def load(self, checkpoint_id: str) -> AgentState:
        """Load state from JSON file"""
        path = self.directory / f"{checkpoint_id}.json"
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Remove metadata before deserializing
        data.pop("_metadata", None)
        
        return await deserialize_agent_state(data, self.environment_registry)
    
    async def list(self) -> List[str]:
        """List all checkpoint IDs"""
        checkpoints = []
        for path in self.directory.glob("*.json"):
            checkpoints.append(path.stem)  # filename without extension
        return sorted(checkpoints)