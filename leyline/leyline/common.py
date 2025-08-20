import uuid
import json
from typing import Any, Dict

# Constants
HEARTBEAT = 30.0  # seconds (legacy, now using ping/pong)

def generate_rid() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())

def build_request_frame(rid: str, method: str, path: str, headers: Dict[str, Any], body: str) -> str:
    """Build a request frame to send to worker."""
    frame = {
        "t": "req",
        "rid": rid,
        "m": method,
        "p": path,
        "h": headers,
        "b": body
    }
    return json.dumps(frame)

def build_chunk_frame(rid: str, chunk: Any, done: bool = False) -> str:
    """Build a chunk frame for streaming responses."""
    frame = {
        "t": "chunk",
        "rid": rid,
        "c": chunk,
        "done": done
    }
    return json.dumps(frame)

def parse_frame(raw_message: str) -> Dict[str, Any]:
    """Parse a JSON frame from WebSocket message."""
    return json.loads(raw_message)