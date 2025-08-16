import json
import time
from leyline import WorkerBase, expose

class EchoWorker(WorkerBase):
    """Simple echo worker for testing."""
    
    @expose(stream=False)
    def echo(self, body: str):
        """Echo the request body back."""
        return {"echo": body, "timestamp": time.time()}
    
    @expose(stream=False)
    def health(self, body: str):
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": time.time()}
    
    @expose(stream=True)
    def stream_echo(self, body: str):
        """Stream the request body character by character."""
        for i, char in enumerate(body):
            time.sleep(0.1)
            yield {"position": i, "character": char}
    
    @expose(stream=False)
    def uppercase(self, body: str):
        """Convert request body to uppercase."""
        try:
            data = json.loads(body) if body else {}
            text = data.get("text", body)
            return {"result": text.upper()}
        except json.JSONDecodeError:
            return {"result": body.upper()}