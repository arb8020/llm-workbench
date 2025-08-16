import json
import time
from leyline import WorkerBase, expose

class ListWorker(WorkerBase):
    """Example worker that demonstrates list operations."""
    
    @expose(stream=False)
    def reverse_list(self, body: str):
        """Reverse a list provided in the request body."""
        try:
            data = json.loads(body) if body else {}
            items = data.get("items", [])
            return {"result": list(reversed(items))}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in request body"}
    
    @expose(stream=True)
    def stream_list(self, body: str):
        """Stream list items one by one with delay."""
        try:
            data = json.loads(body) if body else {}
            items = data.get("items", [])
            delay = data.get("delay", 0.5)
            
            for i, item in enumerate(items):
                time.sleep(delay)
                yield {"index": i, "item": item}
                
        except json.JSONDecodeError:
            yield {"error": "Invalid JSON in request body"}
    
    @expose(stream=False)
    def count_items(self, body: str):
        """Count items in a list."""
        try:
            data = json.loads(body) if body else {}
            items = data.get("items", [])
            return {"count": len(items)}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in request body"}
    
    @expose(stream=True)
    async def async_stream_numbers(self, body: str):
        """Async streaming example - generate numbers."""
        try:
            data = json.loads(body) if body else {}
            count = data.get("count", 10)
            delay = data.get("delay", 0.1)
            
            import asyncio
            for i in range(count):
                await asyncio.sleep(delay)
                yield {"number": i, "square": i * i}
                
        except json.JSONDecodeError:
            yield {"error": "Invalid JSON in request body"}