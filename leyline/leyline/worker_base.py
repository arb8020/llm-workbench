import asyncio
import logging
import json
import inspect
from typing import Any, Callable, Dict, Optional, AsyncGenerator, Union
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from .common import parse_frame, build_chunk_frame

logger = logging.getLogger(__name__)

def expose(stream: bool = False):
    """Decorator to expose a worker method for RPC calls."""
    def decorator(func: Callable) -> Callable:
        func._exposed = True
        func._stream = stream
        return func
    return decorator

class WorkerBase:
    """Base class for workers that connect to the leyline controller."""
    
    def __init__(self, controller_url: str, ping_interval: float = 20.0, ping_timeout: float = 10.0):
        self.controller_url = controller_url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._exposed_methods: Dict[str, Callable] = {}
        self._register_exposed_methods()
    
    def _register_exposed_methods(self):
        """Find and register all @expose decorated methods."""
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_exposed') and method._exposed:
                self._exposed_methods[name] = method
                logger.info(f"Registered exposed method: {name} (stream={getattr(method, '_stream', False)})")
    
    async def run(self, max_retries: int = -1, backoff_base: float = 1.0):
        """Connect to controller and handle requests with retry logic."""
        retry_count = 0
        
        while max_retries < 0 or retry_count <= max_retries:
            try:
                logger.info(f"Connecting to controller at {self.controller_url}")
                async with websockets.connect(
                    self.controller_url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout
                ) as websocket:
                    self.websocket = websocket
                    logger.info("Connected to controller")
                    retry_count = 0  # Reset on successful connection
                    
                    async for message in websocket:
                        try:
                            await self._handle_message(message)
                        except Exception as e:
                            logger.error(f"Error handling message: {e}")
                            
            except (ConnectionClosed, WebSocketException) as e:
                retry_count += 1
                backoff = backoff_base * (2 ** min(retry_count - 1, 6))
                logger.warning(f"Connection lost ({e}), retrying in {backoff}s (attempt {retry_count})")
                await asyncio.sleep(backoff)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        
        logger.error("Max retries exceeded or fatal error, exiting")
    
    
    async def _handle_message(self, raw_message: str):
        """Handle incoming WebSocket message from controller."""
        try:
            logger.info(f"Received message: {raw_message}")
            frame = parse_frame(raw_message)
            
            if frame.get("t") != "req":
                logger.warning(f"Unknown frame type: {frame.get('t')}")
                return
            
            rid = frame["rid"]
            method_name = frame["p"].lstrip("/")  # Remove leading slash
            body = frame["b"]
            
            if method_name not in self._exposed_methods:
                await self._send_error(rid, f"Method '{method_name}' not found")
                return
            
            method = self._exposed_methods[method_name]
            is_stream = getattr(method, '_stream', False)
            
            logger.info(f"Method: {method_name}, is_stream: {is_stream}")
            
            if is_stream:
                await self._handle_streaming_method(rid, method, body)
            else:
                await self._handle_single_method(rid, method, body)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            if 'rid' in locals():
                await self._send_error(rid, str(e))
    
    async def _handle_streaming_method(self, rid: str, method: Callable, body: str):
        """Handle streaming method that yields chunks."""
        try:
            if inspect.isasyncgenfunction(method):
                # Handle async generator functions
                logger.info("Processing async generator function")
                async for chunk in method(body):
                    logger.info(f"Yielding chunk: {chunk}")
                    await self._send_chunk(rid, chunk, done=False)
            elif asyncio.iscoroutinefunction(method):
                # Handle regular async functions
                logger.info("Processing regular async function")
                result = await method(body)
                await self._send_chunk(rid, result, done=False)
            else:
                # Run sync generator in thread
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, method, body)
                if hasattr(result, '__iter__'):
                    for chunk in result:
                        await self._send_chunk(rid, chunk, done=False)
                else:
                    await self._send_chunk(rid, result, done=False)
            
            await self._send_chunk(rid, None, done=True)
            
        except Exception as e:
            logger.error(f"Error in streaming method: {e}")
            await self._send_error(rid, str(e))
    
    async def _handle_single_method(self, rid: str, method: Callable, body: str):
        """Handle single-response method."""
        try:
            logger.info(f"_handle_single_method called for: {method.__name__}")
            if inspect.isasyncgenfunction(method):
                logger.error(f"Async generator {method.__name__} was misrouted to single method handler!")
                await self._send_error(rid, f"Method {method.__name__} should be handled as streaming")
                return
            elif asyncio.iscoroutinefunction(method):
                result = await method(body)
            else:
                # Run sync method in thread
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, method, body)
            
            await self._send_chunk(rid, result, done=True)
            
        except Exception as e:
            logger.error(f"Error in single method: {e}")
            await self._send_error(rid, str(e))
    
    async def _send_chunk(self, rid: str, chunk: Any, done: bool):
        """Send a chunk frame to the controller."""
        if self.websocket:
            frame = build_chunk_frame(rid, chunk, done)
            await self.websocket.send(frame)
    
    async def _send_error(self, rid: str, error_message: str):
        """Send an error as the final chunk."""
        await self._send_chunk(rid, {"error": error_message}, done=True)