import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import websockets
from websockets.exceptions import ConnectionClosed
import uvicorn

from .common import generate_rid, build_request_frame, parse_frame

logger = logging.getLogger(__name__)

class Controller:
    """Central controller that manages worker connections and HTTP API."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, ws_port: Optional[int] = None):
        self.host = host
        self.port = port
        self.ws_port = ws_port or (port + 1)  # WebSocket on port + 1 by default
        self.app = FastAPI(title="Leyline Controller")
        
        # Worker management
        self.workers: Set[websockets.WebSocketServerProtocol] = set()
        self.request_queues: Dict[str, asyncio.Queue] = {}
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI HTTP routes."""
        
        @self.app.post("/{method}")
        async def proxy_request(method: str, request: Request):
            return await self._handle_http_request(method, request)
    
    async def _handle_worker_connection(self, websocket):
        """Handle new worker WebSocket connection."""
        self.workers.add(websocket)
        logger.info(f"Worker connected. Total workers: {len(self.workers)}")
        
        try:
            async for message in websocket:
                await self._handle_worker_message(message)
        except ConnectionClosed:
            logger.info("Worker disconnected")
        except Exception as e:
            if "disconnected" in str(e).lower():
                logger.info("Worker disconnected")
            else:
                logger.error(f"Error in worker connection: {e}")
        finally:
            self.workers.discard(websocket)
            logger.info(f"Worker removed. Total workers: {len(self.workers)}")
    
    async def _handle_worker_message(self, raw_message: str):
        """Handle message from worker."""
        try:
            logger.info(f"Received from worker: {raw_message}")
            frame = parse_frame(raw_message)
            
            if frame.get("t") == "chunk":
                rid = frame["rid"]
                if rid in self.request_queues:
                    await self.request_queues[rid].put(frame)
                else:
                    logger.warning(f"Received chunk for unknown request ID: {rid}")
            else:
                logger.warning(f"Unknown frame type from worker: {frame.get('t')}")
                
        except Exception as e:
            logger.error(f"Error processing worker message: {e}")
    
    async def _handle_http_request(self, method: str, request: Request):
        """Handle HTTP request and proxy to worker."""
        if not self.workers:
            raise HTTPException(status_code=503, detail="No workers available")
        
        # Get request body
        body = await request.body()
        body_str = body.decode('utf-8') if body else ""
        
        # Get headers
        headers = dict(request.headers)
        
        # Generate request ID and create queue
        rid = generate_rid()
        self.request_queues[rid] = asyncio.Queue()
        
        try:
            # Select a worker (simple round-robin for now)
            worker = next(iter(self.workers))
            
            # Send request to worker
            req_frame = build_request_frame(
                rid=rid,
                method=request.method,
                path=f"/{method}",
                headers=headers,
                body=body_str
            )
            
            await worker.send(req_frame)
            
            # Check if client wants streaming response
            accept_header = request.headers.get("accept", "")
            stream_requested = "text/event-stream" in accept_header or request.query_params.get("stream") == "true"
            
            if stream_requested:
                return StreamingResponse(
                    self._stream_response(rid),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                try:
                    result = await self._buffer_response(rid)
                    return result
                finally:
                    # Clean up request queue for buffered responses
                    self.request_queues.pop(rid, None)
                
        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
            # Clean up on error
            self.request_queues.pop(rid, None)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _stream_response(self, rid: str):
        """Stream response chunks as Server-Sent Events."""
        queue = self.request_queues.get(rid)
        if not queue:
            logger.error(f"No queue found for request {rid}")
            yield f"event: error\ndata: Request not found\n\n"
            return
        
        try:
            while True:
                # Wait for chunk from worker
                chunk_frame = await asyncio.wait_for(queue.get(), timeout=30.0)
                
                chunk_data = chunk_frame.get("c")
                is_done = chunk_frame.get("done", False)
                
                if chunk_data is not None:
                    # Send as SSE
                    if isinstance(chunk_data, dict):
                        data = json.dumps(chunk_data)
                    else:
                        data = str(chunk_data)
                    
                    yield f"data: {data}\n\n"
                
                if is_done:
                    break
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response chunks for request {rid}")
            yield f"event: error\ndata: Request timeout\n\n"
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            # Clean up request queue after streaming is complete
            self.request_queues.pop(rid, None)
    
    async def _buffer_response(self, rid: str):
        """Buffer all response chunks and return as single response."""
        queue = self.request_queues[rid]
        result_chunks = []
        
        try:
            while True:
                # Wait for chunk from worker
                chunk_frame = await asyncio.wait_for(queue.get(), timeout=30.0)
                
                chunk_data = chunk_frame.get("c")
                is_done = chunk_frame.get("done", False)
                
                if chunk_data is not None:
                    result_chunks.append(chunk_data)
                
                if is_done:
                    break
            
            # Return single result or array of chunks
            if len(result_chunks) == 1:
                return result_chunks[0]
            else:
                return result_chunks
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response for request {rid}")
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            logger.error(f"Error buffering response: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_websocket_server(self):
        """Run the WebSocket server for worker connections."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.ws_port}")
        
        async with websockets.serve(
            self._handle_worker_connection,
            self.host,
            self.ws_port,
            ping_interval=20,
            ping_timeout=10
        ):
            # Keep the server running
            await asyncio.Future()  # Run forever
    
    def run(self, **kwargs):
        """Run both the HTTP and WebSocket servers."""
        logger.info(f"Starting Leyline Controller")
        logger.info(f"HTTP API: http://{self.host}:{self.port}")
        logger.info(f"Worker WebSocket: ws://{self.host}:{self.ws_port}")
        
        async def run_servers():
            # Start WebSocket server
            ws_task = asyncio.create_task(self._run_websocket_server())
            
            # Start HTTP server
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                **kwargs
            )
            server = uvicorn.Server(config)
            http_task = asyncio.create_task(server.serve())
            
            # Wait for both servers
            await asyncio.gather(ws_task, http_task)
        
        asyncio.run(run_servers())