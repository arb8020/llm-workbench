"""Base classes for inference backends."""

import asyncio
import subprocess
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class BackendConfig:
    """Base configuration for inference backends."""
    model_name: str = "openai-community/gpt2"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Backend-specific arguments for future expansion
    backend_args: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def endpoint_url(self) -> str:
        """Get the backend's endpoint URL."""
        return f"http://{self.host}:{self.port}"


class InferenceBackend(ABC):
    """Base class for inference server backends."""
    
    backend_name: str = "base"
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._startup_timeout = 300  # 5 minutes for model loading
        
    @abstractmethod
    def build_command(self) -> List[str]:
        """Build the command to start the backend server."""
        pass
        
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend server is healthy and ready."""
        pass
    
    async def start(self) -> str:
        """Start the backend server and wait for it to be ready."""
        if self._process is not None:
            raise RuntimeError(f"{self.backend_name} backend already running")
            
        cmd = self.build_command()
        print(f"🚀 Starting {self.backend_name} with command: {' '.join(cmd)}")
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"⏳ Waiting for {self.backend_name} server to be ready...")
        await self._wait_for_ready()
        
        print(f"✅ {self.backend_name} server ready at {self.config.endpoint_url}")
        return self.config.endpoint_url
    
    async def stop(self):
        """Stop the backend server gracefully."""
        if self._process is None:
            return
            
        print(f"🛑 Stopping {self.backend_name} server...")
        self._process.terminate()
        
        try:
            # Wait up to 10 seconds for graceful shutdown
            await asyncio.wait_for(
                self._wait_for_process_end(), 
                timeout=10.0
            )
            print(f"✅ {self.backend_name} server stopped gracefully")
        except asyncio.TimeoutError:
            print(f"⚠️  {self.backend_name} server didn't stop gracefully, forcing...")
            self._process.kill()
            await self._wait_for_process_end()
            print(f"🔥 {self.backend_name} server force-stopped")
        
        self._process = None
    
    @property
    def is_running(self) -> bool:
        """Check if the backend process is still running."""
        return self._process is not None and self._process.poll() is None
    
    async def _wait_for_ready(self):
        """Wait for the backend to be ready to serve requests."""
        for i in range(self._startup_timeout):
            if self.health_check():
                return
                
            # Print progress every 30 seconds
            if i > 0 and i % 30 == 0:
                print(f"   Still waiting... ({i}s elapsed)")
                
            await asyncio.sleep(1)
            
        raise TimeoutError(
            f"{self.backend_name} server not ready after {self._startup_timeout}s"
        )
    
    async def _wait_for_process_end(self):
        """Wait for the process to terminate."""
        while self._process and self._process.poll() is None:
            await asyncio.sleep(0.1)