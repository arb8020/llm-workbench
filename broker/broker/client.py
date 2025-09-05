"""
GPU Broker Client - Main interface for GPU operations
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv

from . import api
from .query import GPUQuery, QueryType
from .types import CloudType, GPUInstance, GPUOffer


class GPUClient:
    """Main client for GPU broker operations
    
    Handles configuration for API keys, SSH keys, and provides
    all GPU search, provisioning, and management functionality.
    
    Examples:
        # Basic usage with environment variables
        client = GPUClient()
        
        # Explicit configuration
        client = GPUClient(
            ssh_key_path="~/.ssh/my_runpod_key",
            api_key="your-runpod-api-key"
        )
        
        # Use the client
        offers = client.search(client.gpu_type.contains("A100"))
        instance = client.create(offers[:3])
    """
    
    def __init__(
        self, 
        ssh_key_path: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize GPU broker client
        
        Args:
            ssh_key_path: Path to SSH private key, or None for auto-discovery
            api_key: RunPod API key, or None to use RUNPOD_API_KEY env var
        """
        self._ssh_key_path = None
        self._api_key = None
        
        # Configure SSH key
        if ssh_key_path:
            self.set_ssh_key_path(ssh_key_path)
        else:
            self._discover_ssh_key()
        
        # Configure API key
        if api_key:
            self._api_key = api_key
        else:
            # Load .env file if it exists
            load_dotenv()
            self._api_key = os.environ.get('RUNPOD_API_KEY')
            if not self._api_key:
                raise ValueError(
                    "RunPod API key required. Set via:\n"
                    "  client = GPUBrokerClient(api_key='your-key')\n"
                    "  export RUNPOD_API_KEY=your-key\n"
                    "  Or add RUNPOD_API_KEY=your-key to .env file"
                )
        
        # Set up query interface
        self._query = GPUQuery()
    
    def set_ssh_key_path(self, path: str):
        """Set SSH private key path
        
        Args:
            path: Path to SSH private key file
        """
        expanded_path = os.path.expanduser(path)
        if not os.path.exists(expanded_path):
            raise ValueError(f"SSH key not found: {expanded_path}")
        self._ssh_key_path = expanded_path
    
    def get_ssh_key_path(self) -> str:
        """Get configured SSH key path"""
        if not self._ssh_key_path:
            raise ValueError("No SSH key configured")
        return self._ssh_key_path
    
    def _discover_ssh_key(self):
        """Auto-discover SSH key from common locations"""
        # Load .env file if it exists
        load_dotenv()
        
        # Check environment variable first
        env_key = os.environ.get('GPU_BROKER_SSH_KEY')
        if env_key:
            self.set_ssh_key_path(env_key)
            return
        
        # Try common key locations
        for key_name in ["id_ed25519", "id_rsa", "id_ecdsa"]:
            key_path = os.path.expanduser(f"~/.ssh/{key_name}")
            if os.path.exists(key_path):
                self._ssh_key_path = key_path
                return
        
        raise ValueError(
            "No SSH key found. Set one via:\n"
            "  client.set_ssh_key_path('~/.ssh/id_ed25519')\n"
            "  export GPU_BROKER_SSH_KEY=~/.ssh/id_ed25519\n"
            "  Or ensure you have a key at ~/.ssh/id_ed25519"
        )
    
    def validate_configuration(self) -> Dict[str, str]:
        """Validate configuration and return status report
        
        Returns:
            Dict with validation results for API key and SSH key
        """
        results = {}
        
        # Validate API key
        if self._api_key and self._api_key != "your_runpod_api_key_here":
            if self._api_key.startswith("rpa_") and len(self._api_key) > 20:
                results["api_key"] = "✅ Valid RunPod API key format"
            else:
                results["api_key"] = "⚠️  API key format may be invalid"
        else:
            results["api_key"] = "❌ No valid API key (check .env file)"
        
        # Validate SSH key
        if self._ssh_key_path and os.path.exists(self._ssh_key_path):
            # Check key permissions (should be 600 or 400)
            perms = oct(os.stat(self._ssh_key_path).st_mode)[-3:]
            if perms in ["600", "400"]:
                results["ssh_key"] = f"✅ SSH key found with secure permissions ({perms})"
            else:
                results["ssh_key"] = f"⚠️  SSH key found but permissions too open ({perms}). Run: chmod 600 {self._ssh_key_path}"
        else:
            results["ssh_key"] = "❌ No SSH key found"
        
        return results
    
    # Query interface - expose as properties
    @property
    def gpu_type(self):
        """Query by GPU type: client.gpu_type.contains('A100')"""
        return self._query.gpu_type
    
    @property 
    def price_per_hour(self):
        """Query by price: client.price_per_hour < 2.0"""
        return self._query.price_per_hour
    
    @property
    def memory_gb(self):
        """Query by memory: client.memory_gb > 24"""
        return self._query.memory_gb
    
    @property
    def vram_gb(self):
        """Query by GPU VRAM: client.vram_gb >= 8"""
        return self._query.vram_gb
    
    @property
    def cloud_type(self):
        """Query by cloud type: client.cloud_type == CloudType.SECURE"""
        return self._query.cloud_type
    
    @property
    def provider(self):
        """Query by provider: client.provider == 'runpod'"""
        return self._query.provider
    
    @property
    def cuda_version(self):
        """Query by CUDA version: client.cuda_version.contains('12.0')"""
        return self._query.cuda_version
    
    # Main API methods
    def search(
        self,
        query: Optional[QueryType] = None,
        # Legacy parameters  
        gpu_type: Optional[str] = None,
        max_price_per_hour: Optional[float] = None,
        provider: Optional[str] = None,
        cuda_version: Optional[str] = None,
        manufacturer: Optional[str] = None,
        # Sorting
        sort: Optional[Callable[[Any], Any]] = None,
        reverse: bool = False
    ) -> List[GPUOffer]:
        """Search for GPU offers
        
        Args:
            query: Pandas-style query (e.g., client.gpu_type.contains("A100"))
            sort: Sort function (e.g., lambda x: x.memory_gb / x.price_per_hour)
            reverse: Sort descending
            
        Returns:
            List of GPU offers
        """
        return api.search(
            query=query,
            gpu_type=gpu_type,
            max_price_per_hour=max_price_per_hour,
            provider=provider,
            cuda_version=cuda_version,
            manufacturer=manufacturer,
            sort=sort,
            reverse=reverse
        )
    
    def create(
        self,
        query: Union[QueryType, List[GPUOffer], GPUOffer] = None,
        image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        name: Optional[str] = None,
        # Search parameters
        gpu_type: Optional[str] = None,
        max_price_per_hour: Optional[float] = None,
        cloud_type: Optional[Union[str, CloudType]] = None,
        provider: Optional[str] = None,
        cuda_version: Optional[str] = None,
        sort: Optional[Callable[[Any], Any]] = None,
        reverse: bool = False,
        # Port exposure configuration
        exposed_ports: Optional[List[int]] = None,
        enable_http_proxy: bool = True,
        # Retry parameters
        max_attempts: int = 3,
        **kwargs
    ) -> Optional['ClientGPUInstance']:
        """Create GPU instance
        
        Args:
            cloud_type: Cloud deployment type ("secure", "community", or CloudType enum)
            exposed_ports: List of ports to expose via HTTP proxy (e.g., [8000] for vLLM)
            enable_http_proxy: Enable RunPod's HTTP proxy for exposed ports
            
        Returns:
            GPU instance with client configuration
        """
        # Build query conditions from convenience parameters (like CLI does)
        query_conditions = []
        
        if max_price_per_hour is not None:
            query_conditions.append(self.price_per_hour < max_price_per_hour)
        
        if gpu_type is not None:
            query_conditions.append(self.gpu_type.contains(gpu_type))
            
        if cloud_type is not None:
            # Convert string to enum if needed
            if isinstance(cloud_type, str):
                cloud_enum = CloudType.SECURE if cloud_type.lower() == "secure" else CloudType.COMMUNITY
            else:
                cloud_enum = cloud_type
            query_conditions.append(self.cloud_type == cloud_enum)
        
        # Merge with existing query if provided
        final_query = query
        if query_conditions:
            conditions_query = query_conditions[0]
            for condition in query_conditions[1:]:
                conditions_query = conditions_query & condition
                
            if final_query is not None:
                final_query = final_query & conditions_query
            else:
                final_query = conditions_query
        
        instance = api.create(
            query=final_query,
            image=image,
            name=name,
            # Don't pass individual search params if we built a query
            gpu_type=None if query_conditions else gpu_type,
            max_price_per_hour=None if query_conditions else max_price_per_hour,
            provider=provider,
            cuda_version=cuda_version,
            sort=sort,
            reverse=reverse,
            exposed_ports=exposed_ports,
            enable_http_proxy=enable_http_proxy,
            max_attempts=max_attempts,
            **kwargs
        )
        
        if instance:
            # Return wrapped instance with client configuration
            return ClientGPUInstance(instance, self)
        return None
    
    def get_instance(self, instance_id: str, provider: Optional[str] = None) -> Optional['ClientGPUInstance']:
        """Get instance details"""
        instance = api.get_instance(instance_id, provider)
        if instance:
            return ClientGPUInstance(instance, self)
        return None
    
    def terminate_instance(self, instance_id: str, provider: Optional[str] = None) -> bool:
        """Terminate instance"""
        return api.terminate_instance(instance_id, provider)
    
    def list_instances(self, provider: Optional[str] = None) -> List['ClientGPUInstance']:
        """List all user's instances"""
        instances = api.list_instances(provider)
        return [ClientGPUInstance(instance, self) for instance in instances]


class ClientGPUInstance:
    """GPU instance with client configuration
    
    Wraps GPUInstance to use client's SSH key configuration
    """
    
    def __init__(self, instance: GPUInstance, client: GPUClient):
        self._instance = instance
        self._client = client
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped instance"""
        return getattr(self._instance, name)
    
    def exec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30):
        """Execute command using client's SSH configuration (synchronous)"""
        if ssh_key_path is None:
            ssh_key_path = self._client.get_ssh_key_path()
        
        return self._instance.exec(command, ssh_key_path, timeout)
    
    async def aexec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30):
        """Execute command using client's SSH configuration (asynchronous)
        
        Args:
            command: Command to execute
            ssh_key_path: SSH private key path (uses client's if not provided)
            timeout: Command timeout in seconds
            
        Returns:
            SSHResult with command output
            
        Example:
            # Single async command
            result = await instance.aexec("nvidia-smi")
            print(f"GPU info: {result.stdout}")
            
            # Multiple commands in parallel
            gpu_info, disk_info, process_info = await asyncio.gather(
                instance.aexec("nvidia-smi --query-gpu=name,memory.total --format=csv"),
                instance.aexec("df -h /"),
                instance.aexec("ps aux | head -10")
            )
        """
        if ssh_key_path is None:
            ssh_key_path = self._client.get_ssh_key_path()
        
        return await self._instance.aexec(command, ssh_key_path, timeout)
    
    def exec_streaming(self, command: str, output_callback=None, ssh_key_path: Optional[str] = None, timeout: int = 30):
        """Execute command with real-time output streaming
        
        Args:
            command: Command to execute
            output_callback: Optional callback function(line, is_stderr) for real-time output
            ssh_key_path: SSH private key path (uses client's if not provided)
            timeout: Command timeout in seconds
        
        Returns:
            Tuple of (success, stdout, stderr)
        
        Example:
            def print_output(line, is_stderr):
                prefix = "ERR" if is_stderr else "OUT"
                print(f"[{prefix}] {line}")
            
            success, stdout, stderr = instance.exec_streaming("nvidia-smi", print_output)
        """
        if ssh_key_path is None:
            ssh_key_path = self._client.get_ssh_key_path()
        
        # Use the streaming SSH client directly
        from .ssh_clients import ParamikoSSHClient, get_ssh_connection_info
        
        try:
            hostname, port, username = get_ssh_connection_info(self._instance)
            client = ParamikoSSHClient()
            
            if client.connect(hostname, port, username, ssh_key_path, timeout):
                return client.execute_streaming(command, timeout, output_callback)
            else:
                return False, "", "SSH connection failed"
                
        except Exception as e:
            return False, "", f"Streaming execution failed: {e}"
    
    def wait_until_ready(self, timeout: int = 300) -> bool:
        """Wait until instance is running"""
        return self._instance.wait_until_ready(timeout)
    
    def wait_until_ssh_ready(self, timeout: int = 300) -> bool:
        """Wait until SSH is ready"""
        return self._instance.wait_until_ssh_ready(timeout)
    
    def refresh(self) -> 'ClientGPUInstance':
        """Refresh the wrapped instance with latest data"""
        updated_instance = self._client.get_instance(self._instance.id)
        if updated_instance:
            # Update the wrapped instance, keeping the wrapper
            self._instance = updated_instance._instance
            return self
        else:
            raise ValueError(f"Could not refresh instance {self._instance.id}")
    
    def terminate(self) -> bool:
        """Terminate this instance"""
        return self._instance.terminate()