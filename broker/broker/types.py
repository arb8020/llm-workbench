"""
Core data types for GPU cloud operations
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List
from enum import Enum


class InstanceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    STOPPED = "stopped"
    TERMINATED = "terminated"
    FAILED = "failed"


class GPUAvailability(str, Enum):
    """GPU availability status"""
    IMMEDIATE = "immediate"
    QUEUED = "queued"
    UNAVAILABLE = "unavailable"


class CloudType(str, Enum):
    """Cloud deployment type for GPU instances"""
    SECURE = "secure"
    COMMUNITY = "community" 
    ALL = "all"


@dataclass
class GPUOffer:
    """A GPU offer from a provider"""
    id: str
    provider: str
    gpu_type: str
    gpu_count: int
    vcpu: int
    memory_gb: int
    storage_gb: int
    price_per_hour: float
    availability_zone: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    # Additional fields for pandas-style queries
    vram_gb: Optional[int] = None
    region: Optional[str] = None
    availability: Optional[GPUAvailability] = None
    spot: bool = False
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    cloud_type: Optional[CloudType] = None


@dataclass 
class GPUInstance:
    """A provisioned GPU instance with convenience methods"""
    id: str
    provider: str
    status: InstanceStatus
    gpu_type: str
    gpu_count: int
    price_per_hour: float
    public_ip: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_username: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    def exec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30) -> 'SSHResult':
        """Execute command via SSH using configured key"""
        self._validate_ssh_ready()
        key_content = self._load_ssh_key(ssh_key_path)
        return self._execute_command(command, key_content, timeout)
    
    def _validate_ssh_ready(self) -> None:
        """Validate that instance has SSH connection details available"""
        if not self.public_ip or not self.ssh_username:
            raise ValueError("Instance SSH details not available - may not be running yet")
    
    def _load_ssh_key(self, ssh_key_path: Optional[str]) -> Optional[str]:
        """Load SSH private key content from file path"""
        import os
        
        if not ssh_key_path:
            return None
        
        key_path = os.path.expanduser(ssh_key_path)
        try:
            with open(key_path, 'r') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load SSH key from {key_path}: {e}")
    
    def _execute_command(self, command: str, key_content: Optional[str], timeout: int) -> 'SSHResult':
        """Execute SSH command and return formatted result"""
        from .ssh_clients import execute_command_sync
        
        success, stdout, stderr = execute_command_sync(
            self, key_content, command, timeout=timeout
        )
        
        return SSHResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            exit_code=0 if success else 1
        )
    
    def ssh_connection_string(self) -> str:
        """Get SSH connection string for use with bifrost or other tools.
        
        Returns:
            SSH connection string in format: user@host:port
            
        Raises:
            ValueError: If instance SSH details not available
        """
        self._validate_ssh_ready()
        return f"{self.ssh_username}@{self.public_ip}:{self.ssh_port}"
    
    def terminate(self) -> bool:
        """Terminate this instance"""
        from . import terminate_instance
        return terminate_instance(self.id, self.provider)
    
    def wait_until_ready(self, timeout: int = 300) -> bool:
        """Wait until instance status is RUNNING"""
        import time
        from . import get_instance
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            updated_instance = get_instance(self.id, self.provider)
            if not updated_instance:
                return False
                
            if updated_instance.status == InstanceStatus.RUNNING:
                # Update this instance with new details
                self.__dict__.update(updated_instance.__dict__)
                return True
            elif updated_instance.status in [InstanceStatus.FAILED, InstanceStatus.TERMINATED]:
                return False
                
            time.sleep(15)  # Check every 15 seconds
        
        return False  # Timeout
    
    def wait_until_ssh_ready(self, timeout: int = 300) -> bool:
        """Wait until instance is running AND SSH is ready for connections"""
        import time
        
        start_time = time.time()
        
        # First wait for instance to be running
        if not self.wait_until_ready(timeout=min(timeout, 300)):
            return False
        
        # Wait for direct SSH assignment
        if not self._wait_for_direct_ssh_assignment(start_time, timeout):
            return False
        
        # Test SSH connectivity
        return self._test_ssh_connectivity()
    
    def _wait_for_direct_ssh_assignment(self, start_time: float, timeout: int) -> bool:
        """Wait for direct SSH to be assigned (not proxy)."""
        import time
        
        print("Waiting for direct SSH to be assigned...")
        while time.time() - start_time < timeout:
            self.refresh()
            
            if self._has_direct_ssh_details():
                print(f"✅ Direct SSH assigned: {self.public_ip}:{self.ssh_port}")
                return True
            
            self._print_ssh_wait_status()
            time.sleep(10)
        
        print("Timeout waiting for direct SSH")
        return False
    
    def _has_direct_ssh_details(self) -> bool:
        """Check if instance has direct SSH details (not proxy)."""
        return (
            self.public_ip and 
            self.ssh_port and 
            self.public_ip != "ssh.runpod.io"
        )
    
    def _print_ssh_wait_status(self) -> None:
        """Print current SSH wait status."""
        if self.public_ip and self.ssh_port:
            if self.public_ip == "ssh.runpod.io":
                print("   Still waiting for direct SSH (currently proxy)...")
            else:
                print("   SSH details available but not recognized as direct...")
        else:
            print("   Still waiting for SSH details...")
    
    def _test_ssh_connectivity(self) -> bool:
        """Test SSH connectivity with a simple command."""
        import time
        
        print("✅ Got direct SSH! Waiting for SSH daemon to initialize...")
        time.sleep(30)  # SSH daemons typically need time to start
        
        try:
            result = self.exec("echo 'ssh_ready'", timeout=30)
            if result.success and "ssh_ready" in result.stdout:
                print("✅ SSH connectivity confirmed!")
                return True
            else:
                print(f"❌ SSH test failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ SSH connection error: {e}")
            return False
    
    def refresh(self) -> 'GPUInstance':
        """Refresh instance details from provider"""
        from . import get_instance
        
        updated_instance = get_instance(self.id, self.provider)
        if updated_instance:
            self.__dict__.update(updated_instance.__dict__)
            return self
        else:
            raise ValueError(f"Could not refresh instance {self.id}")
    
    def get_proxy_urls(self, ports: Optional[List[int]] = None) -> Dict[int, str]:
        """Get RunPod HTTP proxy URLs for exposed ports.
        
        Args:
            ports: Specific ports to get URLs for. If None, returns all exposed ports.
            
        Returns:
            Dict mapping port number to proxy URL
        """
        if self.provider != "runpod":
            return {}
            
        proxy_urls = {}
        
        # If no specific ports requested, try to get all exposed ports from raw_data
        if ports is None:
            ports = self._get_exposed_ports_from_runtime()
            
        if not ports:
            return {}
            
        for port in ports:
            proxy_urls[port] = f"https://{self.id}-{port}.proxy.runpod.net"
            
        return proxy_urls
    
    def get_proxy_url(self, port: int) -> Optional[str]:
        """Get RunPod HTTP proxy URL for a specific port.
        
        Args:
            port: Port number (e.g., 8000 for vLLM)
            
        Returns:
            Proxy URL or None if not available
        """
        if self.provider != "runpod":
            return None
            
        return f"https://{self.id}-{port}.proxy.runpod.net"
    
    def _get_exposed_ports_from_runtime(self) -> List[int]:
        """Extract exposed HTTP ports from runtime data."""
        if not self.raw_data or not isinstance(self.raw_data, dict):
            return []
            
        runtime = self.raw_data.get("runtime", {})
        if not isinstance(runtime, dict):
            return []
            
        ports_data = runtime.get("ports", [])
        if not isinstance(ports_data, list):
            return []
            
        http_ports = []
        for port_info in ports_data:
            if isinstance(port_info, dict):
                private_port = port_info.get("privatePort")
                public_port = port_info.get("publicPort") 
                # Check if it's an HTTP proxy port (not just TCP)
                if private_port and private_port != 22:  # Exclude SSH
                    http_ports.append(private_port)
                    
        return http_ports


@dataclass
class ProvisionRequest:
    """Request to provision a GPU instance"""
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    name: Optional[str] = None
    max_price_per_hour: Optional[float] = None
    provider: Optional[str] = None  # If None, search all providers
    spot_instance: bool = False
    ssh_startup_script: Optional[str] = None  # SSH key injection script
    container_disk_gb: Optional[int] = None  # Container disk size in GB (default: 50)
    volume_disk_gb: Optional[int] = None  # Volume disk size in GB (default: 0)
    # Port exposure configuration
    exposed_ports: Optional[List[int]] = None  # Ports to expose via HTTP proxy
    enable_http_proxy: bool = True  # Enable RunPod's HTTP proxy


@dataclass
class SSHResult:
    """Result of SSH command execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    command: Optional[str] = None


@dataclass
class SSHConfig:
    """SSH connection configuration"""
    hostname: str
    port: int
    username: str
    key_path: Optional[str] = None
    method: Optional[str] = None  # "direct" or "proxy"