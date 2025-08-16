"""
Core data types for GPU cloud operations
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
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
        from .ssh_clients import execute_command_sync
        import os
        
        if not self.public_ip or not self.ssh_username:
            raise ValueError("Instance SSH details not available - may not be running yet")
        
        # Determine which SSH key to use
        key_path = None
        if ssh_key_path:
            # Use provided key path
            key_path = os.path.expanduser(ssh_key_path)
        
        # Load private key if path provided
        private_key_content = None
        if key_path:
            try:
                with open(key_path, 'r') as f:
                    private_key_content = f.read()
            except Exception as e:
                raise ValueError(f"Failed to load SSH key from {key_path}: {e}")
        
        success, stdout, stderr = execute_command_sync(
            self, private_key_content, command, timeout=timeout
        )
        
        return SSHResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            exit_code=0 if success else 1
        )
    
    def run(self, command: str, **kwargs) -> 'SSHResult':
        """Alias for exec()"""
        return self.exec(command, **kwargs)
    
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
        from . import get_instance
        
        start_time = time.time()
        
        # First wait for instance to be running
        if not self.wait_until_ready(timeout=min(timeout, 300)):
            return False
        
        # Wait for direct SSH to be assigned (public_ip != "ssh.runpod.io")
        print("Waiting for direct SSH to be assigned...")
        while time.time() - start_time < timeout:
            self.refresh()  # Update instance details
            
            if self.public_ip and self.ssh_port:
                if self.public_ip != "ssh.runpod.io":
                    print(f"✅ Direct SSH assigned: {self.public_ip}:{self.ssh_port}")
                    break
                else:
                    print("   Still waiting for direct SSH (currently proxy)...")
            else:
                print("   Still waiting for SSH details...")
            
            time.sleep(10)
        else:
            print("Timeout waiting for direct SSH")
            return False
        
        # We have direct SSH - wait for SSH daemon to be ready
        print(f"✅ Got direct SSH! Waiting for SSH daemon to initialize...")
        time.sleep(30)  # SSH daemons typically need time to start
        
        # Test SSH connectivity with a simple command
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