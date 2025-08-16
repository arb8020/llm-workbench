"""vLLM backend configuration."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from ..base import BackendConfig


@dataclass
class VLLMConfig(BackendConfig):
    """Configuration for vLLM inference backend.
    
    Optimized defaults for GPT-2 and small GPU instances.
    """
    backend_name: str = "vllm"
    
    # Model configuration
    model_name: str = "openai-community/gpt2"
    
    # Memory and performance settings (conservative for cheap GPUs)
    gpu_memory_utilization: float = 0.7  # Conservative for stability
    max_model_len: Optional[int] = 1024   # Small context for GPT-2
    
    # Model loading settings
    dtype: str = "auto"
    trust_remote_code: bool = False
    
    # Parallelism settings (single GPU defaults)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Performance optimizations
    enable_chunked_prefill: bool = False
    max_num_batched_tokens: Optional[int] = None
    
    # API settings  
    disable_log_stats: bool = False
    api_key: Optional[str] = None
    
    def to_vllm_args(self) -> List[str]:
        """Convert configuration to vLLM command line arguments."""
        args = [
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--pipeline-parallel-size", str(self.pipeline_parallel_size),
        ]
        
        if self.max_model_len:
            args.extend(["--max-model-len", str(self.max_model_len)])
            
        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
            
        if self.max_num_batched_tokens:
            args.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
            
        if self.trust_remote_code:
            args.append("--trust-remote-code")
            
        if self.disable_log_stats:
            args.append("--disable-log-stats")
            
        if self.api_key:
            args.extend(["--api-key", self.api_key])
            
        # Add any additional backend args
        for key, value in self.backend_args.items():
            # Convert underscores to dashes for CLI args
            cli_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    args.append(f"--{cli_key}")
            else:
                args.extend([f"--{cli_key}", str(value)])
                
        return args
    
    @classmethod
    def for_gpt2_testing(cls, port: int = 8000) -> 'VLLMConfig':
        """Create optimized config for GPT-2 testing on cheap GPUs."""
        return cls(
            model_name="openai-community/gpt2",
            port=port,
            gpu_memory_utilization=0.6,  # Very conservative
            max_model_len=512,           # Small context window
            disable_log_stats=True,      # Reduce noise during testing
        )