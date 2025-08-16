"""vLLM inference backend implementation."""

import requests
from typing import List
from .config import VLLMConfig
from ..base import InferenceBackend


class VLLMBackend(InferenceBackend):
    """vLLM inference server backend.
    
    Manages the lifecycle of a vLLM API server process.
    """
    
    backend_name = "vllm"
    
    def __init__(self, config: VLLMConfig):
        super().__init__(config)
        self.config: VLLMConfig = config  # Type hint for better IDE support
        
    def build_command(self) -> List[str]:
        """Build the vLLM server startup command."""
        cmd = ["python", "-m", "vllm.entrypoints.api_server"]
        cmd.extend(self.config.to_vllm_args())
        return cmd
        
    def health_check(self) -> bool:
        """Check if vLLM server is healthy via the models endpoint."""
        try:
            response = requests.get(
                f"{self.config.endpoint_url}/v1/models",
                timeout=2.0
            )
            if response.status_code == 200:
                data = response.json()
                # Verify our model is actually loaded
                models = data.get("data", [])
                expected_model = self.config.model_name
                
                # vLLM sometimes changes model names slightly
                for model in models:
                    model_id = model.get("id", "")
                    if expected_model in model_id or model_id in expected_model:
                        return True
                        
                print(f"⚠️  vLLM server running but model {expected_model} not found")
                print(f"   Available models: {[m.get('id') for m in models]}")
                return False
                
        except requests.exceptions.RequestException:
            # Server not ready yet
            return False
        except Exception as e:
            print(f"⚠️  Health check error: {e}")
            return False
            
        return False
    
    async def test_inference(self) -> bool:
        """Test that the backend can actually perform inference."""
        try:
            response = requests.post(
                f"{self.config.endpoint_url}/v1/chat/completions",
                json={
                    "model": self.config.model_name,
                    "messages": [
                        {"role": "user", "content": "Hello! This is a test."}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    content = message.get("content", "")
                    print(f"✅ Inference test successful: '{content.strip()}'")
                    return True
            
            print(f"❌ Inference test failed: {response.status_code} - {response.text}")
            return False
            
        except Exception as e:
            print(f"❌ Inference test error: {e}")
            return False