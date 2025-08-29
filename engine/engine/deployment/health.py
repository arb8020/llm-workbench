"""Health checking utilities for remote inference servers."""

import time
from typing import Optional, List


def wait_for_server_ready(
    bifrost_client,
    model_name: str,
    timeout: int = 600,
    require_capabilities: Optional[List[str]] = None
) -> bool:
    """
    Wait for OpenAI-compatible server to be ready.
    
    Args:
        bifrost_client: Bifrost client for remote execution
        model_name: Model name to check in /v1/models response  
        timeout: Max wait time in seconds
        require_capabilities: Optional capabilities to check at /v1/capabilities
        
    Returns:
        True if server becomes ready, False if timeout
    """
    start_time = time.time()
    server_ready = False
    
    while not server_ready and (time.time() - start_time) < timeout:
        try:
            # Check if OpenAI-compatible server is responding
            models_check = bifrost_client.exec("curl -s --connect-timeout 5 http://localhost:8000/v1/models")
            if models_check and model_name.lower() in models_check.lower():
                
                # Check additional capabilities if required (nnsight, custom servers)
                if require_capabilities:
                    caps_check = bifrost_client.exec("curl -s --connect-timeout 5 http://localhost:8000/v1/capabilities")
                    if not caps_check:
                        continue
                    
                    caps_lower = caps_check.lower()
                    if not all(cap.lower() in caps_lower for cap in require_capabilities):
                        continue
                
                server_ready = True
                break
                
            # Fallback: try a simple completions request
            test_completion = bifrost_client.exec(
                f'curl -s --connect-timeout 5 -X POST http://localhost:8000/v1/completions '
                f'-H "Content-Type: application/json" '
                f'-d \'{{"model":"{model_name}","prompt":"test","max_tokens":1}}\''
            )
            if test_completion and ("choices" in test_completion.lower() or "text" in test_completion.lower()):
                server_ready = True
                break
                
        except Exception:
            pass
        
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0:  # Print update every 30 seconds
            print(f"   Still loading {model_name} model... ({elapsed}s elapsed)")
            
        time.sleep(10)
    
    return server_ready