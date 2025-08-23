#!/usr/bin/env python3
"""Universal test client for all inference server types."""

import argparse
import json
import sys
import time
from typing import Optional, Dict, Any
import requests


def test_server_health(url: str) -> bool:
    """Test if server is healthy and responding."""
    try:
        # Try health endpoint first
        health_response = requests.get(f"{url}/health", timeout=5)
        if health_response.status_code == 200:
            return True
    except:
        pass
    
    try:
        # Fallback to models endpoint
        models_response = requests.get(f"{url}/v1/models", timeout=5)
        return models_response.status_code == 200
    except:
        return False


def get_server_capabilities(url: str) -> Dict[str, Any]:
    """Get server capabilities if available."""
    try:
        response = requests.get(f"{url}/v1/capabilities", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def test_non_streaming(url: str, messages: list, model: str = "openai-community/gpt2", 
                      max_tokens: int = 50, temperature: float = 0.1,
                      collect_activations: Optional[Dict] = None) -> Dict[str, Any]:
    """Test non-streaming chat completion."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    # Add interpretability extensions if provided
    if collect_activations:
        payload["collect_activations"] = collect_activations
    
    start_time = time.time()
    response = requests.post(
        f"{url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    end_time = time.time()
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    result = response.json()
    result["_timing"] = {
        "total_time": end_time - start_time,
        "tokens_per_second": result.get("usage", {}).get("completion_tokens", 0) / (end_time - start_time) if end_time > start_time else 0
    }
    
    return result


def test_streaming(url: str, messages: list, model: str = "openai-community/gpt2",
                  max_tokens: int = 50, temperature: float = 0.1) -> Dict[str, Any]:
    """Test streaming chat completion with manual SSE parsing."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    
    start_time = time.time()
    response = requests.post(
        f"{url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=30
    )
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    # Parse SSE stream manually
    chunks = []
    accumulated_content = ""
    
    print("📡 Streaming response:")
    
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
            
        # Parse Server-Sent Events format
        if line.startswith('data: '):
            data = line[6:]  # Remove 'data: ' prefix
            
            if data.strip() == '[DONE]':
                break
                
            try:
                chunk = json.loads(data)
                chunks.append(chunk)
                
                # Extract content from chunk
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        accumulated_content += content
                        print(content, end="", flush=True)
                        
            except json.JSONDecodeError:
                # Skip malformed JSON chunks
                continue
    
    print()  # New line after streaming
    end_time = time.time()
    
    # Construct final result similar to non-streaming
    return {
        "id": chunks[0]["id"] if chunks else "streaming-test",
        "object": "chat.completion",
        "created": chunks[0]["created"] if chunks else int(time.time()),
        "model": model,
        "usage": {
            "prompt_tokens": len(" ".join([m["content"] for m in messages]).split()),
            "completion_tokens": len(accumulated_content.split()),
            "total_tokens": len(" ".join([m["content"] for m in messages]).split()) + len(accumulated_content.split())
        },
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": accumulated_content
            },
            "finish_reason": "stop"
        }],
        "_timing": {
            "total_time": end_time - start_time,
            "tokens_per_second": len(accumulated_content.split()) / (end_time - start_time) if end_time > start_time else 0
        },
        "_chunks_received": len(chunks)
    }


def test_completions_fallback(url: str, prompt: str, model: str = "openai-community/gpt2",
                             max_tokens: int = 50, temperature: float = 0.1,
                             collect_activations: Optional[Dict] = None) -> Dict[str, Any]:
    """Fallback to completions API when chat completions fails due to missing chat template."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    # Add interpretability extensions if provided
    if collect_activations:
        payload["collect_activations"] = collect_activations
    
    start_time = time.time()
    response = requests.post(
        f"{url}/v1/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    end_time = time.time()
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    result = response.json()
    
    # Convert completions response to chat completions format for consistency
    if "choices" in result and len(result["choices"]) > 0:
        completion_text = result["choices"][0].get("text", "")
        
        # Convert to chat completion format
        chat_result = {
            "id": result.get("id", "completion-fallback"),
            "object": "chat.completion",
            "created": result.get("created", int(time.time())),
            "model": model,
            "usage": result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": completion_text
                },
                "finish_reason": result["choices"][0].get("finish_reason", "stop")
            }],
            "_timing": {
                "total_time": end_time - start_time,
                "tokens_per_second": result.get("usage", {}).get("completion_tokens", 0) / (end_time - start_time) if end_time > start_time else 0
            }
        }
        
        # Include activations if present
        if "activations" in result:
            chat_result["activations"] = result["activations"]
            
        return chat_result
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Universal test client for inference servers")
    parser.add_argument("--url", required=True, help="Server URL (e.g., http://localhost:8000)")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt to test with")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--stream", action="store_true", help="Use streaming response")
    parser.add_argument("--model", default="openai-community/gpt2", help="Model name")
    parser.add_argument("--system", default="", help="System message (optional)")
    
    # Interpretability options
    parser.add_argument("--collect-activations", action="store_true", 
                       help="Enable activation collection (for interpretability servers)")
    parser.add_argument("--activation-layers", nargs="+", type=int, default=[6, 12],
                       help="Layers to collect activations from (default: 6 12)")
    parser.add_argument("--activation-hooks", nargs="+", default=["output"],
                       help="Hook points to collect (default: output)")
    
    # Output options
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Clean URL
    url = args.url.rstrip('/')
    
    try:
        # 1. Test server health
        print(f"🔍 Testing server at: {url}")
        if not test_server_health(url):
            print(f"❌ Server at {url} is not responding")
            sys.exit(1)
        print("✅ Server is healthy")
        
        # 2. Get capabilities
        capabilities = get_server_capabilities(url)
        if capabilities:
            print(f"🧠 Server capabilities: {', '.join(k for k, v in capabilities.items() if v)}")
        
        # 3. Prepare messages
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        
        # 4. Prepare activation collection if requested
        collect_activations = None
        if args.collect_activations:
            if not capabilities.get("activation_collection", False):
                print("⚠️  Activation collection requested but server doesn't support it")
                print("   This will likely fail or be ignored")
            
            collect_activations = {
                "layers": args.activation_layers,
                "hook_points": args.activation_hooks,
                "positions": [-1]  # Last token
            }
            print(f"🧠 Activation collection: layers {args.activation_layers}, hooks {args.activation_hooks}")
        
        # 5. Make request
        print(f"📝 Prompt: '{args.prompt}'")
        print(f"🎛️  Settings: max_tokens={args.max_tokens}, temperature={args.temperature}, stream={args.stream}")
        
        try:
            if args.stream:
                result = test_streaming(url, messages, args.model, args.max_tokens, args.temperature)
            else:
                result = test_non_streaming(url, messages, args.model, args.max_tokens, 
                                          args.temperature, collect_activations)
        except Exception as e:
            # Check if it's a chat template error - fallback to completions API
            if "chat template" in str(e).lower() and "gpt2" in args.model.lower():
                print("⚠️  Chat template error detected - falling back to completions API")
                result = test_completions_fallback(url, args.prompt, args.model, args.max_tokens, 
                                                 args.temperature, collect_activations)
            else:
                raise
        
        # 6. Display results
        print(f"\n📊 Results:")
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Formatted output
            response_content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            timing = result.get("_timing", {})
            
            print(f"✨ Response: {response_content}")
            print(f"📈 Usage: {usage.get('completion_tokens', 0)} tokens in {timing.get('total_time', 0):.2f}s")
            print(f"⚡ Speed: {timing.get('tokens_per_second', 0):.1f} tokens/sec")
            
            if "activations" in result:
                activations = result["activations"]
                print(f"🧠 Collected activations: {list(activations.keys())}")
                if args.verbose:
                    for key, data in activations.items():
                        if isinstance(data, list):
                            print(f"   {key}: shape ~{len(data)} (showing first few: {data[:3]}...)")
                        else:
                            print(f"   {key}: {data}")
            
            if args.stream and "_chunks_received" in result:
                print(f"📡 Received {result['_chunks_received']} streaming chunks")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()