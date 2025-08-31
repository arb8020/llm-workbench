#!/usr/bin/env python3
"""Test script for vLLM backend locally before remote deployment."""

import asyncio
import sys
from pathlib import Path

# Add engine package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.backends.vllm import VLLMConfig, VLLMBackend


async def test_vllm_local():
    """Test vLLM backend locally."""
    print("🧪 Testing vLLM backend locally...")
    
    # Create config optimized for GPT-2 testing
    config = VLLMConfig.for_gpt2_testing(port=8000)
    print(f"📋 Config: {config.model_name} on port {config.port}")
    
    # Create backend
    backend = VLLMBackend(config)
    
    try:
        # Start the backend
        endpoint_url = await backend.start()
        print(f"🚀 Backend started at: {endpoint_url}")
        
        # Test inference
        print("🔄 Testing inference...")
        success = await backend.test_inference()
        
        if success:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Inference test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        # Always clean up
        if backend.is_running:
            print("🧹 Cleaning up...")
            await backend.stop()


async def main():
    """Main test function."""
    print("=" * 50)
    print("vLLM Backend Local Test")
    print("=" * 50)
    
    # Check if vLLM is installed
    try:
        import vllm
        print(f"✅ vLLM installed: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM not installed. Install with: pip install vllm")
        return False
    
    # Run the test
    success = await test_vllm_local()
    
    if success:
        print("\n🎉 Local vLLM backend test successful!")
        print("   Ready for remote deployment with broker + bifrost")
    else:
        print("\n💥 Local test failed - fix issues before remote deployment")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())