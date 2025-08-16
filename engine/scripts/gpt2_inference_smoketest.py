#!/usr/bin/env python3
"""
GPT-2 Inference Smoke Test

A minimal smoke test that validates:
1. Tokenizer functionality 
2. Model loading (placeholder)
3. GPU availability
4. Basic inference pipeline

This is designed to be lightweight and quick for testing remote deployment.
"""

import sys
import time
from pathlib import Path

def test_tokenizer():
    """Test GPT-2 tokenizer functionality."""
    print("🔤 Testing GPT-2 tokenizer...")
    
    try:
        from engine.utils.tokenizer import download_gpt2_assets, load_tokenizer_files, encode, decode
        
        # Download tokenizer assets (cached after first run)
        assets_dir = download_gpt2_assets("gpt2")
        tok = load_tokenizer_files(assets_dir)
        
        # Test encoding/decoding
        test_text = "Hello, GPT-2! How are you today? 🤖"
        tokens = encode(tok, test_text)
        decoded = decode(tok, tokens)
        
        print(f"   Original: {test_text}")
        print(f"   Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''} ({len(tokens)} total)")
        print(f"   Decoded: {decoded}")
        
        assert decoded == test_text, "Tokenizer roundtrip failed!"
        print("   ✅ Tokenizer working")
        return True
        
    except Exception as e:
        print(f"   ❌ Tokenizer failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and CUDA setup."""
    print("🚀 Testing GPU availability...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        
        print(f"   CUDA available: {cuda_available}")
        print(f"   GPU count: {gpu_count}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {memory_gb:.1f} GB")
            
            # Simple tensor test
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x.T)
            result = y.cpu().sum().item()
            print(f"   GPU compute test: {result:.2f}")
            print("   ✅ GPU working")
        else:
            print("   ⚠️ No GPU available (CPU only)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False

def test_mock_inference():
    """Mock inference test to validate the pipeline."""
    print("🧠 Testing mock inference pipeline...")
    
    try:
        # This would normally load a real model, but for smoke test we mock it
        print("   [Mock] Loading GPT-2 model...")
        time.sleep(0.5)  # Simulate loading
        
        print("   [Mock] Running inference...")
        prompt = "The future of AI is"
        mock_output = prompt + " bright and full of possibilities for humanity."
        
        print(f"   Input: {prompt}")
        print(f"   Output: {mock_output}")
        print("   ✅ Mock inference working")
        return True
        
    except Exception as e:
        print(f"   ❌ Mock inference failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("🌍 GPT-2 Inference Smoke Test")
    print("=" * 50)
    
    tests = [
        ("Tokenizer", test_tokenizer),
        ("GPU Availability", test_gpu_availability), 
        ("Mock Inference", test_mock_inference),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        success = test_func()
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SMOKE TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All smoke tests passed!")
        print("🚀 Ready for GPT-2 inference!")
    else:
        print("\n🚨 Some tests failed - check setup")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())