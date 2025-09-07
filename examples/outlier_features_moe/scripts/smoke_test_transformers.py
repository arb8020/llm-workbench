#!/usr/bin/env python3
"""
Smoke test for transformers compatibility with GLM-4.5-Air and GPT-OSS models.
Tests model loading without running full analysis.
"""

import sys
import traceback
from typing import List, Dict, Any

def test_model_loading(model_name: str) -> Dict[str, Any]:
    """Test if a model can be loaded with current transformers version."""
    result = {
        "model": model_name,
        "success": False,
        "error": None,
        "config_loaded": False,
        "tokenizer_loaded": False,
        "nnsight_compatible": False
    }
    
    try:
        print(f"\n🔍 Testing {model_name}...")
        
        # Test 1: Config loading
        print("  Step 1: Loading config...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        result["config_loaded"] = True
        print(f"  ✅ Config loaded: {config.model_type}")
        
        # Test 2: Tokenizer loading  
        print("  Step 2: Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        result["tokenizer_loaded"] = True
        print(f"  ✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        # Test 3: NNsight compatibility (without actual model loading)
        print("  Step 3: Testing nnsight compatibility...")
        from nnsight.modeling import LanguageModel
        # Just test if nnsight can create the meta object without loading weights
        try:
            llm = LanguageModel(model_name, device_map="cpu", torch_dtype="auto", load_in_4bit=False)
            result["nnsight_compatible"] = True
            print("  ✅ NNsight compatibility confirmed")
            del llm  # Clean up
        except Exception as e:
            print(f"  ⚠️  NNsight compatibility issue: {str(e)[:100]}...")
        
        result["success"] = True
        print(f"  🎉 {model_name} - COMPATIBLE!")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"  ❌ {model_name} - FAILED: {str(e)[:100]}...")
        if "model_type" in str(e).lower() or "not recognize" in str(e).lower():
            print(f"     This is likely a transformers version issue")
    
    return result

def main():
    """Run smoke tests on problematic models."""
    print("🔬 Transformers Compatibility Smoke Test")
    print("=" * 50)
    
    # Test models that have failed
    test_models = [
        "zai-org/GLM-4.5-Air",
        "openai/gpt-oss-120b"
    ]
    
    # Also test a known working model as control
    control_models = [
        "Qwen/Qwen3-30B-A3B"  # Known to work
    ]
    
    results = []
    
    print("\n📋 Testing problematic models:")
    for model in test_models:
        results.append(test_model_loading(model))
    
    print("\n📋 Testing control models:")
    for model in control_models:
        results.append(test_model_loading(model))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SMOKE TEST SUMMARY")
    print("=" * 50)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['model']}")
        if not result["success"]:
            print(f"     Error: {result['error'][:80]}...")
        else:
            checks = []
            if result["config_loaded"]: checks.append("Config")
            if result["tokenizer_loaded"]: checks.append("Tokenizer") 
            if result["nnsight_compatible"]: checks.append("NNsight")
            print(f"     Working: {', '.join(checks)}")
    
    # Return exit code for automation
    failed_count = sum(1 for r in results if not r["success"])
    if failed_count > 0:
        print(f"\n❌ {failed_count} models failed compatibility test")
        return 1
    else:
        print(f"\n🎉 All {len(results)} models passed compatibility test!")
        return 0

if __name__ == "__main__":
    sys.exit(main())