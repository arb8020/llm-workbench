#!/usr/bin/env python3
"""
Test basic NNsight tutorial patterns to establish working baseline
"""

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

def test_basic_nnsight():
    """Test the most basic NNsight pattern from official docs"""
    print("=== Testing Basic NNsight Tutorial Pattern ===")
    
    model_id = 'willcb/Qwen3-0.6B'
    
    try:
        # Load model and tokenizer
        print(f"Loading model: {model_id}")
        model = LanguageModel(model_id, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        prompt = "Hello, how are you?"
        print(f"Prompt: {prompt}")
        
        # Most basic pattern from NNsight tutorial
        print("\n--- Testing basic trace pattern ---")
        with model.trace(prompt):
            # Save logits from the final layer
            logits = model.lm_head.output.save()
        
        print(f"✅ Basic trace successful!")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits dtype: {logits.dtype}")
        
        # Test generate pattern
        print("\n--- Testing generate pattern ---")
        with model.generate(max_new_tokens=3) as tracer:
            with tracer.invoke(prompt):
                logits_gen = model.lm_head.output.save()
        
        print(f"✅ Generate pattern successful!")
        print(f"Generated logits shape: {logits_gen.shape}")
        print(f"Generated text: {tokenizer.decode(tracer.output[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_nnsight()