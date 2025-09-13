#!/usr/bin/env python3
"""
Debug notebook for NNsight activation capture patterns.

Test different approaches to find what actually works for:
1. Single-pass generation with activation capture
2. Avoiding "Cannot return output of Envoy" errors
3. Getting both text output and saved activations

Run this to test patterns quickly without the full server overhead.
"""

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

def test_pattern_1_basic_trace():
    """Pattern 1: Basic trace (from README examples)"""
    print("=== Pattern 1: Basic trace ===")
    
    model = LanguageModel('openai-community/gpt2', device_map='auto')
    
    try:
        with model.trace('The Eiffel Tower is in the city of') as tracer:
            hidden_states = model.transformer.h[-1].output[0].save()
            output = model.output.save()
        
        print("✅ Trace completed")
        print(f"Hidden states shape: {hidden_states.value.shape}")
        print(f"Output type: {type(output.value)}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_pattern_2_generate_basic():
    """Pattern 2: Basic generate (from README)"""
    print("\n=== Pattern 2: Basic generate ===")
    
    model = LanguageModel('openai-community/gpt2', device_map='auto')
    
    try:
        with model.generate('The Eiffel Tower is in the city of', max_new_tokens=3) as tracer:
            hidden_states = model.transformer.h[-1].output[0].save()
        
        print("✅ Generate completed")
        print(f"Generated text: {tracer.output}")
        print(f"Hidden states shape: {hidden_states.value.shape}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_pattern_3_generate_invoke():
    """Pattern 3: Generate with invoke (from troubleshooting guide)"""
    print("\n=== Pattern 3: Generate + invoke ===")
    
    model = LanguageModel('openai-community/gpt2', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    prompt_text = "The Eiffel Tower is in the city of"
    
    try:
        gen_kwargs = dict(max_new_tokens=3, temperature=0.7, do_sample=True)
        
        with model.generate(**gen_kwargs) as tracer:
            # Register savepoints first
            hidden_states = model.transformer.h[-1].output[0].save()
            logits = model.lm_head.output.save()
            
            # Then invoke
            with tracer.invoke(prompt_text):
                pass
        
        print("✅ Generate + invoke completed")
        gen_ids = tracer.output
        reply_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"Generated text: {reply_text}")
        print(f"Hidden states shape: {hidden_states.value.shape}")
        print(f"Logits shape: {logits.value.shape}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_pattern_4_generate_invoke_layers():
    """Pattern 4: Test layer access patterns"""
    print("\n=== Pattern 4: Generate + invoke with layer selectors ===")
    
    model = LanguageModel('openai-community/gpt2', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    prompt_text = "The Eiffel Tower is in the city of"
    
    try:
        gen_kwargs = dict(max_new_tokens=3, temperature=0.7, do_sample=True)
        
        with model.generate(**gen_kwargs) as tracer:
            # Test different layer access patterns
            activations = {}
            
            # Pattern from server: model.layers[0].input_layernorm.output
            try:
                activations["layer0_in"] = model.model.layers[0].input_layernorm.output.save()
                print("✅ Layer 0 input_layernorm accessible")
            except Exception as e:
                print(f"❌ Layer 0 input_layernorm failed: {e}")
            
            # Pattern: output.logits
            try:
                activations["output_logits"] = model.output.logits.save()
                print("✅ output.logits accessible")
            except Exception as e:
                print(f"❌ output.logits failed: {e}")
            
            # Pattern: lm_head.output  
            try:
                activations["lm_head"] = model.lm_head.output.save()
                print("✅ lm_head.output accessible")
            except Exception as e:
                print(f"❌ lm_head.output failed: {e}")
            
            # Then invoke
            with tracer.invoke(prompt_text):
                pass
        
        print("✅ Generate + invoke with layers completed")
        gen_ids = tracer.output
        reply_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"Generated text: {reply_text}")
        
        # Check which activations worked
        for name, proxy in activations.items():
            try:
                value = proxy.value
                print(f"✅ {name}: {value.shape}")
            except Exception as e:
                print(f"❌ {name} value failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_pattern_5_different_models():
    """Pattern 5: Test with different model (Qwen like in server)"""
    print("\n=== Pattern 5: Test with Qwen model ===")
    
    # Use same model as server for exact comparison
    model_id = 'willcb/Qwen3-0.6B'
    
    try:
        model = LanguageModel(model_id, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt_text = "Hello, how are you?"
        
        gen_kwargs = dict(max_new_tokens=5, temperature=0.1, do_sample=True)
        
        with model.generate(**gen_kwargs) as tracer:
            # Test selectors from our server
            activations = {}
            
            try:
                activations["lm_head"] = model.lm_head.output.save()
            except Exception as e:
                print(f"lm_head.output failed: {e}")
            
            try:
                activations["layer0_in"] = model.model.layers[0].input_layernorm.output.save()
            except Exception as e:
                print(f"layer0 input_layernorm failed: {e}")
            
            with tracer.invoke(prompt_text):
                pass
        
        print("✅ Qwen model test completed")
        gen_ids = tracer.output
        reply_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"Generated text: {reply_text}")
        
        for name, proxy in activations.items():
            try:
                value = proxy.value
                print(f"✅ {name}: {value.shape}")
            except Exception as e:
                print(f"❌ {name} value failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    """Run all test patterns to see what works"""
    print("🧪 Testing NNsight activation capture patterns...\n")
    
    patterns = [
        test_pattern_1_basic_trace,
        test_pattern_2_generate_basic, 
        test_pattern_3_generate_invoke,
        test_pattern_4_generate_invoke_layers,
        test_pattern_5_different_models,
    ]
    
    results = []
    for pattern in patterns:
        try:
            success = pattern()
            results.append(success)
        except Exception as e:
            print(f"❌ Pattern crashed: {e}")
            results.append(False)
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} patterns worked")
    print("Use the working patterns to fix the server!")

if __name__ == "__main__":
    main()