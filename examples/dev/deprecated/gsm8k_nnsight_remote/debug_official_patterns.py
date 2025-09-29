#!/usr/bin/env python3
"""
Debug notebook for OFFICIAL NNsight generate patterns from documentation.

Based on official docs, test the correct generate patterns:
1. Basic generate with prompt
2. Generate with .all() for multi-token
3. Generate without prompt + invoke
4. Generate with .iter[] for specific tokens

These are the patterns I missed in my first debug attempt.
"""

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

def test_official_pattern_1_generate_basic():
    """Official Pattern 1: Basic generate with prompt and save"""
    print("=== Official Pattern 1: Basic generate with prompt ===")
    
    model_id = 'willcb/Qwen3-0.6B'
    
    try:
        model = LanguageModel(model_id, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt = "Hello, how are you?"
        
        # This is the pattern from official docs
        with model.generate(prompt, max_new_tokens=3) as tracer:
            # Save activations during generation
            hidden_states = model.lm_head.output.save()
            
            # Try layer access  
            try:
                layer_states = model.model.layers[0].input_layernorm.output.save()
                print("✅ Layer access worked")
            except Exception as e:
                print(f"❌ Layer access failed: {e}")
        
        print("✅ Basic generate completed")
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Generated: {tokenizer.decode(tracer.output[0]) if hasattr(tracer, 'output') else 'No output attr'}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_official_pattern_2_generate_all():
    """Official Pattern 2: Generate with .all() for multi-token capture"""
    print("\n=== Official Pattern 2: Generate with .all() ===")
    
    model_id = 'willcb/Qwen3-0.6B'
    
    try:
        model = LanguageModel(model_id, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt = "Hello, how are you?"
        
        # Pattern from official walkthrough
        with model.generate(prompt, max_new_tokens=3) as tracer:
            hidden_states = list().save()  # Initialize saveable list
            
            with tracer.all():
                # Collect hidden states across all generation iterations
                hidden_states.append(model.lm_head.output)
        
        print("✅ Generate with .all() completed")
        print(f"Collected {len(hidden_states)} hidden states")
        print(f"Generated: {tokenizer.decode(tracer.output[0]) if hasattr(tracer, 'output') else 'No output attr'}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_official_pattern_3_generate_invoke():
    """Official Pattern 3: Generate without prompt + invoke (from cross-prompt examples)"""
    print("\n=== Official Pattern 3: Generate + invoke ===")
    
    model_id = 'willcb/Qwen3-0.6B'
    
    try:
        model = LanguageModel(model_id, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt = "Hello, how are you?"
        
        # Pattern from cross-prompt docs - no prompt in generate, use invoke
        with model.generate(max_new_tokens=3) as tracer:
            with tracer.invoke(prompt):
                # Save activations during invoke
                hidden_states = model.lm_head.output.save()
                
                # Try layer access
                try:
                    layer_states = model.model.layers[0].input_layernorm.output.save()
                    print("✅ Layer access in invoke worked")
                except Exception as e:
                    print(f"❌ Layer access in invoke failed: {e}")
        
        print("✅ Generate + invoke completed")
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Generated: {tokenizer.decode(tracer.output[0]) if hasattr(tracer, 'output') else 'No output attr'}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_official_pattern_4_generate_iter():
    """Official Pattern 4: Generate with .iter[] for specific iterations"""
    print("\n=== Official Pattern 4: Generate with .iter[] ===")
    
    model_id = 'willcb/Qwen3-0.6B'
    
    try:
        model = LanguageModel(model_id, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt = "Hello, how are you?"
        
        # Pattern from official walkthrough  
        with model.generate(prompt, max_new_tokens=3) as tracer:
            hidden_states = list().save()
            
            with tracer.iter[0:2]:  # Only first 2 iterations
                # Save hidden states for specific iterations
                hidden_states.append(model.lm_head.output)
        
        print("✅ Generate with .iter[] completed")
        print(f"Collected {len(hidden_states)} hidden states")
        print(f"Generated: {tokenizer.decode(tracer.output[0]) if hasattr(tracer, 'output') else 'No output attr'}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_official_pattern_5_multiple_invoke():
    """Official Pattern 5: Multiple invoke calls (cross-prompt pattern)"""
    print("\n=== Official Pattern 5: Multiple invoke calls ===")
    
    model_id = 'willcb/Qwen3-0.6B'
    
    try:
        model = LanguageModel(model_id, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Pattern from cross-prompt examples
        with model.generate(max_new_tokens=3) as tracer:
            with tracer.invoke("First prompt"):
                embeddings = model.transformer.wte.output.save() if hasattr(model, 'transformer') else None
                hidden1 = model.lm_head.output.save()
            
            with tracer.invoke("Second prompt"):  
                hidden2 = model.lm_head.output.save()
                # Could use embeddings here if they exist
        
        print("✅ Multiple invoke completed")
        print(f"First hidden shape: {hidden1.shape}")
        print(f"Second hidden shape: {hidden2.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    """Test all official NNsight generate patterns"""
    print("🧪 Testing OFFICIAL NNsight generate patterns from docs...\n")
    
    patterns = [
        test_official_pattern_1_generate_basic,
        test_official_pattern_2_generate_all,
        test_official_pattern_3_generate_invoke,
        test_official_pattern_4_generate_iter,
        test_official_pattern_5_multiple_invoke,
    ]
    
    results = []
    for pattern in patterns:
        try:
            success = pattern()
            results.append(success)
        except Exception as e:
            print(f"❌ Pattern crashed: {e}")
            results.append(False)
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} official patterns worked")
    print("These are the patterns I should have tested first!")

if __name__ == "__main__":
    main()