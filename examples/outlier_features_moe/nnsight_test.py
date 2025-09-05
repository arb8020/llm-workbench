import os
import torch
from nnsight import LanguageModel
import time

def extract_and_print_activations(model_name="mistralai/Mixtral-8x7B-v0.1", 
                                 text="The quick brown fox jumps over the lazy dog.", 
                                 layer_idx=3):
    """
    Simple function to extract and print residual stream activations before layer k.
    """
    start_time = time.time()
    
    # Check for HuggingFace token
    if not os.environ.get('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set.")
    
    print(f"[{time.time()-start_time:.1f}s] Starting model initialization: {model_name}")
    print("DEBUG: About to call LanguageModel() - this will download if not cached...")
    
    # This step includes download + creating meta model
    model = LanguageModel(model_name, device_map="auto")
    
    print(f"[{time.time()-start_time:.1f}s] DEBUG: LanguageModel object created successfully")
    print(f"Model is on device: {next(model.parameters()).device}")  # Should show "meta"
    
    # Tokenization (fast)
    print(f"[{time.time()-start_time:.1f}s] DEBUG: Starting tokenization...")
    print(f"Input text: '{text}'")
    inputs = model.tokenizer(text, return_tensors="pt")
    print(f"[{time.time()-start_time:.1f}s] DEBUG: Tokenization complete. Shape: {inputs['input_ids'].shape}")
    
    # This is where the real loading happens
    print(f"[{time.time()-start_time:.1f}s] DEBUG: About to enter trace context - MODEL WILL LOAD TO REAL DEVICES NOW")
    print("This is the slow step - model loading from meta to actual hardware...")
    
    with model.trace(inputs) as tracer:
        print(f"[{time.time()-start_time:.1f}s] DEBUG: Inside trace context - model is now loaded!")
        print(f"Model is now on device: {next(model.parameters()).device}")  # Should show actual device
        
        print(f"[{time.time()-start_time:.1f}s] DEBUG: Setting up activation extraction for layer {layer_idx}...")
        
        if layer_idx == 0:
            # Before first layer - use embeddings
            activations = model.model.embed_tokens.output
        else:
            # Before layer k - get output of layer k-1 (which is input to layer k)
            activations = model.model.layers[layer_idx - 1].output[0]
        
        # Save the activations tensor
        activations.save()
        print(f"[{time.time()-start_time:.1f}s] DEBUG: Activation saving set up, about to exit trace context...")
    
    # Trace context has exited - inference is complete
    print(f"[{time.time()-start_time:.1f}s] DEBUG: Trace context exited - inference complete!")
    
    # Print activation details
    print(f"[{time.time()-start_time:.1f}s] Processing results...")
    print(f"Activation shape: {activations.value.shape}")
    print(f"Activation dtype: {activations.value.dtype}")
    print(f"Activation device: {activations.value.device}")
    
    # Print some basic stats
    act_tensor = activations.value
    print(f"\nActivation statistics:")
    print(f"  Min: {act_tensor.min().item():.4f}")
    print(f"  Max: {act_tensor.max().item():.4f}")
    print(f"  Mean: {act_tensor.mean().item():.4f}")
    print(f"  Std: {act_tensor.std().item():.4f}")
    
    # Print first few values as sample
    print(f"\nFirst 5 values of first token, first 10 dimensions:")
    print(act_tensor[0, 0, :10])

    torch.save(activations.value, f'activations_layer{layer_idx}.pt')
    print(f"Activations saved to: activations_layer{layer_idx}.pt")
    
    total_time = time.time() - start_time
    print(f"\n[COMPLETE] Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    return activations.value

if __name__ == "__main__":
    # Simple usage
    activations = extract_and_print_activations(
        model_name="allenai/OLMoE-1B-7B-0125-Instruct",
        text="Hello world, this is a test.",
        layer_idx=3
    )
