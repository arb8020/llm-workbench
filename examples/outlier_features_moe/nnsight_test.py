import os
import torch
from nnsight import LanguageModel

def extract_and_print_activations(model_name="mistralai/Mixtral-8x7B-v0.1", 
                                 text="The quick brown fox jumps over the lazy dog.", 
                                 layer_idx=3):
    """
    Simple function to extract and print residual stream activations before layer k.
    
    Args:
        model_name: HuggingFace model identifier
        text: Input text
        layer_idx: Layer index (0-indexed, so layer_idx=3 means before layer 3)
    """
    # Check for HuggingFace token
    if not os.environ.get('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set.")
    
    print(f"Loading {model_name}...")
    model = LanguageModel(model_name, device_map="auto")
    print(f"Model loaded. Model is on device: {next(model.parameters()).device}")  # This will show which device (CPU/GPU) the model is on


        
    # Tokenize input
    print(f"Input text: '{text}'")
    inputs = model.tokenizer(text, return_tensors="pt")
    print(f"Tokenized input shape: {inputs['input_ids'].shape}")
    
    # Forward pass with tracing to get activations before layer k
    print(f"Extracting activations before layer {layer_idx}...")
    
    with model.trace(inputs) as tracer:
        if layer_idx == 0:
            # Before first layer - use embeddings
            activations = model.model.embed_tokens.output
        else:
            # Before layer k - get output of layer k-1 (which is input to layer k)
            activations = model.model.layers[layer_idx - 1].output[0]
        
        # Save the activations tensor
        activations.save()
    
    # Print activation details
    print(f"\nActivation shape: {activations.value.shape}")
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
    
    return activations.value

if __name__ == "__main__":
    # Simple usage
    activations = extract_and_print_activations(
        text="Hello world, this is a test.",
        layer_idx=3
    )
