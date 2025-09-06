# fix load_activations to work with extract_activations
import os
import json
import torch
import numpy as np
from nnsight import LanguageModel
from pathlib import Path
from datetime import datetime

def extract_activations(
    model_name="allenai/OLMoE-1B-7B-0125-Instruct",
    text="Hello world, this is a test.",
    layers=[0, 1],
    save_dir="./activations"
):
    """
    Extract and save activations from specified layers.
    
    Args:
        model_name: HuggingFace model identifier
        text: Input text to process
        layers: List of layer indices to extract from
        save_dir: Base directory to save activations
    """
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    llm = LanguageModel(model_name, device_map="auto")
    
    # Extract activations
    activations = {}
    with llm.trace(text) as tracer:
        for layer_idx in layers:
            ln_into_attn = llm.model.layers[layer_idx].input_layernorm.output.save()
            ln_into_mlp = llm.model.layers[layer_idx].post_attention_layernorm.output.save()
            
            activations[f"layer_{layer_idx}_ln_attn"] = ln_into_attn
            activations[f"layer_{layer_idx}_ln_mlp"] = ln_into_mlp
        
    # Save activations and metadata
    saved_files = []
    for layer_name, activation_proxy in activations.items():
        # Get the actual tensor
        tensor = activation_proxy.detach().cpu()
        
        # Save activation
        activation_file = run_dir / f"{layer_name}_activations.pt"
        torch.save(tensor, activation_file)
        saved_files.append(str(activation_file))
        
        print(f"Saved {layer_name}: shape={tuple(tensor.shape)} -> {activation_file}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "input_text": text,
        "layers_extracted": layers,
        "timestamp": timestamp, 
        "saved_files": saved_files,
        # Updated to use the actual activation keys from our dictionary
        "shapes": {name: list(activation_proxy.detach().cpu().shape) for name, activation_proxy in activations.items()}
    }
        
    metadata_file = run_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"All files saved in: {run_dir}")
    
    return run_dir, metadata

def load_activations(run_dir):
    """Load activations from a run directory."""
    run_path = Path(run_dir)
    
    # Load metadata
    metadata_file = run_path / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)


    
    
    activations = {}
    for layer_idx in metadata["layers_extracted"]:
        # Load both layernorm activations for this layer
        ln_attn_file = run_path / f"layer_{layer_idx}_ln_attn_activations.pt"
        ln_mlp_file = run_path / f"layer_{layer_idx}_ln_mlp_activations.pt"
        
        activations[f"layer_{layer_idx}_ln_attn"] = torch.load(ln_attn_file, map_location='cpu')
        activations[f"layer_{layer_idx}_ln_mlp"] = torch.load(ln_mlp_file, map_location='cpu')
        
        print(f"Loaded layer_{layer_idx} activations:")
        print(f"  ln_attn shape={tuple(activations[f'layer_{layer_idx}_ln_attn'].shape)}")
        print(f"  ln_mlp shape={tuple(activations[f'layer_{layer_idx}_ln_mlp'].shape)}")
        
    return activations, metadata

if __name__ == "__main__":
    # Extract activations
    run_dir, metadata = extract_activations(
        text="The capital of France is Paris.",
        layers=[0, 1, 2, 3],  # Extract from first 4 layers
        save_dir="./my_activations"
    )
    
    print("\n" + "="*50)
    print("LOADING TEST")
    print("="*50)
    
    # Test loading
    loaded_activations, loaded_metadata = load_activations(run_dir)
    print(f"\nLoaded run from: {loaded_metadata['timestamp']}")
    print(f"Original text: '{loaded_metadata['input_text']}'")
