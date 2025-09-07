import os
import json
import torch
import numpy as np
from nnsight import LanguageModel
from pathlib import Path
from datetime import datetime

def extract_activations_optimized(
    llm,  # Pre-loaded LanguageModel instance
    texts: list[str], 
    layers: list[int] = None,
    save_dir: str = "./activations",
    chunk_size: int = 8
) -> tuple:
    """
    Memory-optimized activation extraction with chunking.
    
    Args:
        llm: Pre-loaded nnsight LanguageModel
        texts: List of input texts to process
        layers: List of layer indices to extract from (None = all layers)
        save_dir: Directory to save results
        chunk_size: Number of layers to process at once
        
    Returns:
        Tuple of (run_dir, metadata)
    """
    import torch
    from pathlib import Path
    from datetime import datetime
    
    assert isinstance(texts, list), f"Expected list of texts, got {type(texts)}"
    assert len(texts) > 0, "texts cannot be empty"
    
    # Auto-detect all layers if None provided
    if layers is None:
        num_layers = len(llm.model.layers)
        layers = list(range(num_layers))
        print(f"Auto-detected {num_layers} layers: {layers[0]}-{layers[-1]}")
    else:
        assert isinstance(layers, list), f"Expected list of layers, got {type(layers)}"
        assert len(layers) > 0, "layers cannot be empty"
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Process layers in chunks to reduce peak memory
    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]
    
    saved_files = []
    layer_chunks = list(chunk_list(layers, chunk_size))
    
    print(f"Processing {len(layers)} layers in {len(layer_chunks)} chunks of {chunk_size}")
    
    for chunk_idx, layers_chunk in enumerate(layer_chunks):
        print(f"  Chunk {chunk_idx + 1}/{len(layer_chunks)}: layers {layers_chunk[0]}-{layers_chunk[-1]}")
        
        # Extract activations for this chunk only
        activations = {}
        with torch.inference_mode(), llm.trace(texts) as tracer:
            for layer_idx in layers_chunk:
                ln_into_attn = llm.model.layers[layer_idx].input_layernorm.output.save()
                ln_into_mlp = llm.model.layers[layer_idx].post_attention_layernorm.output.save()
                
                activations[f"layer_{layer_idx}_ln_attn"] = ln_into_attn
                activations[f"layer_{layer_idx}_ln_mlp"] = ln_into_mlp
        
        # Immediately convert to CPU and save to disk
        for layer_name, activation_proxy in activations.items():
            tensor = activation_proxy.detach().to(torch.bfloat16).cpu()
            assert tensor.dim() == 3, f"Expected 3D tensor for {layer_name}, got shape {tensor.shape}"
            
            # Save activation
            activation_file = run_dir / f"{layer_name}_activations.pt"
            torch.save(tensor, activation_file)
            saved_files.append(str(activation_file))
            
            print(f"    Saved {layer_name}: shape={tuple(tensor.shape)} -> {activation_file}")
        
        # Clear chunk activations and GPU cache
        del activations
        torch.cuda.empty_cache()
    
    # Save metadata
    metadata = {
        "model_name": llm.tokenizer.name_or_path if hasattr(llm.tokenizer, 'name_or_path') else str(llm),
        "num_sequences": len(texts),
        "sequence_texts": texts,
        "layers_extracted": layers,  # Fixed: renamed from "layers" to "layers_extracted" for compatibility
        "chunk_size": chunk_size,
        "num_chunks": len(layer_chunks),
        "saved_files": saved_files,
        "timestamp": timestamp
    }
    
    metadata_file = run_dir / "metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata: {metadata_file}")
    
    return str(run_dir), metadata


def extract_activations_batch(model, texts: list[str], layers: list[int]) -> dict[str, torch.Tensor]:
    """
    Pure function: extract activations from batch of texts.
    
    Args:
        model: Loaded nnsight LanguageModel
        texts: List of input texts to process
        layers: List of layer indices to extract from
        
    Returns:
        Dict mapping layer names to activation tensors
    """
    assert isinstance(texts, list), f"Expected list of texts, got {type(texts)}"
    assert len(texts) > 0, "texts cannot be empty"
    assert isinstance(layers, list), f"Expected list of layers, got {type(layers)}"
    assert len(layers) > 0, "layers cannot be empty"
    
    activations = {}
    with model.trace(texts) as tracer:
        for layer_idx in layers:
            ln_into_attn = model.model.layers[layer_idx].input_layernorm.output.save()
            ln_into_mlp = model.model.layers[layer_idx].post_attention_layernorm.output.save()
            
            activations[f"layer_{layer_idx}_ln_attn"] = ln_into_attn
            activations[f"layer_{layer_idx}_ln_mlp"] = ln_into_mlp
    
    # Convert proxies to tensors
    result = {}
    for layer_name, activation_proxy in activations.items():
        tensor = activation_proxy.detach().cpu()
        assert tensor.dim() == 3, f"Expected 3D tensor for {layer_name}, got shape {tensor.shape}"
        result[layer_name] = tensor
    
    return result


def extract_activations(
    model_name="allenai/OLMoE-1B-7B-0125-Instruct",
    texts=None,
    layers=None,
    save_dir="./activations"
):
    """
    Extract and save activations from specified layers.
    
    Args:
        model_name: HuggingFace model identifier
        texts: List of input texts to process (or single string for backwards compatibility)
        layers: List of layer indices to extract from. If None, extracts from all layers.
        save_dir: Base directory to save activations
    """
    # Backwards compatibility: convert single text to list
    if isinstance(texts, str):
        texts = [texts]
    elif texts is None:
        texts = ["Hello world, this is a test."]
    
    assert isinstance(texts, list), f"Expected list of texts, got {type(texts)}"
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    llm = LanguageModel(model_name, device_map="auto")
    
    # Determine layers if not specified
    if layers is None:
        # Get all layer indices from the model
        try:
            num_layers = len(llm.model.layers)
            layers = list(range(num_layers))
            print(f"Auto-detected {num_layers} layers: {layers}")
        except AttributeError:
            # Fallback for models with different structure
            raise ValueError(f"Could not auto-detect layers for model {model_name}. Please specify --layers explicitly.")
    
    # Extract activations using pure batch function
    activations = extract_activations_batch(llm, texts, layers)
        
    # Save activations and metadata
    saved_files = []
    for layer_name, tensor in activations.items():
        # Save activation
        activation_file = run_dir / f"{layer_name}_activations.pt"
        torch.save(tensor, activation_file)
        saved_files.append(str(activation_file))
        
        print(f"Saved {layer_name}: shape={tuple(tensor.shape)} -> {activation_file}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "sequence_texts": texts,
        "batch_size": len(texts),
        "layers_extracted": layers,
        "timestamp": timestamp, 
        "saved_files": saved_files,
        "shapes": {name: list(tensor.shape) for name, tensor in activations.items()}
    }
        
    metadata_file = run_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"All files saved in: {run_dir}")
    
    return run_dir, metadata


if __name__ == "__main__":
    # Extract activations
    run_dir, metadata = extract_activations(
        texts=["The capital of France is Paris.", "Hello world, this is a test."],
        layers=[0, 1, 2, 3],  # Extract from first 4 layers
        save_dir="./my_activations"
    )