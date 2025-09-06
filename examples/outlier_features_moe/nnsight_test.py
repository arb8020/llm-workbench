import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchinfo import summary
import torch

def explore_model_structure(model_name="allenai/OLMoE-1B-7B-0125-Instruct", layer_idx=0):
    """
    Explore the structure of a model using torchinfo
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # Create sample input
    sample_text = "Hello world"
    inputs = tokenizer(sample_text, return_tensors="pt")
    
    print("\n=== MODEL SUMMARY ===")
    # Show overall model structure
    print(summary(model, 
                 input_data=dict(**inputs),
                 depth=3, 
                 col_names=("input_size", "output_size", "num_params")))
    
    print("\n=== LAYER DETAIL ===")
    # Show specific layer structure if available
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layer = model.transformer.h[layer_idx]
        # Analyze the layer structure
        print(summary(layer,
                     input_data=(inputs['input_ids'],),  
                     depth=3,
                     col_names=("input_size", "output_size", "num_params")))
        
        # Show attention mechanism
        if hasattr(layer, 'attn'):
            print("\n=== ATTENTION MECHANISM ===")
            print(summary(layer.attn,
                        input_data=(inputs['input_ids'],),
                        depth=2,
                        col_names=("input_size", "output_size", "num_params")))
            
        # Show MLP structure
        if hasattr(layer, 'mlp'):
            print("\n=== MLP STRUCTURE ===")
            print(summary(layer.mlp,
                        input_data=(inputs['input_ids'],),
                        depth=2,
                        col_names=("input_size", "output_size", "num_params")))
    
    return model

if __name__ == "__main__":
    model = explore_model_structure()
