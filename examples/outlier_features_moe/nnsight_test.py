import os
import torch
from nnsight import LanguageModel

def extract_targeted_activations(
    model_name="allenai/OLMoE-1B-7B-0125-Instruct",
    text="Hello world, this is a test.",
):
    llm = LanguageModel(model_name, device_map="auto")
    saved = {}

    with llm.trace(text) as tracer:
        for L in [0, 1]:  # Only layers 0 and 1
            layer = llm.model.layers[L]
            
            # Save input layernorm weights
            input_ln = layer.input_layernorm.weight.save()
            post_attn_ln = layer.post_attention_layernorm.weight.save()
            
            saved[f"layer_{L}"] = {
                "input_layernorm_weight": input_ln,
                "post_attention_layernorm_weight": post_attn_ln
            }

    # Extract and print shapes
    for layer_name, tensors in saved.items():
        print(f"\n{layer_name}:")
        for name, proxy in tensors.items():
            tensor = proxy.value.detach().cpu()
            print(f"  {name}: shape={tuple(tensor.shape)}")

if __name__ == "__main__":
    extract_targeted_activations()
