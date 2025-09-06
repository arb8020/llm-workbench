import os
import time
import torch
from nnsight import LanguageModel

def extract_targeted_activations(model_name="allenai/OLMoE-1B-7B-0125-Instruct",
                                text="Hello world, this is a test.",
                                target_layers=[0, 1, 2, 3]):
    """
    Extract activations from specific components where outliers occur:
    - Attention projections: q_proj, k_proj, v_proj, o_proj
    - MoE components: gate, experts (adapting for MoE instead of standard FFN)
    """
    start_time = time.time()

    if not os.environ.get('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set.")

    print(f"[{time.time()-start_time:.1f}s] Loading model: {model_name}")
    model = LanguageModel(model_name, device_map="auto")

    print(f"[{time.time()-start_time:.1f}s] Starting tokenization...")
    inputs = model.tokenizer(text, return_tensors="pt")
    print(f"Input text: '{text}'")
    print(f"Token shape: {inputs['input_ids'].shape}")

    # Storage for all activations
    all_activations = {}

    print(f"[{time.time()-start_time:.1f}s] Setting up activation hooks...")

    with model.trace(inputs) as tracer:
        print(f"[{time.time()-start_time:.1f}s] Inside trace context...")
        
        n_layers = len(model.model.layers)
        print(f"Model has {n_layers} layers, extracting from layers: {target_layers}")
        
        for layer_idx in target_layers:
            if layer_idx >= n_layers:
                print(f"Warning: Layer {layer_idx} doesn't exist (max: {n_layers-1})")
                continue
                
            layer = model.model.layers[layer_idx]
            layer_activations = {}
            
            # Attention projection INPUTS (activations going INTO the projection layers)
            print(f"  Setting up attention hooks for layer {layer_idx}...")
            layer.self_attn.q_proj.input.save()
            layer.self_attn.k_proj.input.save() 
            layer.self_attn.v_proj.input.save()
            layer.self_attn.o_proj.input.save()
            
            # FFN expansion layer INPUT (activations going INTO the first FFN layer)
            print(f"  Setting up MoE hooks for layer {layer_idx}...")
            layer.mlp.gate.input.save()
            # For MoE, we might also want to check experts input
            # layer.mlp.experts.input.save()
            
            # Store references for later extraction
            all_activations[f"layer_{layer_idx}"] = {
                'q_proj_input': layer.self_attn.q_proj.input,
                'k_proj_input': layer.self_attn.k_proj.input,
                'v_proj_input': layer.self_attn.v_proj.input,
                'o_proj_input': layer.self_attn.o_proj.input,
                'gate_input': layer.mlp.gate.input,
                # 'experts_input': layer.mlp.experts.input,
            }

    print(f"[{time.time()-start_time:.1f}s] Trace complete, extracting activations...")

    # Extract and process all saved activations
    extracted_activations = {}
    
    for layer_name, components in all_activations.items():
        print(f"\nProcessing {layer_name}:")
        layer_data = {}
        
        for comp_name, node in components.items():
            try:
                # Get the activation tensor
                act_tensor = getattr(node, "value", node)
                act_cpu = act_tensor.detach().cpu()
                
                # Convert to float for stable statistics
                stats_tensor = act_cpu.float() if act_cpu.dtype in (torch.float16, torch.bfloat16) else act_cpu
                
                print(f"  {comp_name}: shape={tuple(act_cpu.shape)}, "
                      f"dtype={act_cpu.dtype}, "
                      f"min={stats_tensor.min().item():.3f}, "
                      f"max={stats_tensor.max().item():.3f}, "
                      f"mean={stats_tensor.mean().item():.3f}")
                
                layer_data[comp_name] = act_cpu
                
            except Exception as e:
                print(f"  {comp_name}: ERROR - {e}")
                
        extracted_activations[layer_name] = layer_data

    # Save all activations
    save_path = 'targeted_activations.pt'
    torch.save(extracted_activations, save_path)
    print(f"\nAll activations saved to: {save_path}")

    total_time = time.time() - start_time
    print(f"\n[COMPLETE] Total runtime: {total_time:.1f} seconds")

    return extracted_activations

if __name__ == "__main__":
    activations = extract_targeted_activations(
        target_layers=[0, 1, 2, 3]  # Start with just a few layers for testing
    )
