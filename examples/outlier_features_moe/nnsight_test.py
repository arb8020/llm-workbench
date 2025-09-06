import torch
from nnsight import LanguageModel

def explore_model_structure(model_name="allenai/OLMoE-1B-7B-0125-Instruct", layer_idx=0):
    """
    Explore the structure of a specific layer to find all available components
    """
    print(f"Loading model: {model_name}")
    llm = LanguageModel(model_name, device_map="auto", torch_dtype="auto")
    
    print(f"\n=== OVERALL MODEL STRUCTURE ===")
    print("Top-level attributes:")
    for attr in sorted(dir(llm.model)):
        if not attr.startswith('_'):
            try:
                obj = getattr(llm.model, attr)
                if hasattr(obj, '__len__') and not isinstance(obj, str):
                    print(f"  {attr}: {type(obj)} (len={len(obj)})")
                else:
                    print(f"  {attr}: {type(obj)}")
            except:
                print(f"  {attr}: <error accessing>")
    
    print(f"\n=== LAYER {layer_idx} STRUCTURE ===")
    layer = llm.model.layers[layer_idx]
    
    def explore_module(module, name="", depth=0, max_depth=3):
        indent = "  " * depth
        print(f"{indent}{name}: {type(module).__name__}")
        
        if depth >= max_depth:
            return
            
        for attr_name in sorted(dir(module)):
            if attr_name.startswith('_'):
                continue
                
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, torch.nn.Module):
                    full_name = f"{name}.{attr_name}" if name else attr_name
                    explore_module(attr, full_name, depth + 1, max_depth)
                elif callable(attr) and not isinstance(attr, type):
                    continue  # Skip methods
                else:
                    try:
                        attr_info = str(type(attr))
                        if hasattr(attr, 'shape'):
                            attr_info += f" shape={attr.shape}"
                        print(f"{indent}  {attr_name}: {attr_info}")
                    except:
                        pass
            except:
                continue
    
    explore_module(layer)
    
    print(f"\n=== ATTENTION STRUCTURE ===")
    if hasattr(layer, 'self_attn'):
        explore_module(layer.self_attn, "self_attn", depth=0, max_depth=2)
    
    print(f"\n=== MLP STRUCTURE ===")
    if hasattr(layer, 'mlp'):
        explore_module(layer.mlp, "mlp", depth=0, max_depth=2)
    
    print(f"\n=== LAYER NORM CANDIDATES ===")
    for attr_name in dir(layer):
        if 'norm' in attr_name.lower() and not attr_name.startswith('_'):
            attr = getattr(layer, attr_name)
            print(f"  {attr_name}: {type(attr)}")
    
    return llm

if __name__ == "__main__":
    model = explore_model_structure()
