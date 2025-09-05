import os
import nnsight
from nnsight import LanguageModel
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Analyze language model architecture')
    parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x7B-v0.1",
                      help='HuggingFace model identifier')
    args = parser.parse_args()

    # Check for HuggingFace token
    if not os.environ.get('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set. Some models may not be accessible.")
    
    model = LanguageModel(args.model, device_map="auto")

    # Group by everything after the layer number
    layer_components = defaultdict(list)

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            # Split on common layer patterns
            import re
            parts = re.split(r'\.(\d+)\.', name)
            if len(parts) > 2:
                # parts[0] = prefix, parts[1] = layer_num, parts[2] = component
                component = parts[2]
                if component not in layer_components[parts[0]]:
                    layer_components[parts[0]].append(component)
                    print(f"{parts[0]}.N.{component}: {module.weight.shape}")

if __name__ == "__main__":
    main()
