
import os
import torch
from nnsight import LanguageModel
import time

def extract_and_print_activations(model_name="mistralai/Mixtral-8x7B-v0.1",
                                  text="The quick brown fox jumps over the lazy dog.",
                                  layer_idx=3):
    """
    Extract and print residual stream activations *before* layer k.
    For k > 0 we hook layers[k].input (i.e., what goes into block k).
    For k == 0 we use the token embeddings.
    """
    start_time = time.time()

    if not os.environ.get('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set.")

    print(f"[{time.time()-start_time:.1f}s] Starting model initialization: {model_name}")
    print("DEBUG: About to call LanguageModel() - this will download if not cached...")

    print(f'using model: {model_name}')
    model = LanguageModel(model_name, device_map="auto")

    print(f"[{time.time()-start_time:.1f}s] DEBUG: LanguageModel object created successfully")
    print(f"Model is on device: {next(model.parameters()).device}")  # likely 'meta' pre-trace

    print(f"[{time.time()-start_time:.1f}s] DEBUG: Starting tokenization...")
    print(f"Input text: '{text}'")
    inputs = model.tokenizer(text, return_tensors="pt")
    print(f"[{time.time()-start_time:.1f}s] DEBUG: Tokenization complete. Shape: {inputs['input_ids'].shape}")

    print(f"[{time.time()-start_time:.1f}s] DEBUG: About to enter trace context - MODEL WILL LOAD TO REAL DEVICES NOW")
    print("This is the slow step - model loading from meta to actual hardware...")

    with model.trace(inputs) as tracer:
        print(f"[{time.time()-start_time:.1f}s] DEBUG: Inside trace context - model is now loaded!")
        print(f"Model is now on device: {next(model.parameters()).device}")

        print(f"[{time.time()-start_time:.1f}s] DEBUG: Setting up activation extraction for layer {layer_idx}...")

        if layer_idx == 0:
            node = model.model.embed_tokens.output              # embeddings (batch, seq, d_model)
        else:
            node = model.model.layers[layer_idx].input[0]       # input to block k (batch, seq, d_model)

        node.save()
        print(f"[{time.time()-start_time:.1f}s] DEBUG: Activation saving set up, about to exit trace context...")

    print(f"[{time.time()-start_time:.1f}s] DEBUG: Trace context exited - inference complete!")
    print(f"[{time.time()-start_time:.1f}s] Processing results...")

    act_tensor = node.value                                    # <-- valid here
    print(f"Activation shape: {tuple(act_tensor.shape)}")
    print(f"Activation dtype: {act_tensor.dtype}")
    print(f"Activation device: {act_tensor.device}")

    print(f"\nActivation statistics:")
    print(f"  Min:  {act_tensor.min().item():.4f}")
    print(f"  Max:  {act_tensor.max().item():.4f}")
    print(f"  Mean: {act_tensor.mean().item():.4f}")
    print(f"  Std:  {act_tensor.std().item():.4f}")

    print(f"\nFirst 5 values of first token, first 10 dimensions:")
    print(act_tensor[0, 0, :10])

    torch.save(act_tensor, f'activations_layer{layer_idx}.pt')
    print(f"Activations saved to: activations_layer{layer_idx}.pt")

    total_time = time.time() - start_time
    print(f"\n[COMPLETE] Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    return act_tensor

if __name__ == "__main__":
    activations = extract_and_print_activations(
        model_name="allenai/OLMoE-1B-7B-0125-Instruct",
        text="Hello world, this is a test.",
        layer_idx=3
    )

