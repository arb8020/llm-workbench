
import os
import time
import torch
from nnsight import LanguageModel

def extract_and_print_activations(model_name="mistralai/Mixtral-8x7B-v0.1",
                                  text="The quick brown fox jumps over the lazy dog.",
                                  layer_idx=3):
    """
    Extract and print residual stream activations before layer k.
    For k > 0 we hook layers[k].input (i.e., what goes into block k).
    For k == 0 we use the token embeddings.
    """
    start_time = time.time()

    if not os.environ.get('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set.")

    print(f"[{time.time()-start_time:.1f}s] Starting model initialization: {model_name}")
    print("DEBUG: About to call LanguageModel() - this will download if not cached...")

    print(f"using model: {model_name}")
    model = LanguageModel(model_name, device_map="auto")

    print(f"[{time.time()-start_time:.1f}s] DEBUG: LanguageModel object created successfully")
    try:
        print(f"Model is on device: {next(model.parameters()).device}")
    except StopIteration:
        print("Model parameters not initialized yet (meta).")

    print(f"[{time.time()-start_time:.1f}s] DEBUG: Starting tokenization...")
    print(f"Input text: '{text}'")
    inputs = model.tokenizer(text, return_tensors="pt")
    print(f"[{time.time()-start_time:.1f}s] DEBUG: Tokenization complete. Shape: {inputs['input_ids'].shape}")

    print(f"[{time.time()-start_time:.1f}s] DEBUG: About to enter trace context - MODEL WILL LOAD TO REAL DEVICES NOW")
    print("This is the slow step - model loading from meta to actual hardware...")

    with model.trace(inputs) as tracer:
        print(f"[{time.time()-start_time:.1f}s] DEBUG: Inside trace context - model is now loaded!")
        try:
            print(f"Model is now on device: {next(model.parameters()).device}")
        except StopIteration:
            pass

        # Validate layer index
        n_layers = len(model.model.layers)
        if layer_idx < 0 or layer_idx > n_layers:
            raise ValueError(f"layer_idx must be in [0, {n_layers}] (0 for embeddings). Got {layer_idx}.")

        print(f"[{time.time()-start_time:.1f}s] DEBUG: Setting up activation extraction for layer {layer_idx}...")

        if layer_idx == 0:
            # Output of the embedding lookup (batch, seq, d_model)
            node = model.model.embed_tokens.output
        else:
            layer_mod = model.model.layers[layer_idx]
            # nnsight versions may expose either .input or .inputs
            if hasattr(layer_mod, "input"):
                node = layer_mod.input[0]
            elif hasattr(layer_mod, "inputs"):
                node = layer_mod.inputs[0]
            else:
                raise RuntimeError("Could not find layer input on this nnsight version.")

        node.save()
        print(f"[{time.time()-start_time:.1f}s] DEBUG: Activation saving set up, about to exit trace context...")

    print(f"[{time.time()-start_time:.1f}s] DEBUG: Trace context exited - inference complete!")
    print(f"[{time.time()-start_time:.1f}s] Processing results...")

    # In some nnsight versions, `node` is already a Tensor after the trace.
    # In others, it's a Node with a .value property. Handle both:
    act_tensor = getattr(node, "value", node)

    # Move to CPU for printing/saving (and optionally cast for stable stats)
    act_for_stats = act_tensor.detach().cpu()
    stats_tensor = act_for_stats.float() if act_for_stats.dtype in (torch.float16, torch.bfloat16) else act_for_stats

    print(f"Activation shape: {tuple(act_for_stats.shape)}")
    print(f"Activation dtype: {act_for_stats.dtype}")
    print(f"Activation device: {act_for_stats.device}")

    print(f"\nActivation statistics:")
    print(f"  Min:  {stats_tensor.min().item():.4f}")
    print(f"  Max:  {stats_tensor.max().item():.4f}")
    print(f"  Mean: {stats_tensor.mean().item():.4f}")
    print(f"  Std:  {stats_tensor.std().item():.4f}")

    out_path = f'activations_layer{layer_idx}.pt'
    torch.save(act_for_stats, out_path)
    print(f"Activations saved to: {out_path}")

    total_time = time.time() - start_time
    print(f"\n[COMPLETE] Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    return act_for_stats

if __name__ == "__main__":
    activations = extract_and_print_activations(
        model_name="allenai/OLMoE-1B-7B-0125-Instruct",
        text="Hello world, this is a test.",
        layer_idx=3
    )
