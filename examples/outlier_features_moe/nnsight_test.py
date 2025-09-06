import os
import time
import torch
from nnsight import LanguageModel

def extract_targeted_activations(
    model_name="allenai/OLMoE-1B-7B-0125-Instruct",
    text="Hello world, this is a test.",
    target_layers=(0, 1, 2, 3),
    capture_router_logits=True,
    capture_experts=False,          # set True to also hook per-expert MLPs (may be many)
    experts_to_capture=(0,),        # which expert indices to try to capture if capture_experts=True
    save_file="targeted_activations.pt",
):
    t0 = time.time()

    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN environment variable not set.")

    print(f"[{time.time()-t0:.1f}s] Loading model: {model_name}")
    llm = LanguageModel(model_name, device_map="auto")

    # Just to show token shape (trace will handle tokenization itself)
    token_info = llm.tokenizer(text, return_tensors="pt")
    print(f"Input text: '{text}'")
    print(f"Token shape: {tuple(token_info['input_ids'].shape)}")

    # Hold saved proxies (inside trace) so we can read .value after
    saved = {}

    print(f"[{time.time()-t0:.1f}s] Setting up hooks...")
    with llm.trace(text) as tracer:
        n_layers = len(llm.model.layers)
        print(f"Model has {n_layers} layers; extracting from: {list(target_layers)}")

        for L in target_layers:
            if L >= n_layers:
                print(f"  Warning: layer {L} >= n_layers ({n_layers}) – skipping.")
                continue

            layer = llm.model.layers[L]
            layer_saved = {}

            # Attention projections: capture INPUT to each Linear
            # Note: inputs are tuples, so index [0]
            q_in = layer.self_attn.q_proj.input[0].save()
            k_in = layer.self_attn.k_proj.input[0].save()
            v_in = layer.self_attn.v_proj.input[0].save()
            o_in = layer.self_attn.o_proj.input[0].save()

            layer_saved["q_proj_input"] = q_in
            layer_saved["k_proj_input"] = k_in
            layer_saved["v_proj_input"] = v_in
            layer_saved["o_proj_input"] = o_in

            # MoE: router (gate) input and optionally router logits
            gate_in = layer.mlp.gate.input[0].save()
            layer_saved["gate_input"] = gate_in

            if capture_router_logits:
                router_logits = layer.mlp.gate.output.save()
                layer_saved["router_logits"] = router_logits

            # Optional: capture per‑expert MLP projection inputs (only experts that are actually executed will have values)
            if capture_experts:
                exp_dict = {}
                for e in experts_to_capture:
                    try:
                        exp = layer.mlp.experts[e]
                        exp_dict[f"expert_{e}_gate_proj_input"] = exp.gate_proj.input[0].save()
                        exp_dict[f"expert_{e}_up_proj_input"]   = exp.up_proj.input[0].save()
                        exp_dict[f"expert_{e}_down_proj_input"] = exp.down_proj.input[0].save()
                    except Exception as err:
                        print(f"    Could not set expert hooks for layer {L}, expert {e}: {err}")
                if exp_dict:
                    layer_saved["experts"] = exp_dict

            saved[f"layer_{L}"] = layer_saved

    print(f"[{time.time()-t0:.1f}s] Trace complete; extracting tensors...")

    # Pull tensors off proxies
    extracted = {}
    for layer_name, comps in saved.items():
        print(f"\nProcessing {layer_name}:")
        layer_out = {}
        for name, proxy in comps.items():
            if isinstance(proxy, dict):  # experts sub-dict
                sub_out = {}
                for ename, eproxy in proxy.items():
                    try:
                        t = eproxy.value
                        tcpu = t.detach().cpu()
                        stats = tcpu.float() if tcpu.dtype in (torch.float16, torch.bfloat16) else tcpu
                        print(f"  {ename}: shape={tuple(tcpu.shape)} dtype={tcpu.dtype} "
                              f"min={stats.min().item():.3f} max={stats.max().item():.3f} mean={stats.mean().item():.3f}")
                        sub_out[ename] = tcpu
                    except Exception as e:
                        print(f"  {ename}: not available (module likely not executed for this input) - {e}")
                layer_out[name] = sub_out
                continue

            try:
                t = proxy.value
                tcpu = t.detach().cpu()
                stats = tcpu.float() if tcpu.dtype in (torch.float16, torch.bfloat16) else tcpu
                print(f"  {name}: shape={tuple(tcpu.shape)} dtype={tcpu.dtype} "
                      f"min={stats.min().item():.3f} max={stats.max().item():.3f} mean={stats.mean().item():.3f}")
                layer_out[name] = tcpu
            except Exception as e:
                print(f"  {name}: ERROR - {e}")

        extracted[layer_name] = layer_out

    torch.save(extracted, save_file)
    print(f"\nSaved to: {save_file}")
    print(f"[COMPLETE] Total runtime: {time.time()-t0:.1f}s")
    return extracted

if __name__ == "__main__":
    _ = extract_targeted_activations(
        target_layers=(0, 1, 2, 3),
        capture_router_logits=True,
        capture_experts=False,      # flip to True if you want per‑expert inputs too
    )
