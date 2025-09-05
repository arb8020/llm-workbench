# weights.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
import pathlib
import numpy as np
import re

# -----------------------------
# GPT-2 Weight Loading
# -----------------------------

@dataclass(frozen=True)
class GPT2Weights:
    """
    Immutable struct holding raw numpy arrays keyed by canonical GPT-2 names.
    Keeps it generic so your pure-numpy inference can wire these in.
    """
    params: Dict[str, np.ndarray]  # e.g., "wte", "wpe", "h.0.attn.c_attn.weight", ...

def _maybe_load_safetensors(model_dir: pathlib.Path) -> Dict[str, np.ndarray] | None:
    try:
        from safetensors.numpy import load_file
    except ImportError:
        return None
        
    # Check for single file
    for fname in ["model.safetensors", "pytorch_model.safetensors"]:
        path = model_dir / fname
        if path.exists():
            return load_file(str(path))
            
    # Check for sharded files
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        out = {}
        for shard in shards:
            out.update(load_file(str(shard)))
        return out
    return None

def _maybe_load_pytorch_bin(model_dir: pathlib.Path) -> Dict[str, np.ndarray] | None:
    try:
        import torch
    except ImportError:
        return None
        
    bin_path = model_dir / "pytorch_model.bin"
    if not bin_path.exists():
        return None
        
    state = torch.load(str(bin_path), map_location="cpu")
    return {k: v.detach().cpu().numpy() for k, v in state.items()}

def _maybe_load_npz(model_dir: pathlib.Path) -> Dict[str, np.ndarray] | None:
    # Support a single-file export you may have created yourself.
    npzs = list(model_dir.glob("*.npz"))
    if not npzs:
        return None
    data = np.load(str(npzs[0]))
    return {k: data[k] for k in data.files}

def _canonicalize(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Keep HF naming, but surface a few friendly aliases.
    """
    out = dict(params)
    # Friendly aliases for embeddings
    if "transformer.wte.weight" in out:
        out["wte"] = out["transformer.wte.weight"]
    if "transformer.wpe.weight" in out:
        out["wpe"] = out["transformer.wpe.weight"]
    # Layer norm aliases
    for k in list(out.keys()):
        if k.endswith(".ln_1.weight"):
            out[k.replace(".ln_1.weight", ".ln1.gamma")] = out[k]
        if k.endswith(".ln_1.bias"):
            out[k.replace(".ln_1.bias", ".ln1.beta")] = out[k]
        if k.endswith(".ln_2.weight"):
            out[k.replace(".ln_2.weight", ".ln2.gamma")] = out[k]
        if k.endswith(".ln_2.bias"):
            out[k.replace(".ln_2.bias", ".ln2.beta")] = out[k]
        if k.endswith(".ln_f.weight"):
            out[k.replace(".ln_f.weight", ".lnf.gamma")] = out[k]
        if k.endswith(".ln_f.bias"):
            out[k.replace(".ln_f.bias", ".lnf.beta")] = out[k]
    return out

def load_gpt2_weights(model_dir: str | pathlib.Path) -> GPT2Weights:
    """
    Load GPT-2 weights from a directory containing model files.
    Supports safetensors, pytorch_model.bin, or .npz formats.
    """
    model_dir = pathlib.Path(model_dir)
    
    loaders = [_maybe_load_safetensors, _maybe_load_pytorch_bin, _maybe_load_npz]
    for loader in loaders:
        params = loader(model_dir)
        if params is not None:
            return GPT2Weights(params=_canonicalize(params))
    
    raise FileNotFoundError(f"No supported weight files found in {model_dir}")

# -----------------------------
# LLaMA Weight Loading
# -----------------------------

@dataclass(frozen=True)
class LlamaWeights:
    """
    Immutable struct holding raw numpy arrays keyed by canonical LLaMA names.
    Keeps it generic so your pure-numpy inference can wire these in.
    """
    params: Dict[str, np.ndarray]  # e.g., "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", ...

def _maybe_load_safetensors_llama(model_dir: pathlib.Path) -> Dict[str, np.ndarray] | None:
    try:
        from safetensors.numpy import load_file
    except ImportError:
        return None
        
    # Check for single file
    for fname in ["model.safetensors", "pytorch_model.safetensors"]:
        path = model_dir / fname
        if path.exists():
            return load_file(str(path))
            
    # Check for sharded files
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        out = {}
        for shard in shards:
            out.update(load_file(str(shard)))
        return out
    return None

def _maybe_load_pytorch_bin_llama(model_dir: pathlib.Path) -> Dict[str, np.ndarray] | None:
    try:
        import torch
    except ImportError:
        return None
        
    bin_path = model_dir / "pytorch_model.bin"
    if not bin_path.exists():
        return None
        
    state = torch.load(str(bin_path), map_location="cpu")
    return {k: v.detach().cpu().numpy() for k, v in state.items()}

def _maybe_load_npz_llama(model_dir: pathlib.Path) -> Dict[str, np.ndarray] | None:
    # Support a single-file export you may have created yourself.
    npzs = list(model_dir.glob("*.npz"))
    if not npzs:
        return None
    data = np.load(str(npzs[0]))
    return {k: data[k] for k in data.files}

def _canonicalize_llama(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Keep HF naming, but surface a few friendly aliases for LLaMA.
    """
    out = dict(params)
    # Friendly aliases for embeddings and common layers
    if "model.embed_tokens.weight" in out:
        out["embed_tokens"] = out["model.embed_tokens.weight"]
    if "model.norm.weight" in out:
        out["norm"] = out["model.norm.weight"]
    if "lm_head.weight" in out:
        out["lm_head"] = out["lm_head.weight"]
    
    # Add layer-specific aliases for easier access
    for k in list(out.keys()):
        if "model.layers." in k:
            # Extract layer number and component
            parts = k.split(".")
            if len(parts) >= 4:
                layer_num = parts[2]
                component = ".".join(parts[3:])
                alias = f"layer_{layer_num}.{component}"
                out[alias] = out[k]
    
    return out

def load_llama_weights(model_dir: str | pathlib.Path) -> LlamaWeights:
    """
    Load LLaMA weights from a directory containing model files.
    Supports safetensors, pytorch_model.bin, or .npz formats.
    """
    model_dir = pathlib.Path(model_dir)
    
    loaders = [_maybe_load_safetensors_llama, _maybe_load_pytorch_bin_llama, _maybe_load_npz_llama]
    for loader in loaders:
        params = loader(model_dir)
        if params is not None:
            return LlamaWeights(params=_canonicalize_llama(params))
    
    raise FileNotFoundError(f"No supported weight files found in {model_dir}")

# -----------------------------
# Helper to download from HuggingFace
# -----------------------------

def download_llama_weights(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", cache_dir: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    """
    Download LLaMA model weights from HuggingFace.
    Returns path to directory containing model weights.
    """
    from huggingface_hub import snapshot_download
    
    local_dir = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.bin", "*.json"]
    )
    
    return pathlib.Path(local_dir)

def download_gpt2_weights(model_name: str = "gpt2", cache_dir: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    """
    Download GPT-2 model weights from HuggingFace.
    Returns path to directory containing model weights.
    """
    from huggingface_hub import snapshot_download
    
    local_dir = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.bin", "*.json"]
    )
    
    return pathlib.Path(local_dir)

def validate_gpt2_weights(weights: Dict[str, any], expected_keys: list[str]) -> None:
    """
    Validate that all expected keys exist in weights dictionary.
    Provides helpful error messages with suggestions for missing keys.
    """
    missing_keys = []
    available_keys = set(weights.keys())
    
    for key in expected_keys:
        if key not in weights:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing weight keys: {missing_keys}")
        print(f"📋 Available keys: {sorted(available_keys)}")
        
        # Try to suggest alternatives
        suggestions = {}
        for missing in missing_keys:
            closest_matches = [k for k in available_keys if missing.split('.')[-1] in k]
            if closest_matches:
                suggestions[missing] = closest_matches[:3]  # Top 3 suggestions
        
        if suggestions:
            print("\n💡 Suggested alternatives:")
            for missing, matches in suggestions.items():
                print(f"  '{missing}' -> try: {matches}")
        
        raise KeyError(f"Missing required weight keys: {missing_keys}")
    
    print("✅ All required weight keys found")

def validate_llama_weights(weights: Dict[str, any], expected_keys: list[str]) -> None:
    """
    Validate that all expected keys exist in weights dictionary.
    Provides helpful error messages with suggestions for missing keys.
    """
    missing_keys = []
    available_keys = set(weights.keys())
    
    for key in expected_keys:
        if key not in weights:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing weight keys: {missing_keys}")
        print(f"📋 Available keys: {sorted(available_keys)}")
        
        # Try to suggest alternatives
        suggestions = {}
        for missing in missing_keys:
            closest_matches = [k for k in available_keys if missing.split('.')[-1] in k]
            if closest_matches:
                suggestions[missing] = closest_matches[:3]  # Top 3 suggestions
        
        if suggestions:
            print("\n💡 Suggested alternatives:")
            for missing, matches in suggestions.items():
                print(f"  '{missing}' -> try: {matches}")
        
        raise KeyError(f"Missing required weight keys: {missing_keys}")
    
    print("✅ All required weight keys found")

def load_and_print_llama_weights_jax() -> Dict[str, 'Array']:
    import jax
    import jax.numpy as jnp
    """Load and analyze LLaMA model weights from any source."""
    
    model_dir = download_llama_weights("meta-llama/Llama-3.1-8B-Instruct")
    weights_obj = load_llama_weights(model_dir)
    
    # Convert to JAX arrays 
    weights = {name: jnp.array(param) for name, param in weights_obj.params.items()}
    
    # Group parameters by layer type for cleaner printing
    layer_groups = {}
    for name, param in weights.items():
        # Extract layer type and number using regex for LLaMA structure
        match = re.match(r'model\.layers\.(\d+)\.(.+)', name)
        if match:
            layer_num, rest = match.groups()
            # Take the full remaining path as the key to preserve nested names
            key = rest
            if key not in layer_groups:
                layer_groups[key] = {}
            layer_groups[key][int(layer_num)] = param.shape
        else:
            # Non-layer parameters (embeddings, etc)
            layer_groups[name] = {0: param.shape}

    # Print structured parameter info
    print("\n🔍 LLaMA Model Architecture:")
    for group_name, layers in layer_groups.items():
        # Get list of layer indices
        indices = sorted(layers.keys())
        # All shapes should be identical
        shape = layers[indices[0]]
        # Format indices as comma-separated list
        if len(indices) > 5:
            idx_str = f"0-{max(indices)} ({len(indices)} layers)"
        else:
            idx_str = ",".join(str(i) for i in indices)
        print(f"  {group_name}: {shape} (layers {idx_str})")
    
    total_params = sum(p.size for p in weights.values())
    print(f"\nTotal parameters: {total_params:,}")
    
    return weights

def load_and_print_gpt2_weights_jax() -> Dict[str, 'Array']:
    import jax
    import jax.numpy as jnp
    """Load and analyze model weights from any source."""
    
    model_dir = download_gpt2_weights("gpt2")
    weights_obj = load_gpt2_weights(model_dir)
    
    # Convert to JAX arrays 
    weights = {name: jnp.array(param) for name, param in weights_obj.params.items()}
    
    # Group parameters by layer type for cleaner printing
    layer_groups = {}
    for name, param in weights.items():
        # Extract layer type and number using regex
        match = re.match(r'h\.(\d+)\.(.+)', name)
        if match:
            layer_num, rest = match.groups()
            # Take the full remaining path as the key to preserve nested names
            key = rest
            if key not in layer_groups:
                layer_groups[key] = {}
            layer_groups[key][int(layer_num)] = param.shape
        else:
            # Non-layer parameters (embeddings, etc)
            layer_groups[name] = {0: param.shape}

    # Print structured parameter info
    print("\n🔍 Model Architecture:")
    for group_name, layers in layer_groups.items():
        # Get list of layer indices
        indices = sorted(layers.keys())
        # All shapes should be identical
        shape = layers[indices[0]]
        # Format indices as comma-separated list
        idx_str = ",".join(str(i) for i in indices)
        print(f"  {group_name}: {shape} (layers {idx_str})")
    
    total_params = sum(p.size for p in weights.values())
    print(f"\nTotal parameters: {total_params:,}")
    
    return weights

# -----------------------------
# Weight shape inspection
# -----------------------------

def _shape_hints_llama(weights: LlamaWeights) -> Dict[str, Tuple[int, ...]]:
    # Pull a few common shapes to sanity-check the checkpoint.
    hints = {}
    for name in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight", 
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]:
        if name in weights.params:
            hints[name] = tuple(weights.params[name].shape)
    return hints

def _shape_hints(weights: GPT2Weights) -> Dict[str, Tuple[int, ...]]:
    # Pull a few common shapes to sanity-check the checkpoint.
    hints = {}
    for name in [
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.ln_f.weight",
    ]:
        if name in weights.params:
            hints[name] = tuple(weights.params[name].shape)
    return hints

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, help="Path to HF GPT-2 model dir or .npz (auto-downloads if not provided)")
    ap.add_argument("--download", type=str, default="gpt2", help="HF model name to download weights from (default: gpt2)")
    args = ap.parse_args()

    model_dir = pathlib.Path(args.model) if args.model else download_gpt2_weights(args.download)
    if not args.model:
        print(f"Weights downloaded to: {model_dir}")

    w = load_gpt2_weights(model_dir)
    hints = _shape_hints(w)
    for k, v in hints.items():
        print(f"{k}: {v}")
    print("Weights loaded ✅  (above are a few shape checks)")
