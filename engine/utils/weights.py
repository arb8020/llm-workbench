# weights.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
import pathlib
import numpy as np

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
        if k.endswith(".ln_1.weight"): out[k.replace(".ln_1.weight", ".ln1.gamma")] = out[k]
        if k.endswith(".ln_1.bias"):   out[k.replace(".ln_1.bias",   ".ln1.beta")]  = out[k]
        if k.endswith(".ln_2.weight"): out[k.replace(".ln_2.weight", ".ln2.gamma")] = out[k]
        if k.endswith(".ln_2.bias"):   out[k.replace(".ln_2.bias",   ".ln2.beta")]  = out[k]
        if k.endswith(".ln_f.weight"): out[k.replace(".ln_f.weight", ".lnf.gamma")] = out[k]
        if k.endswith(".ln_f.bias"):   out[k.replace(".ln_f.bias",   ".lnf.beta")]  = out[k]
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
# Helper to download from HuggingFace
# -----------------------------

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

# -----------------------------
# Weight shape inspection
# -----------------------------

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