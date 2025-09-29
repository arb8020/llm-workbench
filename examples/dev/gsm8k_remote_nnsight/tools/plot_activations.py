#!/usr/bin/env python3
"""Plot NNsight activation tensors from a local activations folder using plotext.

Usage examples:
  # Point to a folder that contains metadata.json and .pt files
  python examples/gsm8k_remote_nnsight/tools/plot_activations.py \
      --path examples/gsm8k_remote_nnsight/results/gsm8k_full_hidden_YYYYMMDD_HHMMSS/activations/gsm8k_0001 \
      --list

  # Plot mean over hidden dimension vs. tokens for a given artifact key
  python examples/gsm8k_remote_nnsight/tools/plot_activations.py \
      --path <folder-with-metadata.json> \
      --key layer_12_input_layernorm_output \
      --mode mean_hidden

  # Plot a single hidden feature vs. tokens
  python examples/gsm8k_remote_nnsight/tools/plot_activations.py \
      --path <folder-with-metadata.json> \
      --key layer_12_input_layernorm_output \
      --mode feature --feature-index 0

  # Plot histogram of values for a specific token index
  python examples/gsm8k_remote_nnsight/tools/plot_activations.py \
      --path <folder-with-metadata.json> \
      --key layer_12_input_layernorm_output \
      --mode hist --token-index -1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch


def load_metadata(dir_or_file: str) -> tuple[Dict[str, Any], Path]:
    p = Path(dir_or_file)
    if p.is_dir():
        meta_path = p / "metadata.json"
    else:
        meta_path = p
        p = meta_path.parent
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta, p


def resolve_tensor_path(artifact: Dict[str, Any], base_dir: Path) -> Path:
    """Resolve the tensor file path saved in metadata to a local file path.

    If metadata 'file' is a remote-style path (e.g., 'activations/run-.../.../file.pt'),
    prefer the file with the same basename in base_dir.
    """
    f = artifact.get("file")
    if not f:
        raise ValueError("Artifact is missing 'file' field")
    cand = base_dir / Path(f).name
    if cand.exists():
        return cand
    # Fallback: try relative to CWD
    p = Path(f)
    if p.exists():
        return p
    # As a last resort, try interpreting relative to base_dir's parent
    rel = base_dir.parent / f
    return rel


def main():
    ap = argparse.ArgumentParser(description="Plot NNsight activations with plotext")
    ap.add_argument("--path", required=True, help="Folder containing metadata.json or path to metadata.json")
    ap.add_argument("--list", action="store_true", help="List available artifact keys and exit")
    ap.add_argument("--key", default=None, help="Artifact key to load (e.g., layer_12_input_layernorm_output)")
    ap.add_argument("--mode", default="mean_hidden", choices=["mean_hidden", "feature", "hist"], help="Plot mode")
    ap.add_argument("--feature-index", type=int, default=0, help="Hidden feature index for mode=feature")
    ap.add_argument("--token-index", type=int, default=-1, help="Token index for mode=hist (default last token)")
    ap.add_argument("--width", type=int, default=100, help="Plot width")
    ap.add_argument("--height", type=int, default=30, help="Plot height")
    args = ap.parse_args()

    # Lazy import plotext to allow running without it installed
    try:
        import plotext as plt
    except Exception as e:
        raise SystemExit("plotext is required: pip install plotext") from e

    meta, base_dir = load_metadata(args.path)
    arts = meta.get("artifacts", {})
    if not arts:
        raise SystemExit("No artifacts found in metadata.json")

    if args.list:
        print("Available artifacts:")
        for k, v in arts.items():
            print(f" - {k}: shape={v.get('shape')} saved_shape={v.get('saved_shape')} file={Path(v.get('file','')).name}")
        return

    key = args.key or next(iter(arts.keys()))
    if key not in arts:
        raise SystemExit(f"Artifact key not found: {key}\nUse --list to see available keys.")

    tpath = resolve_tensor_path(arts[key], base_dir)
    if not tpath.exists():
        raise SystemExit(f"Tensor file not found: {tpath}")
    t = torch.load(tpath)
    if not isinstance(t, torch.Tensor):
        raise SystemExit(f"Loaded object is not a tensor: {type(t)}")
    if t.dim() < 3:
        raise SystemExit(f"Expected tensor of shape [B, L, H], got {list(t.shape)}")

    # Assume batch-first
    b, L, H = t.shape[0], t.shape[1], t.shape[2]
    if b != 1:
        print(f"Warning: batch size is {b}; plotting only the first element")
    x = list(range(L))

    plt.clear_figure()
    plt.figure(figsize=(args.width, args.height))
    plt.title(f"{key} | shape {list(t.shape)}")
    plt.xlabel("token index")

    if args.mode == "mean_hidden":
        y = t[0].mean(dim=-1).detach().cpu().tolist()
        plt.ylabel("mean over hidden")
        plt.plot(x, y)
    elif args.mode == "feature":
        idx = max(0, min(H - 1, args.feature_index))
        y = t[0, :, idx].detach().cpu().tolist()
        plt.ylabel(f"feature {idx}")
        plt.plot(x, y)
    elif args.mode == "hist":
        ti = args.token_index if args.token_index >= 0 else (L - 1)
        ti = max(0, min(L - 1, ti))
        vals = t[0, ti, :].detach().cpu().tolist()
        plt.clt(); plt.cld()
        plt.title(f"{key} | token {ti} histogram (H={H})")
        plt.hist(vals, bins=30)
        plt.show()
        return

    plt.show()


if __name__ == "__main__":
    main()

