from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import uuid

import torch

from .config import InterventionConfig


def _tensor_to_save(t: torch.Tensor, sample_hidden: int | None) -> torch.Tensor:
    if sample_hidden is None:
        return t
    if t.dim() >= 1 and t.shape[-1] > sample_hidden:
        return t[..., :sample_hidden]
    return t


def _save_tensor(path: Path, t: torch.Tensor, fmt: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pt":
        torch.save(t.cpu(), path)
    elif fmt == "npy":
        import numpy as np
        np.save(str(path), t.detach().cpu().numpy())
        # np.save adds .npy if not present
        if not str(path).endswith(".npy"):
            path = Path(str(path) + ".npy")
    else:
        raise ValueError("Unsupported save format")
    return str(path)


def write_activation_set(
    run_dir: Path,
    request_id: str,
    tensors: Dict[str, torch.Tensor],
    cfg: InterventionConfig,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"request_id": request_id, "artifacts": {}}
    root = run_dir / request_id
    root.mkdir(parents=True, exist_ok=True)

    for key, t in tensors.items():
        try:
            t = t.detach()
        except Exception:
            pass
        t = t.to(torch.float32) if t.is_floating_point() else t
        t_s = _tensor_to_save(t, cfg.sample_hidden)
        fname_base = key.replace("/", "_").replace(".", "_")
        if cfg.save_format == "pt":
            fpath = root / f"{fname_base}.pt"
        else:
            fpath = root / f"{fname_base}.npy"
        saved_path = _save_tensor(fpath, t_s, cfg.save_format)
        out["artifacts"][key] = {
            "file": saved_path,
            "shape": list(t.shape),
            "saved_shape": list(t_s.shape),
            "dtype": str(t.dtype),
        }

    if cfg.include_metadata_json:
        meta_path = root / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(out, f, indent=2)
        out["index"] = str(meta_path)

    return out


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
