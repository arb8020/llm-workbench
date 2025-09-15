from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import List, Optional, Dict, Any
import time


SUPPORTED_HOOK_POINTS = {
    # Residual stream pre-attn and pre-MLP for Qwen-style blocks
    "input_layernorm.output",
    "post_attention_layernorm.output",
}


@dataclass
class InterventionConfig:
    enabled: bool = False
    # Selection
    layers: List[int] = field(default_factory=list)
    hook_points: List[str] = field(default_factory=lambda: [
        "input_layernorm.output",
        "post_attention_layernorm.output",
    ])

    # Capture mode
    # - trace: single forward pass over prompt
    # - generate: capture while generating new tokens
    mode: str = "trace"  # one of {"trace", "generate"}

    # Storage
    save_dir: Path = Path("./activations")
    per_request_subdir: bool = True
    sample_hidden: Optional[int] = None  # if set, slice last dim to first N features
    save_format: str = "pt"  # one of {"pt", "npy"}

    # Extra flags
    include_metadata_json: bool = True

    def validate(self, n_layers: int | None = None) -> None:
        if self.mode not in {"trace", "generate"}:
            raise ValueError("mode must be 'trace' or 'generate'")
        for hp in self.hook_points:
            if hp not in SUPPORTED_HOOK_POINTS:
                raise ValueError(f"Unsupported hook point: {hp}")
        if self.sample_hidden is not None and self.sample_hidden <= 0:
            raise ValueError("sample_hidden must be positive when set")
        if self.save_format not in {"pt", "npy"}:
            raise ValueError("save_format must be 'pt' or 'npy'")
        if n_layers is not None:
            for i in self.layers:
                if i < 0 or i >= n_layers:
                    raise ValueError(f"layer index out of range: {i} (n_layers={n_layers})")

    def ensure_dirs(self) -> Path:
        base = Path(self.save_dir)
        if self.per_request_subdir:
            ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = base / f"run-{ts}"
        else:
            run_dir = base
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Coerce Path to str for JSON
        d["save_dir"] = str(self.save_dir)
        return d

    def clone(self) -> "InterventionConfig":
        return replace(self)
