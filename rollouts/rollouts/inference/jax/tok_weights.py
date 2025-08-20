# tok_and_weights.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import json
import re
import unicodedata
import pathlib
import numpy as np

# -----------------------------
# GPT-2 Byte-Level BPE Tokenizer
# -----------------------------

@dataclass(frozen=True)
class GPT2Tokenizer:
    """Immutable struct of GPT-2 tokenizer state."""
    encoder: Dict[str, int]                       # token -> id
    decoder: Dict[int, str]                       # id -> token
    bpe_ranks: Dict[Tuple[str, str], int]         # pair -> rank
    byte_encoder: Dict[int, str]                  # raw byte -> unicode char
    byte_decoder: Dict[str, int]                  # unicode char -> raw byte
    pat: re.Pattern = field(default=re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.IGNORECASE), init=False)

def _bytes_to_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    # GPT-2's reversible byte<->unicode mapping
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    return byte_encoder, byte_decoder

def _get_pairs(word: Tuple[str, ...]) -> set[Tuple[str, str]]:
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs

def load_tokenizer_files(gpt2_assets_dir: str | pathlib.Path) -> GPT2Tokenizer:
    """
    Expects two files in gpt2_assets_dir:
      - encoder.json  (maps token -> id)
      - vocab.bpe     (merge list starting after the first header line)
    You can grab them from any HF GPT-2 repo (e.g., openai-community/gpt2).
    """
    gpt2_assets_dir = pathlib.Path(gpt2_assets_dir)
    with open(gpt2_assets_dir / "encoder.json", "r", encoding="utf-8") as f:
        encoder = json.load(f)
    with open(gpt2_assets_dir / "vocab.bpe", "r", encoding="utf-8") as f:
        merges = f.read().splitlines()
    merges = merges[1:]  # skip header line
    merges = [tuple(m.split()) for m in merges if m and not m.startswith("#")]

    bpe_ranks = {pair: i for i, pair in enumerate(merges)}
    decoder = {v: k for k, v in encoder.items()}
    byte_encoder, byte_decoder = _bytes_to_unicode()
    return GPT2Tokenizer(
        encoder=encoder,
        decoder=decoder,
        bpe_ranks=bpe_ranks,
        byte_encoder=byte_encoder,
        byte_decoder=byte_decoder,
    )

def _bpe(tokenizer: GPT2Tokenizer, token: str, cache: Dict[str, str]) -> str:
    if token in cache:
        return cache[token]
    word = tuple(token)
    pairs = _get_pairs(word)
    if not pairs:
        cache[token] = token
        return token
    while True:
        ranked = {pair: tokenizer.bpe_ranks.get(pair, 10**10) for pair in pairs}
        bpe_pair = min(ranked, key=ranked.get)
        if bpe_pair not in tokenizer.bpe_ranks:
            break
        first, second = bpe_pair
        new_word: List[str] = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break
            if i < len(word)-1 and word[i] == first and word[i+1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = _get_pairs(word)
    out = " ".join(word)
    cache[token] = out
    return out

def encode(tokenizer: GPT2Tokenizer, text: str) -> List[int]:
    # Normalize to NFC like HF does for GPT-2
    text = unicodedata.normalize("NFC", text)
    # Byte-level: map raw bytes to unicode chars
    bpe_cache: Dict[str, str] = {}
    b2u = tokenizer.byte_encoder
    out: List[int] = []
    for token in tokenizer.pat.findall(text):
        token_bytes = token.encode("utf-8")
        token_trans = "".join(b2u[b] for b in token_bytes)
        for bpe_token in _bpe(tokenizer, token_trans, bpe_cache).split(" "):
            out.append(tokenizer.encoder[bpe_token])
    return out

def decode(tokenizer: GPT2Tokenizer, ids: List[int]) -> str:
    # Convert BPE tokens back to byte-level string, then to UTF-8
    u2b = tokenizer.byte_decoder
    text = "".join(tokenizer.decoder[i] for i in ids)
    # Each entry in decoder is in byte-level unicode space; map back to bytes
    byte_str = bytes([u2b[ch] for ch in text])
    return byte_str.decode("utf-8", errors="replace")

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
        from safetensors.numpy import load_file as st_load
    except Exception:
        return None
    for fname in ["model.safetensors", "pytorch_model.safetensors"]:
        path = model_dir / fname
        if path.exists():
            return {k: v for k, v in st_load(str(path)).items()}
    # Some repos shard: model-00001-of-000xx.safetensors
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        from safetensors.numpy import load_file as st_load
        out = {}
        for shard in shards:
            out.update({k: v for k, v in st_load(str(shard)).items()})
        return out
    return None

def _maybe_load_pytorch_bin(model_dir: pathlib.Path) -> Dict[str, np.ndarray] | None:
    try:
        import torch
    except Exception:
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
    Load GPT-2 weights from a Hugging Face-style directory (preferred),
    or from a single .npz export. Returns all tensors as numpy arrays.
    """
    model_dir = pathlib.Path(model_dir)
    params = (
        _maybe_load_safetensors(model_dir)
        or _maybe_load_pytorch_bin(model_dir)
        or _maybe_load_npz(model_dir)
    )
    if params is None:
        raise FileNotFoundError(
            f"Could not find weights in {model_dir}. "
            "Expected model.safetensors / pytorch_model.bin / *.npz"
        )
    return GPT2Weights(params=_canonicalize(params))

# -----------------------------
# Tiny smoke tests
# -----------------------------

def _roundtrip_test(tok: GPT2Tokenizer) -> None:
    s = "Hello, world! 123 🍕"
    ids = encode(tok, s)
    back = decode(tok, ids)
    assert back == s, f"roundtrip failed:\n{back!r}\n!=\n{s!r}"

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
    ap.add_argument("--assets", type=str, required=True, help="Path with encoder.json and vocab.bpe")
    ap.add_argument("--model",  type=str, required=True, help="Path to HF GPT-2 model dir or .npz")
    args = ap.parse_args()

    tok = load_tokenizer_files(args.assets)
    _roundtrip_test(tok)
    print("Tokenizer roundtrip ✅")

    w = load_gpt2_weights(args.model)
    hints = _shape_hints(w)
    for k, v in hints.items():
        print(f"{k}: {v}")
    print("Weights loaded ✅  (above are a few shape checks)")

