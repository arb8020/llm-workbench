# tokenizer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import json, re, unicodedata, pathlib

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
    pat: re.Pattern = field(default=re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""), init=False)

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
    Load tokenizer from directory containing:
      - vocab.json (maps token -> id) OR encoder.json
      - merges.txt (merge list) OR vocab.bpe  
    """
    gpt2_assets_dir = pathlib.Path(gpt2_assets_dir)
    
    # Find encoder file
    encoder_file = None
    for name in ["vocab.json", "encoder.json"]:
        if (gpt2_assets_dir / name).exists():
            encoder_file = gpt2_assets_dir / name
            break
    if not encoder_file:
        raise FileNotFoundError(f"No encoder file found in {gpt2_assets_dir}")
        
    # Find merges file  
    merges_file = None
    for name in ["merges.txt", "vocab.bpe"]:
        if (gpt2_assets_dir / name).exists():
            merges_file = gpt2_assets_dir / name
            break
    if not merges_file:
        raise FileNotFoundError(f"No merges file found in {gpt2_assets_dir}")
    
    with open(encoder_file, "r", encoding="utf-8") as f:
        encoder = json.load(f)
    with open(merges_file, "r", encoding="utf-8") as f:
        merges = f.read().splitlines()
    merges = merges[1:] if merges and merges[0].startswith("#") else merges  # skip header if present
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
# Helper to download from HuggingFace
# -----------------------------

def download_gpt2_assets(model_name: str = "gpt2", cache_dir: str | pathlib.Path = None) -> pathlib.Path:
    """
    Download GPT-2 tokenizer files using HuggingFace Hub.
    Returns path to directory containing vocab.json and merges.txt.
    """
    from huggingface_hub import hf_hub_download
    import shutil
    import os
    
    # Map model names to actual repo IDs
    repo_mapping = {
        "gpt2": "openai-community/gpt2",
        "gpt2-medium": "openai-community/gpt2-medium", 
        "gpt2-large": "openai-community/gpt2-large",
        "gpt2-xl": "openai-community/gpt2-xl"
    }
    repo_id = repo_mapping.get(model_name, model_name)
    
    # Create a proper cache directory
    if cache_dir is None:
        cache_dir = pathlib.Path.home() / ".cache" / "inference" / "tokenizer_assets" / model_name
    else:
        cache_dir = pathlib.Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files to cache directory if they don't exist
    for filename in ["vocab.json", "merges.txt"]:
        local_path = cache_dir / filename
        if not local_path.exists():
            file_path = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copy2(file_path, local_path)
    
    return cache_dir

# -----------------------------
# Tokenizer smoke test
# -----------------------------

def _roundtrip_test(tok: GPT2Tokenizer) -> None:
    s = "Hello, world! 123 🍕"
    ids = encode(tok, s)
    back = decode(tok, ids)
    assert back == s, f"roundtrip failed:\n{back!r}\n!=\n{s!r}"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=str, help="Path with encoder.json and vocab.bpe (auto-downloads if not provided)")
    ap.add_argument("--download", type=str, default="gpt2", help="HF model name to download assets from (default: gpt2)")
    args = ap.parse_args()

    # Get tokenizer assets
    assets_dir = pathlib.Path(args.assets) if args.assets else download_gpt2_assets(args.download)
    if not args.assets:
        print(f"Assets downloaded to: {assets_dir}")

    tok = load_tokenizer_files(assets_dir)
    _roundtrip_test(tok)
    print("Tokenizer roundtrip ✅")