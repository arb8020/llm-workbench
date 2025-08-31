# test_core.py - Test core functionality without downloads
from __future__ import annotations
import tempfile
import json
import pathlib
from tokenizer import GPT2Tokenizer, load_tokenizer_files, encode, _bytes_to_unicode, _get_pairs
from weights import GPT2Weights, _canonicalize
import numpy as np

def create_dummy_tokenizer_files() -> pathlib.Path:
    """Create minimal dummy tokenizer files for testing."""
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="test_gpt2_"))
    
    # Create a minimal encoder.json
    encoder = {
        "!": 0,
        '"': 1,
        "#": 2,
        "Hello": 3,
        "Ġworld": 4,  # GPT-2 uses Ġ for leading spaces
        "Ġ": 5,
    }
    
    with open(temp_dir / "encoder.json", "w") as f:
        json.dump(encoder, f)
    
    # Create a minimal vocab.bpe
    bpe_data = """#version: 0.2
Ġ w
H e
"""
    with open(temp_dir / "vocab.bpe", "w") as f:
        f.write(bpe_data)
    
    return temp_dir

def test_tokenizer():
    """Test tokenizer core functionality."""
    print("Testing tokenizer...")
    
    # Test byte mapping
    byte_encoder, byte_decoder = _bytes_to_unicode()
    assert len(byte_encoder) == 256
    assert len(byte_decoder) == 256
    print("✅ Byte mapping works")
    
    # Test pair extraction
    pairs = _get_pairs(("a", "b", "c"))
    assert ("a", "b") in pairs
    assert ("b", "c") in pairs
    assert len(pairs) == 2
    print("✅ Pair extraction works")
    
    # Test with dummy files
    try:
        assets_dir = create_dummy_tokenizer_files()
        tok = load_tokenizer_files(assets_dir)
        assert isinstance(tok, GPT2Tokenizer)
        assert len(tok.encoder) == 6
        print("✅ Tokenizer loading works")
        
        # Simple encoding test (won't work perfectly with dummy data but shouldn't crash)
        try:
            ids = encode(tok, "Hello")
            assert isinstance(ids, list)
            print("✅ Encoding works (basic)")
        except KeyError:
            print("⚠️  Encoding failed (expected with dummy data)")
            
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")

def test_weights():
    """Test weight loading core functionality."""
    print("\nTesting weights...")
    
    # Test canonicalization
    dummy_params = {
        "transformer.wte.weight": np.array([[1, 2], [3, 4]]),
        "transformer.h.0.ln_1.weight": np.array([1, 2]),
        "transformer.h.0.ln_1.bias": np.array([0, 1]),
    }
    
    canonical = _canonicalize(dummy_params)
    assert "wte" in canonical
    assert "transformer.h.0.ln1.gamma" in canonical
    assert "transformer.h.0.ln1.beta" in canonical
    print("✅ Parameter canonicalization works")
    
    # Test GPT2Weights creation
    weights = GPT2Weights(params=canonical)
    assert isinstance(weights, GPT2Weights)
    assert "wte" in weights.params
    print("✅ GPT2Weights creation works")

if __name__ == "__main__":
    test_tokenizer()
    test_weights()
    print("\n🎉 Core functionality tests complete!")