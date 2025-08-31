#!/usr/bin/env python3
"""
JAX GPT-2 implementation that starts as dummy and can be gradually filled out.

This script begins with dummy outputs and can be incrementally improved until
it matches HuggingFace GPT-2 logits exactly.

Usage:
    python engine/scripts/hello_gpt2_jax.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from engine.utils.comparison import compare_logits, get_hf_logits
except ImportError:
    print("Warning: Could not import comparison utilities")
    def compare_logits(*args, **kwargs):
        return {"message": "comparison not available"}
    def get_hf_logits(*args, **kwargs):
        return np.zeros((1, 5, 50257))  # dummy


class DummyGPT2:
    """Dummy GPT-2 implementation that starts simple and can be filled out."""
    
    def __init__(self, vocab_size: int = 50257, d_model: int = 768):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # TODO: Replace with real parameters loaded from HuggingFace
        self.dummy_weights = self._init_dummy_weights()
    
    def _init_dummy_weights(self) -> Dict[str, jnp.ndarray]:
        """Initialize dummy weights. Replace with real GPT-2 weights later."""
        key = jax.random.PRNGKey(42)
        
        return {
            # TODO: Add real GPT-2 architecture weights
            "dummy_embedding": jax.random.normal(key, (self.vocab_size, self.d_model)) * 0.01,
            "dummy_output": jax.random.normal(key, (self.d_model, self.vocab_size)) * 0.01,
        }
    
    def forward(self, input_ids_BL: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through dummy GPT-2.
        
        Args:
            input_ids_BL: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            logits_BLV: Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids_BL.shape
        
        # PHASE 1: Return completely dummy logits
        return self._phase1_dummy_logits(batch_size, seq_len)
        
        # TODO: Uncomment phases as you implement them
        # return self._phase2_embedding_lookup(input_ids_BL)
        # return self._phase3_add_positional(input_ids_BL)  
        # return self._phase4_add_attention(input_ids_BL)
        # return self._phase5_add_mlp(input_ids_BL)
        # return self._phase6_full_gpt2(input_ids_BL)
    
    def _phase1_dummy_logits(self, batch_size: int, seq_len: int) -> jnp.ndarray:
        """Phase 1: Return completely random logits."""
        print("🎲 Phase 1: Using dummy random logits")
        key = jax.random.PRNGKey(123)
        return jax.random.normal(key, (batch_size, seq_len, self.vocab_size)) * 0.1
    
    def _phase2_embedding_lookup(self, input_ids_BL: jnp.ndarray) -> jnp.ndarray:
        """Phase 2: Implement token embedding lookup."""
        print("📝 Phase 2: Token embedding lookup")
        # TODO: Implement embedding lookup
        # embeddings_BLD = self.weights['wte'][input_ids_BL]  
        # return jnp.matmul(embeddings_BLD, self.weights['lm_head'])
        raise NotImplementedError("Phase 2 not implemented yet")
    
    def _phase3_add_positional(self, input_ids_BL: jnp.ndarray) -> jnp.ndarray:
        """Phase 3: Add positional embeddings.""" 
        print("📍 Phase 3: Adding positional embeddings")
        # TODO: Add positional embeddings
        raise NotImplementedError("Phase 3 not implemented yet")
    
    def _phase4_add_attention(self, input_ids_BL: jnp.ndarray) -> jnp.ndarray:
        """Phase 4: Add attention mechanism."""
        print("🧠 Phase 4: Adding attention layers")
        # TODO: Implement attention blocks
        raise NotImplementedError("Phase 4 not implemented yet")
    
    def _phase5_add_mlp(self, input_ids_BL: jnp.ndarray) -> jnp.ndarray:
        """Phase 5: Add MLP layers."""
        print("🔧 Phase 5: Adding MLP layers") 
        # TODO: Implement MLP blocks
        raise NotImplementedError("Phase 5 not implemented yet")
    
    def _phase6_full_gpt2(self, input_ids_BL: jnp.ndarray) -> jnp.ndarray:
        """Phase 6: Full GPT-2 implementation."""
        print("🚀 Phase 6: Full GPT-2 implementation")
        # TODO: Complete GPT-2 implementation
        raise NotImplementedError("Phase 6 not implemented yet")


def test_gpt2_comparison():
    """Test our JAX GPT-2 against HuggingFace implementation."""
    print("🧪 Testing JAX GPT-2 vs HuggingFace GPT-2")
    print("=" * 50)
    
    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    # Use GPU if available, otherwise CPU
    device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
    print(f"Using device: {device}")
    
    # Test input: "Hello world"
    test_input = np.array([[15496, 995]])  # "Hello world" tokens for GPT-2
    print(f"Test input shape: {test_input.shape}")
    print(f"Test tokens: {test_input.tolist()}")
    
    with jax.default_device(device):
        # Get HuggingFace reference logits
        print("\n📚 Getting HuggingFace reference logits...")
        try:
            hf_logits = get_hf_logits(test_input, model_name="gpt2")
            print(f"HF logits shape: {hf_logits.shape}")
            print(f"HF logits range: [{hf_logits.min():.3f}, {hf_logits.max():.3f}]")
        except Exception as e:
            print(f"Failed to get HF logits: {e}")
            hf_logits = np.random.randn(1, 2, 50257) * 0.1  # dummy fallback
        
        # Get our JAX model logits  
        print("\n🔥 Getting JAX model logits...")
        model = DummyGPT2()
        jax_input = jnp.array(test_input)
        jax_logits = model.forward(jax_input)
        jax_logits_np = np.array(jax_logits)
        
        print(f"JAX logits shape: {jax_logits_np.shape}")
        print(f"JAX logits range: [{jax_logits_np.min():.3f}, {jax_logits_np.max():.3f}]")
        
        # Compare the two
        print("\n📊 Comparing logits...")
        comparison = compare_logits(
            jax_logits_np, 
            hf_logits,
            rtol=1e-3,
            atol=1e-5,
            verbose=True
        )
        
        # Next steps guidance
        print("\n" + "=" * 50)
        if comparison.get('all_close', False):
            print("🎉 SUCCESS! JAX model matches HuggingFace!")
        else:
            print("📋 Next steps to improve accuracy:")
            print("1. Uncomment _phase2_embedding_lookup in forward()")
            print("2. Load real GPT-2 weights from HuggingFace")
            print("3. Implement token embedding lookup")  
            print("4. Add positional embeddings")
            print("5. Implement attention mechanism")
            print("6. Add MLP layers and layer norms")
            
            max_diff = comparison.get('max_abs_diff', float('inf'))
            if max_diff > 10:
                print("💡 Large difference suggests missing core components")
            elif max_diff > 1:
                print("💡 Medium difference suggests architecture mismatch") 
            else:
                print("💡 Small difference suggests numerical precision issues")


if __name__ == "__main__":
    print("🚀 JAX GPT-2 Implementation - Phase by Phase")
    print("Starting with dummy implementation...")
    print()
    
    try:
        test_gpt2_comparison()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Script completed!")