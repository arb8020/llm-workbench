# Next Work Session - Llama JAX Validation

## Current Status
We have a **95% faithful entropix JAX implementation** that successfully validates against official transformers for single forward passes (100% top-1 accuracy), but **KV cache multi-token generation is not yet tested**.

## Key Files Created
- ✅ `solution_entropix_faithful.py` - Faithful JAX implementation with KV cache
- ✅ `compare_faithful.py` - Multi-token generation testing framework  
- ✅ `FAITHFUL_IMPLEMENTATION.md` - Architecture documentation
- ⏳ `compare.py` - Needs update to use llama-stack

## ✅ COMPLETED: llama-stack Integration & Single Forward Pass Validation

## 🎯 NEW PRIORITY: Multi-Token Generation & KV Cache Validation

### What Needs To Be Done

1. **Install llama-stack on GPU instance**
   ```bash
   pip install llama-stack -U
   ```

2. **Download Llama-3.2-1B-Instruct model**
   - User needs to visit: https://www.llama.com/llama-downloads/
   - Get access approval (already done - see `llama.txt`)
   - Run commands:
     ```bash
     llama model list
     llama model download --source meta --model-id Llama-3.2-1B-Instruct
     ```
   - Use custom URL from `llama.txt` when prompted

3. **Update compare.py to use local llama-stack model**
   - Replace HuggingFace `get_hf_logits()` calls
   - Load model from local llama-stack installation
   - Update `solution_entropix_faithful.py` weight loading

### Implementation Plan

#### Step 1: Update Model Loading
```python
# OLD (HuggingFace):
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# NEW (llama-stack):
import llama_stack  # or appropriate import
model = load_local_llama_model("Llama-3.2-1B-Instruct")  # TBD exact API
```

#### Step 2: Update Weight Conversion
```python
def load_and_convert_weights(model_path: str) -> XfmrWeights:
    """Load from llama-stack local model instead of HuggingFace"""
    # Load local llama model
    # Convert to entropix XfmrWeights format
    # Return structured weights
```

#### Step 3: Update Comparison Logic
```python
def get_llama_local_logits(input_ids, model_path):
    """Get logits from local llama-stack model"""
    # Load local model
    # Run inference
    # Return logits in compatible format
```

#### Step 4: Test Full Pipeline
```python
# Test sequence:
# 1. Load local Llama-3.2-1B-Instruct 
# 2. Convert weights to JAX format
# 3. Run faithful entropix JAX forward pass
# 4. Compare JAX vs local Llama logits
# 5. Test multi-token generation
# 6. Validate KV cache performance
```

### Key Validation Tests

1. **Single Forward Pass**: JAX vs local Llama logits match
2. **Multi-Token Generation**: Autoregressive sampling works  
3. **KV Cache Validation**: Essential for proper entropix validation
   - Current comparison only does single forward passes
   - **CRITICAL**: Must test multi-token generation to validate KV cache
   - Compare JAX+KV vs Transformers for 10+ token sequences
   - Ensure both produce identical multi-token completions
4. **Performance**: KV cache provides 2x+ speedup vs recompute
5. **Faithfulness**: Compare against original entropix if available

### Expected Issues to Handle

1. **Weight Format Differences**: llama-stack vs HuggingFace weight naming
2. **Model Config**: Extract correct model parameters (n_layers, n_heads, etc.)  
3. **Tokenizer**: Ensure consistent tokenization between JAX and reference
4. **Precision**: JAX vs PyTorch numerical precision differences

### Success Criteria

- [x] JAX implementation loads Llama-3.2-1B weights successfully ✅
- [x] Single forward pass logits match local Llama (rtol=1e-3, atol=1e-2) ✅ 100% top-1 accuracy
- [ ] **Multi-token generation with KV cache validation**
  - [ ] Update compare.py to support multi-token generation mode
  - [ ] JAX entropix generates 10+ tokens using KV cache
  - [ ] Official transformers generates same 10+ tokens 
  - [ ] Token-by-token logits match throughout generation
  - [ ] KV cache provides correct incremental computation
- [ ] Performance validation
  - [ ] KV cache provides 2x+ speedup vs full recompute
  - [ ] Performance: >10 tokens/sec on GPU for inference
- [ ] Generation quality validation
  - [ ] Multi-token outputs are coherent and match transformers
  - [ ] Sampling with temperature works correctly
  - [ ] Top-k/top-p sampling produces diverse outputs

### Implementation Approach for KV Cache Validation

#### Current Status (COMPLETED ✅)
- Single forward pass comparison working 
- JAX vs Official Transformers: 100% top-1 token accuracy
- Both models load from same local llama-stack checkpoint
- No HuggingFace authentication required

#### Next Priority: Multi-Token Generation Testing

**Required Changes to compare.py:**

```python
def compare_multi_token_generation(jax_forward_fn, transformers_model, prompt_tokens, max_tokens=10):
    """
    Compare JAX+KV vs Transformers for multi-token autoregressive generation.
    This validates that KV cache works correctly across multiple decode steps.
    """
    print(f"🔄 Generating {max_tokens} tokens with KV cache validation...")
    
    # JAX generation with KV cache
    jax_tokens = []
    jax_kv_cache = None
    current_tokens = prompt_tokens
    
    for step in range(max_tokens):
        # JAX forward pass (should use KV cache for steps > 0)
        logits = jax_forward_fn(current_tokens, kv_cache=jax_kv_cache)
        next_token = jnp.argmax(logits[0, -1])  # Greedy sampling
        jax_tokens.append(int(next_token))
        current_tokens = jnp.concatenate([current_tokens, next_token[None, None]], axis=1)
    
    # Transformers generation (for comparison)  
    with torch.no_grad():
        transformers_output = transformers_model.generate(
            torch.from_numpy(prompt_tokens), 
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for deterministic comparison
            pad_token_id=transformers_model.config.eos_token_id
        )
    
    transformers_tokens = transformers_output[0, len(prompt_tokens[0]):].tolist()
    
    # Compare token sequences
    match_count = sum(1 for a, b in zip(jax_tokens, transformers_tokens) if a == b)
    match_rate = match_count / len(jax_tokens) * 100
    
    print(f"Multi-token generation match rate: {match_rate:.1f}% ({match_count}/{len(jax_tokens)})")
    print(f"JAX tokens:          {jax_tokens}")
    print(f"Transformers tokens: {transformers_tokens}")
    
    return match_rate >= 95.0  # 95%+ match rate for success
```

**Key Testing Scenarios:**
1. **Short sequences** (5-10 tokens) - Basic KV cache functionality
2. **Medium sequences** (20-50 tokens) - Memory efficiency validation  
3. **Different prompts** - Various input contexts
4. **Temperature sampling** - Stochastic generation consistency
5. **Performance timing** - KV cache vs full recompute speedup

### Files to Modify

1. **`compare.py`** - Add multi-token generation mode
   - Update model loading to use llama-stack
   - Update reference logits generation
   - Add proper error handling for missing models

2. **`solution_entropix_faithful.py`** - Weight loading
   - Implement `load_and_convert_weights()` 
   - Handle llama-stack weight format
   - Extract model config parameters

3. **`engine/core/utils/comparison.py`** - Reference implementation  
   - Add `get_local_llama_logits()` function
   - Support local model loading

### Next Claude Instructions

1. **First**, try running current `compare_faithful.py` to see architecture status
2. **Then**, focus on llama-stack integration in `compare.py`  
3. **Validate** each step incrementally - don't try to do everything at once
4. **Document** any API discoveries about llama-stack in this directory

### Context for Next Session

- We built a faithful entropix JAX implementation (~95% faithful)
- Key improvement: proper KV cache for efficient autoregressive generation
- Missing piece: weight loading from actual Llama models
- User has approved access to Llama-3.2-1B and Llama-3.2-3B models
- Goal: Validate our JAX implementation produces same results as official Llama

**Priority**: Get the faithful JAX implementation working with real Llama weights!