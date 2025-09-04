# Next Work Session - Llama JAX Validation

## Current Status
We have a **95% faithful entropix JAX implementation** ready for validation, but need to integrate with official Llama 3.2 models.

## Key Files Created
- ✅ `solution_entropix_faithful.py` - Faithful JAX implementation with KV cache
- ✅ `compare_faithful.py` - Multi-token generation testing framework  
- ✅ `FAITHFUL_IMPLEMENTATION.md` - Architecture documentation
- ⏳ `compare.py` - Needs update to use llama-stack

## PRIORITY: Update compare.py for llama-stack Integration

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
3. **KV Cache**: Performance improvement vs recompute
4. **Faithfulness**: Compare against original entropix if available

### Expected Issues to Handle

1. **Weight Format Differences**: llama-stack vs HuggingFace weight naming
2. **Model Config**: Extract correct model parameters (n_layers, n_heads, etc.)  
3. **Tokenizer**: Ensure consistent tokenization between JAX and reference
4. **Precision**: JAX vs PyTorch numerical precision differences

### Success Criteria

- [ ] JAX implementation loads Llama-3.2-1B weights successfully
- [ ] Single forward pass logits match local Llama (rtol=1e-3, atol=1e-2)
- [ ] Multi-token generation produces coherent outputs
- [ ] KV cache provides 2x+ speedup for generation
- [ ] Performance: >10 tokens/sec on GPU for inference

### Files to Modify

1. **`compare.py`** - Main integration work
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