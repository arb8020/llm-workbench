# Llama3 JAX Implementation TODO

## Critical Bugs to Fix

### 1. **RMS Normalization Bug** (Line 547)
```python
# ❌ Current - undefined variable
normalized = weight * (inputs * rms)

# ✅ Fix - use correct variable name  
normalized = weight * (x_BLD * rms)
```

### 2. **Rotary Embedding Bugs** 
**Lines 465-466:** Undefined variable `rot_LK2_C`
```python
# ❌ Current
q_rot_c_BLH_K2 = q_c_BLH_K2 * rot_LK2_C[None, :, None, :]

# ✅ Fix  
q_rot_c_BLH_K2 = q_c_BLH_K2 * rot_LK2_complex[None, :, None, :]
```

**Lines 473-474:** Undefined variables in reshape
```python
# ❌ Current
q_out_BLHK = q_ri_BLH_K2_2.reshape(B, L, H, K).astype(dtype)

# ✅ Fix
q_out_BLHK = q_realimag_concat_BLH_K2_2.reshape(B, L, H, K).astype(dtype)
```

**Line 476:** Undefined return variables
```python
# ❌ Current
return xq_out, xk_out

# ✅ Fix
return q_out_BLHK, k_out_BLGK
```

### 3. **SwiGLU Function** (Lines 400-402)
```python
# ❌ Current - missing operands and return
back_down_BLD = jnp.einsum('blh,dh->bld')  # Missing operands
return  # Returns nothing

# ✅ Fix
back_down_BLD = jnp.einsum('blh,dh->bld', gated_output_BLH, down_proj_DH)
return back_down_BLD
```

## Missing Implementations

### 4. **Complete Grouped Query Attention** (Line 488+)
Currently just has `pass` - needs full implementation:
- Apply rotary embeddings to Q/K
- Update KV cache
- Compute attention scores
- Apply softmax
- Weight values by attention scores
- Output projection

### 5. **Complete Llama3 Block** (Line 556+)
Missing:
- Pre-attention RMS norm
- Attention computation
- Residual connection
- Pre-FFN RMS norm  
- Feed-forward computation
- Final residual connection

### 6. **Complete Forward Pass** (Line 567+)
Currently returns zeros - needs:
- Token embedding
- Loop through all transformer blocks
- Final RMS norm
- LM head projection

### 7. **Missing Helper Functions**
- `scale_frequencies()` - Referenced in `apply_ntk_scaling()` but not defined
- Attention masking function for causal generation
- Proper weight extraction and organization

## Configuration Fixes

### 8. **Config Parameter Consistency**
```python
# ❌ Current in grouped_query_attn()
H = config.num_heads  # Undefined attribute

# ✅ Fix  
H = config.n_heads
```

## Reference Implementation

Use the Entropix single-file implementation (`/tmp/entropix-single-file/llama3_forward_pass.py`) as reference for:
- Complete attention mechanism
- Proper KV cache integration
- Feed-forward network structure
- Transformer block organization
- Forward pass flow

## Priority Order

1. **Fix critical bugs** (items 1-3) - prevents compilation
2. **Implement core functions** (items 4-6) - enables basic functionality  
3. **Add missing helpers** (item 7) - completes implementation
4. **Test against HuggingFace** - verify correctness

## Testing

After implementing, verify:
- Code compiles without errors
- Forward pass produces correct output shapes
- Logits match HuggingFace Llama3 implementation