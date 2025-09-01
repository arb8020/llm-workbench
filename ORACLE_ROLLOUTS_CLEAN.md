# Oracle Analysis via Clean Rollouts System

Based on the detailed analysis, here are specific fixes for the GPT-2 precision issues:

1. **Fix Attention Bias Implementation**
```python
def attention(q: Array, k: Array, v: Array, attention_bias: Array, mask: Array) -> Array:
    # Add missing attention bias
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(q.shape[-1])
    # Important: Add attention bias before mask
    scores = scores + attention_bias
    # Use exact HF masking value
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1) * -10000.0
    scores = scores + mask[None, None, :, :]
    return jnp.matmul(jax.nn.softmax(scores, axis=-1), v)
```

2. **Match HuggingFace GELU Implementation**
```python
def gelu_new(x: Array) -> Array:
    """Matches HuggingFace's exact GELU implementation"""
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
```

3. **Fix Layer Normalization Precision**
```python
def layer_norm(x: Array, weights: Dict[str, Array], layer_name: str, epsilon: float = 1e-5) -> Array:
    gamma = weights[f"{layer_name}.weight"]
    beta = weights[f"{layer_name}.bias"]
    
    # Use float32 for intermediate calculations
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # Use unbiased variance matching PyTorch
    var = jnp.var(x, axis=-1, keepdims=True, ddof=1)
    normalized = (x - mean) / jnp.sqrt(var + epsilon)
    return (gamma * normalized + beta).astype(x.dtype)
```

4. **Correct Matrix Multiplication Order**
```python
def linear_transform(x: Array, weights: Dict[str, Array], prefix: str) -> Array:
    weight = weights[f"{prefix}.weight"]
    bias = weights[f"{prefix}.bias"]
    
    # Ensure correct matrix multiplication order and transposition
    x = x.astype(jnp.float32)
    weight = weight.astype(jnp.float32)
    
    # Match PyTorch's matmul convention
    output = jnp.matmul(x, weight.T)
    if bias is not None:
        output = output + bias
    
    return output.astype(x.dtype)
```

5. **Add Dtype Management**
```python
def gpt2_forward(weights: Dict[str, Array], input_ids: Array, config: GPT2Config) -> Array:
    # Ensure consistent dtype throughout
    dtype = jnp.float16 if config.torch_dtype == "float16" else jnp.float32
    
    def cast_weights(w):
        return {k: v.astype(dtype) for k, v in w.items()}
    
    weights = cast_weights(weights)
    # ... rest of forward pass ...
```

6. **Fix Weight Loading Precision**
```python
def convert_hf_weights_to_jax_format(hf_weights: Dict[str, Array]) -> Dict[str, Array]:
    converted = {}
    for name, param in hf_weights.items():
        # Ensure weights are loaded in correct precision
        if 'bias' in name or 'weight' in name:
            # Use float32 for weights and biases
            param = param.astype(jnp.float32)
        converted[name.replace('transformer.', '')] = param
    return converted
```

7. **Add Configuration Options**
```python
@dataclass
class PrecisionConfig:
    use_float32_attention: bool = True
    match_pytorch_gelu: bool = True
    unbiased_variance: bool = True
    epsilon: float = 1e-5
    mask_value: float = -10000.0
```

8. **Add Testing Utilities**
```python
def compare_outputs(jax_output: Array, torch_output: Array, rtol: float = 1e-5, atol: float = 1e-5) -> Dict:
    """Detailed output comparison"""
    return {
        'max_abs_diff': jnp.max(jnp.abs(jax_output - torch_output)),
        'mean_abs_diff': jnp.mean(jnp.abs(jax_output - torch_output)),
        'within_tolerance': jnp.allclose(jax_output, torch_output, rtol=rtol, atol=atol),
        'dtype_match': jax_output.dtype == torch_output.dtype
    }
```

These fixes should bring the implementation much closer to HuggingFace's precision. The key points are:
- Matching exact GELU implementation
- Using correct dtype management
- Adding missing attention bias
- Fixing layer normalization variance calculation
- Ensuring consistent matrix multiplication conventions
- Using proper masking values

After implementing these fixes, the max absolute difference should reduce significantly from 0.089 to near float32 precision levels (~1e-7).