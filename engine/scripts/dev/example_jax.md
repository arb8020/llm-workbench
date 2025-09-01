# JAX LLM Implementation Examples and Resources

## 🏆 Top JAX LLM Codebases (2025)

### 1. MaxText - Google's Flagship JAX LLM Framework
**Link**: https://github.com/AI-Hypercomputer/maxtext
- Pure Python/JAX, scales from single host to massive clusters
- Supports Llama 2/3/4, Gemma, Mistral, Qwen, DeepSeek models
- High performance with MFU optimization
- Recent 2025 updates with multi-token prediction

### 2. EasyLM - Comprehensive LLM Framework
**Link**: https://github.com/young-geng/EasyLM
- One-stop solution for training, finetuning, evaluation, and serving
- Built on Hugging Face transformers/datasets
- Multi-TPU/GPU support with pjit
- Clean, customizable codebase

### 3. Flax NNX (New 2024-2025) - Google's Updated Flax API
**Link**: https://github.com/google/flax
- Third-generation Flax API replacing Linen
- First-class Python reference semantics
- Easier debugging and model inspection

## 🔧 Specific Model Implementations

- **llama-2-jax**: https://github.com/ayaka14732/llama-2-jax - Clean Llama 2 implementation
- **starcoder-jax**: https://github.com/affjljoo3581/starcoder-jax - StarCoder/GPTBigCode models
- **awesome-jax-flax-llms**: https://github.com/dhyaneesh/awesome-jax-flax-llms - Curated collection

## 📚 Learning Resources

### Google's Scaling Book (2025)
**Link**: https://jax-ml.github.io/scaling-book/inference/

Key architectural patterns from the Scaling Book:

```python
from jax import Array
from typing import NamedTuple

# Modular Engine Interface
class Engine:
    def prefill(self, input_tokens):
        """Generate a KV cache from input tokens"""
        pass
    
    def insert(self, kv_cache):
        """Insert a KV cache into generation batch"""
        pass
    
    def generate(self, batched_kv_caches):
        """Generate one token per batch entry"""
        pass

# Strong typing with NamedTuple for state
class TransformerState(NamedTuple):
    kv_cache: Array  # Key-Value cache
    logits: Array    # Token probabilities
```

**Key Principles**:
- Continuous batching
- Prefix caching 
- Speculative sampling
- Modular inference engine design
- Functional decomposition of inference into discrete, composable steps

### NVIDIA Triton JAX Examples
**Link**: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/examples/jax/README.html
**GitHub**: https://github.com/triton-inference-server/python_backend/tree/r22.10/examples/jax

Complete JAX model implementation example:

```python
import json
import numpy as np
import jax.numpy as jnp
import triton_python_backend_utils as pb_utils
from typing import List, Any

def AddSub(input_0: jnp.ndarray, input_1: jnp.ndarray) -> List[jnp.ndarray]:
    """
    Perform addition and subtraction operations using JAX.
    
    Args:
        input_0 (jnp.ndarray): First input array
        input_1 (jnp.ndarray): Second input array
    
    Returns:
        List[jnp.ndarray]: List containing sum and difference of inputs
    """
    output_0 = jnp.add(input_0, input_1)
    output_1 = jnp.subtract(input_0, input_1)
    return [output_0, output_1]

class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        """Initialize model configuration and output data types."""
        self.model_config = json.loads(args['model_config'])
        
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT0")
        output1_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT1")
        
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

    def execute(self, requests: List[Any]) -> List[Any]:
        """Process inference requests using JAX operations."""
        responses = []
        
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            
            # Convert to JAX arrays and process
            jax_input_0 = jnp.array(in_0.as_numpy())
            jax_input_1 = jnp.array(in_1.as_numpy())
            
            # Pure JAX computation
            outputs = AddSub(jax_input_0, jax_input_1)
            
            # Convert back to Triton tensors
            out_0 = pb_utils.Tensor("OUTPUT0", outputs[0].astype(self.output0_dtype))
            out_1 = pb_utils.Tensor("OUTPUT1", outputs[1].astype(self.output1_dtype))
            
            responses.append(pb_utils.InferenceResponse([out_0, out_1]))
        
        return responses
```

## 🎯 Recommended Patterns for GPT2 Implementation

Based on the examples above, here are the key patterns to follow:

### 1. Type-First Functional Design

```python
from jax import Array
from typing import NamedTuple, Dict, Tuple
import jax.numpy as jnp

class GPT2Config(NamedTuple):
    """Configuration for GPT-2 model."""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12

class GPT2State(NamedTuple):
    """Immutable state container for inference"""
    tokens: Array
    kv_cache: Array
    position: int
```

### 2. Pure Functions with Clear Types

```python
def gpt2_prefill(weights: Dict[str, Array], tokens: Array, 
                config: GPT2Config) -> GPT2State:
    """Pure prefill function - generate initial KV cache"""
    # Implementation here
    pass

def gpt2_generate_token(weights: Dict[str, Array], state: GPT2State,
                       config: GPT2Config) -> Tuple[Array, GPT2State]:
    """Pure generation step - return logits and updated state"""
    # Implementation here
    pass

def gpt2_forward(weights: Dict[str, Array], input_ids: Array, 
                config: GPT2Config) -> Array:
    """Main forward pass function"""
    # Implementation here
    pass
```

### 3. Modular Component Functions

```python
def embedding_lookup(weights: Dict[str, Array], tokens: Array) -> Array:
    """Token embedding lookup"""
    return weights['wte'][tokens]

def positional_encoding(weights: Dict[str, Array], seq_len: int) -> Array:
    """Positional embedding"""
    return weights['wpe'][:seq_len]

def layer_norm(x: Array, weights: Dict[str, Array], layer_name: str) -> Array:
    """Layer normalization"""
    gamma = weights[f'{layer_name}.weight']
    beta = weights[f'{layer_name}.bias']
    return jnp.layer_norm(x, gamma=gamma, beta=beta)

def attention(x: Array, weights: Dict[str, Array], layer_idx: int) -> Array:
    """Multi-head attention"""
    # Implementation here
    pass

def mlp(x: Array, weights: Dict[str, Array], layer_idx: int) -> Array:
    """MLP block"""
    # Implementation here
    pass
```

## 🚀 Architecture Principles

1. **Separation of Concerns**: Separate prefill and generation functions
2. **Pure Functions**: No side effects, explicit state passing
3. **Strong Typing**: Use NamedTuple for configuration and state
4. **Composability**: Small, focused functions that can be composed
5. **JAX Transformations**: Design for `jit`, `vmap`, `pmap`
6. **Immutable State**: Use functional state transformations

## 📖 Additional Resources

- **JAX I/O 2025 workshops**: Building transformers with Flax NNX
- **Mesh Transformer JAX**: https://github.com/kingoflolz/mesh-transformer-jax
- **JAX Documentation**: https://jax.readthedocs.io/
- **Flax Documentation**: https://flax.readthedocs.io/