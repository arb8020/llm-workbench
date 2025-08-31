# Engine

A modular inference system supporting multiple backends and frameworks for LLM serving.

## Architecture

The engine is designed with pluggable backends that can be swapped without changing client code. It supports both external backends (vLLM, SGLang, Transformers) and custom backends with framework-specific implementations (JAX, PyTorch, TinyGrad).

See [architecture.md](./architecture.md) for detailed design documentation.

## Directory Structure

```
engine/
├── backends/                    # Backend implementations
│   ├── base.py                 # Abstract backend interface
│   ├── vllm/                   # vLLM backend (production-ready)
│   ├── transformers/           # HuggingFace Transformers backend
│   └── custom/                 # Custom modular backends
│       ├── app.py              # FastAPI server
│       └── frameworks/         # Framework implementations
│           ├── jax/            # JAX implementation
│           ├── torch/          # PyTorch implementation
│           └── tinygrad/       # TinyGrad implementation
├── scripts/                    # Development & testing
│   ├── dev/                    # Rapid prototyping scripts
│   ├── reference/              # Archived implementations
│   └── deploy/                 # Deployment helpers
├── core/                       # Shared infrastructure
│   ├── interfaces/             # Protocol definitions
│   ├── utils/                  # Framework-agnostic utilities
│   └── registry/               # Backend factory/registry
├── deployment/                 # Deployment utilities (legacy)
└── servers/                    # Server implementations (legacy)
```

## Quick Start

### Using vLLM Backend

```python
from engine.backends.vllm import VLLMBackend, VLLMConfig

# Configure and start vLLM backend
config = VLLMConfig(
    model_name="openai-community/gpt2",
    host="0.0.0.0",
    port=8000
)

backend = VLLMBackend(config)
endpoint_url = await backend.start()

print(f"vLLM server ready at {endpoint_url}")
```

### Using Custom Framework Backend

```python
from engine.backends.custom.frameworks.jax import JAXBackend

# JAX backend for custom implementations
backend = JAXBackend("openai-community/gpt2")
# Implementation in progress...
```

### Development Scripts

```bash
# Rapid prototyping
python engine/scripts/dev/single_file_gpt2.py

# Test equivalence with HuggingFace
python engine/scripts/dev/test_gpt2_equivalence.py

# Benchmark layer implementations
python engine/scripts/dev/benchmark_layers.py
```

## Backends

### External Backends

- **vLLM**: High-throughput production serving with optimized kernels
- **SGLang**: Structured generation and function calling (planned)
- **Transformers**: Simple HuggingFace integration for baseline comparisons

### Custom Framework Backends

- **JAX**: Research-focused, easy differentiation, XLA compilation
- **PyTorch**: Production-ready with extensive ecosystem and custom kernels
- **TinyGrad**: Minimalist framework for fast iteration and education

Each custom framework follows the same structure:
- `models/`: Model implementations (GPT-2, Llama, etc.)
- `layers/`: Neural network layers (attention, MLP, LayerNorm)
- `execution/`: Scheduling, batching, and serving logic  
- `kernels/`: Framework-specific optimized kernels

## Development Workflow

### Phase 1: Foundation (Current)
1. ✅ Restructured to modular architecture
2. 🔄 Building JAX GPT-2 implementation with layer-by-layer validation
3. 📋 Development scripts for equivalence testing

### Phase 2: Framework Expansion
1. Add PyTorch custom backend
2. Implement Transformers backend for baseline
3. Add custom kernels for performance

### Phase 3: Production Features
1. Multi-GPU support within frameworks
2. Advanced scheduling and batching
3. Multi-node coordination

## Examples

The `examples/` directory contains complete deployment workflows:

### Simple vLLM Deployment
```bash
python examples/deploy_inference_server/simple_vllm/deploy.py
```
- Provisions GPU on cloud provider
- Deploys and starts vLLM server
- Returns OpenAI-compatible endpoint

### Interpretability Server
```bash
python examples/deploy_inference_server/simple_vllm_nnsight/deploy.py
```
- vLLM server with nnsight integration
- Activation collection and intervention capabilities
- Extended API for interpretability research

## Configuration

### Backend Configuration

Each backend accepts a configuration object that extends `BackendConfig`:

```python
@dataclass
class BackendConfig:
    model_name: str = "openai-community/gpt2"
    host: str = "0.0.0.0"
    port: int = 8000
    backend_args: Dict[str, Any] = field(default_factory=dict)
```

### vLLM Configuration

```python
@dataclass
class VLLMConfig(BackendConfig):
    # vLLM specific options
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    # ... additional vLLM options
```

## API Compatibility

All backends expose OpenAI-compatible APIs:

- `/v1/models` - List available models
- `/v1/completions` - Text completion
- `/v1/chat/completions` - Chat completion  
- `/health` - Health check endpoint

Custom backends can extend with additional endpoints for framework-specific features.

## Contributing

### Adding a New Framework

1. Create directory: `backends/custom/frameworks/{framework}/`
2. Implement required structure: `models/`, `layers/`, `execution/`, `kernels/`
3. Create framework-specific engine in `engine.py`
4. Add tests in `scripts/dev/`

### Adding a New External Backend

1. Create directory: `backends/{backend}/`
2. Implement `BackendConfig` and `InferenceBackend` subclasses
3. Follow the interface defined in `backends/base.py`
4. Add configuration and startup logic

## Testing

```bash
# Test backend functionality
python -m pytest tests/

# Run development scripts
python engine/scripts/dev/test_gpt2_equivalence.py

# Benchmark performance
python engine/scripts/dev/benchmark_layers.py
```

## Deployment

The engine is designed for flexible deployment:

- **Local Development**: Direct Python execution
- **Remote GPU**: Using broker + bifrost for cloud deployment
- **Container**: Docker images with backend-specific dependencies
- **Kubernetes**: Scalable serving with multiple backend types

See `examples/deploy_inference_server/` for complete deployment workflows.

## TODO

### Integration Tasks

- [ ] **Integrate GSM8K examples with engine backends**: Currently GSM8K remote examples (`examples/gsm8k_remote/`) deploy vLLM directly via `vllm.entrypoints.openai.api_server` instead of using the engine's backend architecture. Should refactor to use `engine.backends.vllm` for consistency.

- [ ] **Add Transformers backend implementation**: Create `backends/transformers/` with HuggingFace integration for baseline comparisons.

- [ ] **Implement GPT-2 equivalence testing**: Create comprehensive testing in `scripts/dev/test_gpt2_equivalence.py` to validate custom implementations against HuggingFace.

- [ ] **Build custom framework layers**: Implement attention, MLP, and other layers in `backends/custom/frameworks/jax/layers/` and `backends/custom/frameworks/torch/layers/`.

- [ ] **Add FastAPI server**: Implement `backends/custom/app.py` that delegates to framework backends.

## License

MIT License - see LICENSE file for details.