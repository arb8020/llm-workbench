# Engine Architecture

## Overview

The engine is designed as a modular inference system supporting multiple backends with different frameworks and serving strategies. The architecture separates concerns between serving infrastructure, backend implementations, and framework-specific code.

## Directory Structure

```
engine/
├── backends/                     # Backend implementations for inference
│   ├── base.py                  # Abstract backend interface
│   ├── vllm/                    # vLLM backend (existing)
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── engine.py
│   ├── sglang/                  # SGLang backend (future)
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── engine.py
│   ├── transformers/            # HuggingFace Transformers backend
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── engine.py
│   └── custom/                  # Modular custom backends
│       ├── __init__.py
│       ├── base_custom.py       # Common interface for custom backends
│       ├── app.py               # FastAPI server (OpenAI compatible)
│       └── frameworks/          # ML framework implementations
│           ├── __init__.py
│           ├── jax/             # JAX implementation
│           │   ├── __init__.py
│           │   ├── models/      # Model implementations (GPT-2, Llama, etc.)
│           │   │   ├── __init__.py
│           │   │   ├── gpt2.py
│           │   │   └── base_model.py
│           │   ├── layers/      # Neural network layers
│           │   │   ├── __init__.py
│           │   │   ├── attention.py
│           │   │   ├── mlp.py
│           │   │   ├── layernorm.py
│           │   │   └── embeddings.py
│           │   ├── execution/   # Scheduling, batching, serving
│           │   │   ├── __init__.py
│           │   │   ├── scheduler.py
│           │   │   ├── batcher.py
│           │   │   └── cache.py
│           │   ├── kernels/     # Custom JAX kernels
│           │   │   ├── __init__.py
│           │   │   └── attention_kernels.py
│           │   └── engine.py    # JAX backend entry point
│           ├── torch/           # PyTorch implementation
│           │   ├── __init__.py
│           │   ├── models/      # Model implementations
│           │   │   ├── __init__.py
│           │   │   ├── gpt2.py
│           │   │   └── base_model.py
│           │   ├── layers/      # Neural network layers
│           │   │   ├── __init__.py
│           │   │   ├── attention.py
│           │   │   ├── mlp.py
│           │   │   ├── layernorm.py
│           │   │   └── embeddings.py
│           │   ├── execution/   # Scheduling, batching, serving
│           │   │   ├── __init__.py
│           │   │   ├── scheduler.py
│           │   │   ├── batcher.py
│           │   │   └── cache.py
│           │   ├── kernels/     # Custom PyTorch/Triton kernels
│           │   │   ├── __init__.py
│           │   │   ├── triton_kernels.py
│           │   │   └── cuda_kernels.py
│           │   └── engine.py    # PyTorch backend entry point
│           └── tinygrad/        # TinyGrad implementation (future)
│               ├── __init__.py
│               ├── models/
│               ├── layers/
│               ├── execution/
│               └── engine.py
├── scripts/                     # Development & testing workflows
│   ├── dev/                     # Development scripts
│   │   ├── __init__.py
│   │   ├── test_gpt2_equivalence.py    # HF vs our implementation
│   │   ├── single_file_gpt2.py         # Rapid prototyping
│   │   ├── benchmark_layers.py         # Layer-by-layer testing
│   │   └── profile_inference.py        # Performance profiling
│   ├── reference/               # Reference implementations (archive)
│   │   ├── __init__.py
│   │   └── monolithic_gpt2.py          # Original single-file version
│   └── deploy/                  # Deployment helpers
│       ├── __init__.py
│       ├── docker/              # Docker configurations
│       └── kubernetes/          # K8s configurations
├── core/                        # Shared infrastructure
│   ├── __init__.py
│   ├── interfaces/              # Protocol definitions
│   │   ├── __init__.py
│   │   ├── backend.py           # Backend protocol
│   │   ├── model.py             # Model protocol
│   │   └── server.py            # Server protocol
│   ├── utils/                   # Framework-agnostic utilities
│   │   ├── __init__.py
│   │   ├── tokenizer.py         # Tokenization utilities
│   │   ├── weights.py           # Weight loading/conversion
│   │   └── config.py            # Configuration management
│   └── registry/                # Backend factory/registry
│       ├── __init__.py
│       └── backend_registry.py  # Dynamic backend loading
├── servers/                     # Server implementations (existing)
│   ├── __init__.py
│   └── openai_compat.py
├── deployment/                  # Deployment utilities (existing)
│   ├── __init__.py
│   └── health.py
└── utils/                       # Engine utilities (existing)
    ├── __init__.py
    ├── tokenizer.py
    └── weights.py
```

## Architecture Principles

### 1. Backend Abstraction
- All backends implement the same abstract interface (`backends/base.py`)
- Backends can be swapped without changing client code
- Each backend manages its own lifecycle and configuration

### 2. Framework Modularity
- Custom backends support multiple frameworks under `frameworks/` (JAX, PyTorch, TinyGrad)
- Frameworks share similar structure: `models/`, `layers/`, `execution/`, `kernels/`
- Framework choice is implementation detail, not API concern

### 3. Server Architecture
- FastAPI server in `custom/app.py` acts as delegator to backends
- Server is framework-agnostic and handles routing, middleware, OpenAI compatibility
- Backends focus on inference, server handles HTTP/API concerns

### 4. Development Workflow
- `scripts/dev/` for rapid prototyping and testing
- `scripts/reference/` for archiving working implementations
- Layer-by-layer development and validation against reference implementations

## Backend Types

### External Backends
- **vLLM**: Production-ready, high-throughput serving
- **SGLang**: Structured generation and function calling
- **Transformers**: Simple HuggingFace integration

### Custom Backends
- **JAX**: Research-focused, easy differentiation, XLA compilation
- **PyTorch**: Production-ready, extensive ecosystem, custom kernels
- **TinyGrad**: Minimalist, educational, fast iteration

## Future Considerations

### Multi-GPU Support
- Framework-specific distributed strategies in `execution/distributed/`
- Shared tensor parallel utilities in `core/utils/`

### Multi-Node Support  
- Node-aware scheduling in `execution/`
- Network communication abstractions in `core/`

### Performance
- Custom kernels in framework-specific `kernels/` directories
- Shared performance utilities in `core/utils/`
- Benchmarking and profiling in `scripts/dev/`

## Development Phases

### Phase 1: Foundation
1. Implement base FastAPI server in `custom/app.py`
2. Create development scripts for GPT-2 equivalence testing
3. Build JAX GPT-2 implementation with layer-by-layer validation

### Phase 2: Framework Expansion
1. Add PyTorch custom backend
2. Implement Transformers backend for baseline
3. Add custom kernels for performance

### Phase 3: Production Features
1. Multi-GPU support within frameworks
2. Advanced scheduling and batching
3. Multi-node coordination

### Phase 4: Optimization
1. Custom kernel development
2. Memory optimization
3. Serving optimization

## TODOs: Future Architecture Enhancements

### Integration & Orchestration
- [ ] Triton integration strategy - when/how to use as serving layer over backends
- [ ] Backend routing logic - model-to-backend mapping strategies
- [ ] Multi-backend request distribution and load balancing

### State & Session Management
- [ ] KV cache strategies per backend type (in-memory, Redis, etc.)
- [ ] Stateful vs stateless serving trade-offs
- [ ] Multi-turn conversation session handling

### Operational Concerns
- [ ] Model loading & weight management - HuggingFace conversion utilities
- [ ] Shared weight storage and model registry patterns
- [ ] Observability - metrics, monitoring, request tracing across backends
- [ ] Configuration management - runtime updates, environment overrides

### Advanced Patterns
- [ ] JAX functional programming guidelines - pure functions, transformations
- [ ] Custom kernel development patterns per framework
- [ ] Performance profiling and optimization workflows

*Note: These can be fleshed out as implementation progresses and requirements become clearer.*