# vLLM Recipes: Push-Button Model Serving

## Problem Statement

The current vLLM deployment experience is painful and error-prone. Users face:

1. **Manual GPU memory calculations** - trial-and-error with `--gpu-memory-utilization`, `--max-model-len`
2. **Authentication surprises** - discovering gated model access issues after expensive GPU provisioning
3. **Complex configuration** - dozens of CLI flags, tensor parallel sizing, port management
4. **Environment setup hell** - manual dependency installation, token management, environment variables
5. **Resource waste** - provisioning wrong GPU sizes, debugging on expensive instances

## Solution: Configuration-Driven Recipes

Transform the painful manual process into declarative YAML recipes with intelligent validation and automation.

### Architecture

```
broker (provision) → bifrost (deploy) → vllm (serve)
```

1. **Recipe defines requirements** (model, GPU specs, vLLM config)
2. **Broker provisions optimal GPU** (auto-sizing, cost optimization)
3. **Bifrost deploys and configures** (dependencies, environment, model serving)
4. **User gets working endpoint** (no debugging required)

## Implementation Plan

### Directory Structure
```
examples/vllm_recipes/
├── run_recipe.py                    # Single push-button script
├── recipes/
│   ├── llama_70b_chat.yaml         # Solves complaint.txt exactly
│   ├── llama_8b_dev.yaml           # Lightweight development
│   ├── code_llama_jupyter.yaml     # Interactive coding assistant
│   ├── multi_model_serving.yaml    # Multiple models on one instance
│   ├── openai_compatible.yaml      # Drop-in OpenAI API replacement
│   └── gpu_poor_options.yaml       # CPU fallback recipes
├── templates/
│   ├── base_vllm.yaml             # Common vLLM configurations
│   ├── chat_template.yaml         # Chat-optimized settings
│   └── completion_template.yaml   # Text completion settings
└── README.md                       # Usage guide and examples
```

### Core Components

#### 1. Recipe Schema
```yaml
# Recipe format
name: "Human-readable description"
model: "huggingface/model-name"
instance:
  min_vram: 80                     # Auto-select appropriate GPU
  max_price: 1.50                  # Cost guardrails
  gpu_types: ["RTX 4090", "A100"]  # Preference list
  jupyter: false                   # Optional Jupyter server
deployment:
  method: "bifrost"                # Always use bifrost for deployment
  pre_commands: []                 # Setup commands before vLLM
  vllm_args:                       # vLLM-specific configuration
    gpu_memory_utilization: 0.95
    tensor_parallel_size: 2
    max_model_len: 4096
    served_model_name: "custom-name"
    chat_template: "llama3"
validation:
  check_hf_token: true            # Validate model access before provisioning
  estimate_costs: true            # Show cost estimates
  warn_download_time: true        # Warn about large model downloads
ports:
  - 8000                          # vLLM API endpoint
environment:
  HF_TOKEN: "${HF_TOKEN}"         # Environment variables
  CUDA_VISIBLE_DEVICES: "0"
```

#### 2. Push-Button Runner Script
```python
#!/usr/bin/env python3
"""
Single script to run any vLLM recipe with intelligent validation
and automatic deployment via broker + bifrost.
"""

def run_recipe(recipe_file, overrides=None):
    """
    Main entry point for recipe execution
    
    1. Load and validate recipe
    2. Pre-flight checks (model access, cost estimation)
    3. Provision GPU via broker
    4. Deploy and configure via bifrost
    5. Health check and return endpoint
    """
    
def validate_recipe(recipe):
    """
    Pre-flight validation to prevent expensive failures:
    - HF token access for gated models
    - Model size vs GPU VRAM requirements
    - Cost estimation and user confirmation
    - Network/bandwidth requirements
    """
    
def build_vllm_command(recipe):
    """
    Generate optimal vLLM command from recipe configuration
    - Auto-configure tensor parallelism based on GPU count
    - Set memory utilization based on model size
    - Apply chat templates and serving optimizations
    """
```

#### 3. Example Recipes

**complaint.txt Solution (`llama_70b_chat.yaml`)**:
```yaml
name: "Llama 70B Chat Server - Complaint.txt Solution"
model: "meta-llama/Llama-3.3-70B-Instruct"
instance:
  min_vram: 80                     # Ensures adequate memory
  max_price: 1.50
  gpu_types: ["A100", "RTX 4090"]
deployment:
  method: "bifrost"
  vllm_args:
    gpu_memory_utilization: 0.95   # Optimal setting, no trial-and-error
    tensor_parallel_size: 2        # Auto-configured for 70B model
    max_model_len: 4096
    served_model_name: "llama-70b-chat"
    chat_template: "llama3"
validation:
  check_hf_token: true            # Prevents auth surprises
  estimate_costs: true
  warn_download_time: true
ports:
  - 8000
```

**Development Setup (`llama_8b_dev.yaml`)**:
```yaml
name: "Llama 8B Development Server"
model: "meta-llama/Llama-3.1-8B-Instruct"
instance:
  min_vram: 16                    # Cost-effective for development
  max_price: 0.50
deployment:
  method: "bifrost"
  vllm_args:
    gpu_memory_utilization: 0.90
    tensor_parallel_size: 1       # Single GPU sufficient
    max_model_len: 8192          # Longer context for development
validation:
  check_hf_token: true
```

**Jupyter Integration (`code_llama_jupyter.yaml`)**:
```yaml
name: "Code Llama with Jupyter Lab"
model: "codellama/CodeLlama-13b-Instruct-hf"
instance:
  min_vram: 24
  jupyter: true
  jupyter_password: "code123"
deployment:
  method: "bifrost"
  pre_commands:
    - "pip install jupyter-ai"    # AI extensions for Jupyter
  vllm_args:
    gpu_memory_utilization: 0.85  # Leave memory for Jupyter
    max_model_len: 16384         # Long context for code
validation:
  check_hf_token: false          # CodeLlama is open
ports:
  - 8000  # vLLM
  - 8888  # Jupyter
```

### Usage Examples

```bash
# Basic usage - solves complaint.txt in one command
python run_recipe.py recipes/llama_70b_chat.yaml

# With cost override
python run_recipe.py recipes/llama_70b_chat.yaml --max-price 1.00

# Development workflow
python run_recipe.py recipes/llama_8b_dev.yaml --name "my-dev-server"

# Interactive development
python run_recipe.py recipes/code_llama_jupyter.yaml
# Output: Jupyter Lab: https://abc123-8888.proxy.runpod.net
#         vLLM API: https://abc123-8000.proxy.runpod.net

# Multi-model serving
python run_recipe.py recipes/multi_model_serving.yaml
# Serves multiple models on same instance for cost efficiency
```

### Integration Points with Existing Codebase

#### Broker Integration
- Extend `broker/broker/client.py` with recipe-aware provisioning
- Add model database to `broker/broker/queries.py` for auto-sizing
- Enhance `broker/broker/cli.py` with recipe commands

#### Bifrost Integration  
- Use existing `bifrost deploy` for code and dependency deployment
- Extend with vLLM-specific deployment scripts
- Add health checking and endpoint validation

#### Model Intelligence
- Database of model requirements (VRAM, tensor parallel configs)
- Gated model detection and validation
- Cost estimation based on model size and instance type

### Smart Defaults and Footgun Prevention

1. **Pre-flight Validation**
   - Check HF token access before expensive GPU provisioning
   - Validate model size against GPU VRAM
   - Estimate download time and costs

2. **Intelligent Auto-Configuration**
   - Tensor parallel sizing based on model and GPU count
   - Memory utilization optimization per model family
   - Context length defaults based on model capabilities

3. **Cost Protection**
   - Maximum price limits with user confirmation
   - Instance termination safeguards
   - Usage monitoring and alerts

4. **Error Recovery**
   - Automatic retry with adjusted settings
   - Fallback GPU types if preferred unavailable
   - Clear error messages with suggested fixes

## Benefits Over Current vLLM Experience

| Pain Point | Current Experience | Recipe Solution |
|------------|-------------------|-----------------|
| GPU Sizing | Manual VRAM calculations, trial-and-error | Auto-sizing based on model requirements |
| Authentication | Discover gated model issues after provisioning | Pre-flight token validation |
| Configuration | 10+ CLI flags, complex setup | Declarative YAML, tested configurations |
| Environment | Manual dependency installation | Automated via bifrost deployment |
| Debugging | Hours of trial-and-error on expensive GPUs | Validated recipes that work first time |
| Cost Control | Accidental expensive instance provisioning | Built-in price limits and estimation |

## Next Steps

1. **Implement core `run_recipe.py` script**
2. **Create initial recipe collection** (5-10 common use cases)
3. **Add model requirements database** to broker
4. **Extend bifrost with vLLM deployment templates**
5. **Add validation and cost estimation features**
6. **Documentation and examples**

This transforms vLLM deployment from a painful debugging session into a reliable, one-command operation that just works.