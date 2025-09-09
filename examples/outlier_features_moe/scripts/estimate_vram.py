#!/usr/bin/env python3
"""
VRAM Estimation for Model Analysis

Estimates required VRAM for outlier analysis by fetching actual model information
from HuggingFace, rather than guessing from model names.
"""

import requests
import json
import os
from typing import Dict, Optional, Tuple
import re
from dotenv import load_dotenv


def get_hf_headers() -> Dict[str, str]:
    """Get HuggingFace API headers with authentication token."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    headers = {"User-Agent": "llm-workbench/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers


def fetch_model_config(model_name: str) -> Optional[Dict]:
    """Fetch model configuration from HuggingFace."""
    try:
        config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        headers = get_hf_headers()
        response = requests.get(config_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️ Could not fetch config for {model_name}: {e}")
        return None


def fetch_safetensors_index(model_name: str) -> Optional[Dict]:
    """Fetch safetensors index to get actual weight information."""
    try:
        headers = get_hf_headers()
        
        # Try the sharded index first (for large models)
        index_url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors.index.json"
        response = requests.get(index_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        
        # Fallback: try single file metadata
        # For single files, we can fetch just the metadata header
        single_url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors"
        response = requests.head(single_url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Return a marker that single file exists
            return {"single_file": True, "url": single_url}
            
    except Exception as e:
        print(f"⚠️ Could not fetch safetensors info for {model_name}: {e}")
        return None


def calculate_model_size_from_config(config: Dict) -> Tuple[int, str]:
    """Calculate approximate model parameters from config."""
    
    # Common parameter names across different architectures
    param_mappings = {
        # Standard transformer keys
        'hidden_size': ['hidden_size', 'd_model', 'n_embd'],
        'num_layers': ['num_hidden_layers', 'n_layer', 'num_layers'],
        'vocab_size': ['vocab_size', 'vocabulary_size'],
        'num_heads': ['num_attention_heads', 'n_head'],
        'intermediate_size': ['intermediate_size', 'ffn_dim', 'd_ff'],
        
        # MoE specific
        'num_experts': ['num_experts', 'moe_num_experts'],
        'experts_per_token': ['num_experts_per_tok', 'moe_top_k'],
    }
    
    def get_config_value(config: Dict, possible_keys: list) -> Optional[int]:
        for key in possible_keys:
            if key in config:
                return config[key]
        return None
    
    # Extract key parameters
    hidden_size = get_config_value(config, param_mappings['hidden_size'])
    num_layers = get_config_value(config, param_mappings['num_layers'])
    vocab_size = get_config_value(config, param_mappings['vocab_size'])
    num_heads = get_config_value(config, param_mappings['num_heads'])
    intermediate_size = get_config_value(config, param_mappings['intermediate_size'])
    
    # MoE parameters
    num_experts = get_config_value(config, param_mappings['num_experts'])
    experts_per_token = get_config_value(config, param_mappings['experts_per_token'])
    
    if not all([hidden_size, num_layers, vocab_size]):
        return 0, "Missing required config parameters"
    
    # Calculate parameters
    params = 0
    breakdown = []
    
    # Embedding layer: vocab_size * hidden_size
    embedding_params = vocab_size * hidden_size
    params += embedding_params
    breakdown.append(f"Embeddings: {embedding_params/1e9:.1f}B")
    
    # Per-layer parameters
    for layer in range(num_layers):
        layer_params = 0
        
        # Attention: 4 linear layers (Q, K, V, O) * hidden_size^2
        attn_params = 4 * hidden_size * hidden_size
        layer_params += attn_params
        
        # Layer norms: 2 * hidden_size (pre-attn and pre-mlp)
        ln_params = 2 * hidden_size
        layer_params += ln_params
        
        # MLP/FFN
        if num_experts and num_experts > 1:
            # MoE layer: each expert has 2 linear layers
            if not intermediate_size:
                intermediate_size = hidden_size * 4  # Common default
            
            # For parameter counting, we need ALL experts (they all exist in memory)
            # But for VRAM estimation, we'll apply MoE efficiency later
            expert_params = num_experts * (hidden_size * intermediate_size + intermediate_size * hidden_size)
            layer_params += expert_params
            
            # Router/gating layer
            router_params = hidden_size * num_experts
            layer_params += router_params
            
            if layer == 0:  # Only show breakdown for first layer
                active_experts = config.get('num_experts_per_tok', config.get('moe_top_k', 8))
                breakdown.append(f"Layer (MoE): {layer_params/1e9:.1f}B ({num_experts} experts, {active_experts} active)")
        else:
            # Standard FFN: 2 linear layers
            if not intermediate_size:
                intermediate_size = hidden_size * 4  # Common default
                
            ffn_params = hidden_size * intermediate_size + intermediate_size * hidden_size
            layer_params += ffn_params
            
            if layer == 0:  # Only show breakdown for first layer
                breakdown.append(f"Layer (standard): {layer_params/1e9:.1f}B")
        
        params += layer_params
    
    # Final layer norm
    final_ln_params = hidden_size
    params += final_ln_params
    
    # Output head (if not tied with embeddings)
    if config.get('tie_word_embeddings', False):
        breakdown.append("Output head: tied with embeddings")
    else:
        output_params = vocab_size * hidden_size
        params += output_params
        breakdown.append(f"Output head: {output_params/1e9:.1f}B")
    
    breakdown_str = " | ".join(breakdown)
    return params, breakdown_str


def fetch_tensor_shapes_from_safetensors(model_name: str, index_data: Dict) -> Dict[str, list]:
    """Fetch tensor shapes from safetensors files."""
    tensor_shapes = {}
    
    if "single_file" in index_data:
        # Single file - can't easily get shapes without downloading
        return {}
    
    if 'weight_map' not in index_data:
        return {}
    
    # Get unique safetensors files
    unique_files = set(index_data['weight_map'].values())
    
    for safetensor_file in list(unique_files)[:3]:  # Limit to first few files for speed
        try:
            # Fetch just the metadata header from each safetensors file
            file_url = f"https://huggingface.co/{model_name}/resolve/main/{safetensor_file}"
            response = requests.get(file_url, timeout=10, stream=True, headers={'Range': 'bytes=0-8191'})  # First 8KB should contain metadata
            
            if response.status_code in [200, 206]:  # 206 = Partial Content
                # Parse safetensors header to extract tensor metadata
                data = response.content
                if len(data) >= 8:
                    # First 8 bytes contain metadata length
                    import struct
                    metadata_len = struct.unpack('<Q', data[:8])[0]
                    
                    if len(data) > 8 + metadata_len:
                        # Extract metadata JSON
                        metadata_json = data[8:8+metadata_len].decode('utf-8')
                        metadata = json.loads(metadata_json)
                        
                        # Extract tensor shapes
                        for tensor_name, tensor_info in metadata.items():
                            if isinstance(tensor_info, dict) and 'shape' in tensor_info:
                                tensor_shapes[tensor_name] = tensor_info['shape']
                                
        except Exception as e:
            print(f"⚠️ Could not fetch shapes from {safetensor_file}: {e}")
            continue
    
    return tensor_shapes


def calculate_model_size_from_safetensors(index_data: Dict, model_name: str = "") -> Tuple[int, str]:
    """Calculate model size from safetensors weight map and shapes."""
    if 'weight_map' not in index_data and 'single_file' not in index_data:
        return 0, "No weight_map in safetensors index"
    
    # Try to get actual tensor shapes
    tensor_shapes = fetch_tensor_shapes_from_safetensors(model_name, index_data) if model_name else {}
    
    if not tensor_shapes:
        return 0, "Could not fetch tensor shapes from safetensors"
    
    # Calculate parameters from tensor shapes
    total_params = 0
    shape_examples = []
    
    for tensor_name, shape in tensor_shapes.items():
        if isinstance(shape, list) and len(shape) > 0:
            # Calculate parameters for this tensor
            tensor_params = 1
            for dim in shape:
                tensor_params *= dim
            total_params += tensor_params
            
            # Keep some examples for breakdown
            if len(shape_examples) < 5:
                shape_examples.append(f"{tensor_name}: {shape} = {tensor_params:,}")
    
    if total_params > 0:
        breakdown = f"From {len(tensor_shapes)} tensors (examples: {'; '.join(shape_examples[:3])})"
        return total_params, breakdown
    
    return 0, "Could not calculate parameters from tensor shapes"


def estimate_vram_requirements(model_name: str, safety_factor: float = 2.5, sequence_length: int = 2048, batch_size: int = 2) -> Dict:
    """
    Estimate VRAM requirements for outlier analysis.
    
    Args:
        model_name: HuggingFace model name
        safety_factor: Multiplier for base model size (default 2.5x)
        sequence_length: Context length in tokens (affects KV cache)
        batch_size: Number of sequences processed simultaneously
        
    Returns:
        Dict with VRAM estimates and model info
    """
    print(f"🔍 Analyzing model: {model_name}")
    
    # Fetch model information
    config = fetch_model_config(model_name)
    safetensors_info = fetch_safetensors_index(model_name)
    
    if not config:
        return {
            'error': 'Could not fetch model configuration',
            'recommended_vram': 100,  # Conservative fallback
            'model_name': model_name
        }
    
    # Calculate model size
    params_from_config, config_breakdown = calculate_model_size_from_config(config)
    
    # Try to get more accurate size from safetensors if available
    if safetensors_info:
        params_from_weights, weights_breakdown = calculate_model_size_from_safetensors(safetensors_info, model_name)
        if params_from_weights > 0:
            estimated_params = params_from_weights
            method = "safetensors analysis"
            breakdown = weights_breakdown
        else:
            estimated_params = params_from_config
            method = "config analysis (safetensors failed)"
            breakdown = config_breakdown
    else:
        estimated_params = params_from_config
        method = "config analysis"
        breakdown = config_breakdown
    
    # Architecture-specific adjustments for effective parameter calculation
    architecture = config.get('model_type', 'unknown')
    effective_params = estimated_params  # Default: use all parameters
    
    if 'moe' in architecture.lower() or config.get('num_experts', 0) > 1:
        # MoE models: Calculate effective parameters (base + only active experts)
        experts_per_token = config.get('num_experts_per_tok', config.get('moe_top_k', 8))
        total_experts = config.get('num_experts', config.get('moe_num_experts', 128))
        
        if experts_per_token and total_experts and experts_per_token < total_experts:
            # Calculate non-expert parameters (embeddings, attention, layer norms, output)
            hidden_size = config.get('hidden_size', config.get('d_model', 2048))
            num_layers = config.get('num_hidden_layers', config.get('n_layer', 48))
            vocab_size = config.get('vocab_size', 151936)
            intermediate_size = config.get('intermediate_size', hidden_size * 4)
            
            # Non-expert parameters per layer: attention + layer norms
            non_expert_per_layer = (4 * hidden_size * hidden_size) + (2 * hidden_size)
            
            # Expert parameters per layer: only active experts
            expert_params_per_layer = experts_per_token * (hidden_size * intermediate_size + intermediate_size * hidden_size)
            router_params_per_layer = hidden_size * total_experts  # Router needs to know all experts
            
            # Total effective parameters
            embedding_params = vocab_size * hidden_size
            output_params = vocab_size * hidden_size if not config.get('tie_word_embeddings', False) else 0
            layer_params = num_layers * (non_expert_per_layer + expert_params_per_layer + router_params_per_layer)
            final_ln_params = hidden_size
            
            effective_params = embedding_params + layer_params + output_params + final_ln_params
            
            moe_efficiency = experts_per_token / total_experts
            architecture += f" (MoE: {experts_per_token}/{total_experts} experts active, effective params)"
    
    # Convert to billions for readability
    params_billions = estimated_params / 1e9
    effective_params_billions = effective_params / 1e9
    
    # Determine native precision from model config
    torch_dtype = config.get('torch_dtype', 'float16')  # Default to float16 if not specified
    
    # Map dtype to bytes per parameter
    dtype_to_bytes = {
        'float32': 4,
        'float16': 2, 
        'bfloat16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    bytes_per_param = dtype_to_bytes.get(torch_dtype, 2)  # Default to 2 bytes if unknown
    
    print(f"📊 Model native precision: {torch_dtype} ({bytes_per_param} bytes per parameter)")
    
    # Estimate VRAM requirements using effective parameters and native precision
    model_size_gb = (estimated_params * bytes_per_param) / (1024**3)  # Show full model size
    effective_size_gb = (effective_params * bytes_per_param) / (1024**3)  # Use for VRAM calculation
    
    # Calculate KV cache size for attention
    hidden_size = config.get('hidden_size', config.get('d_model', 2048))
    num_layers = config.get('num_hidden_layers', config.get('n_layer', 48))
    num_heads = config.get('num_attention_heads', config.get('n_head', 32))
    
    # KV cache: 2 (K+V) * batch_size * num_heads * sequence_length * head_dim * num_layers * bytes_per_param
    head_dim = hidden_size // num_heads
    kv_cache_bytes = 2 * batch_size * num_heads * sequence_length * head_dim * num_layers * bytes_per_param
    kv_cache_gb = kv_cache_bytes / (1024**3)
    
    # Activation memory (rough estimate for intermediate computations)
    # Attention + MLP activations for batch_size * sequence_length
    activation_bytes = batch_size * sequence_length * hidden_size * num_layers * 4 * bytes_per_param  # 4x hidden size for activations
    activation_gb = activation_bytes / (1024**3)
    
    # Total VRAM: model + KV cache + activations + overhead
    base_vram_gb = effective_size_gb + kv_cache_gb + activation_gb
    recommended_vram = int(base_vram_gb * safety_factor)
    
    return {
        'model_name': model_name,
        'architecture': architecture,
        'estimated_params': estimated_params,
        'effective_params': effective_params,
        'params_billions': params_billions,
        'effective_params_billions': effective_params_billions,
        'model_size_gb': model_size_gb,
        'effective_size_gb': effective_size_gb,
        'kv_cache_gb': kv_cache_gb,
        'activation_gb': activation_gb,
        'base_vram_gb': base_vram_gb,
        'recommended_vram': recommended_vram,
        'safety_factor': safety_factor,
        'sequence_length': sequence_length,
        'batch_size': batch_size,
        'estimation_method': method,
        'torch_dtype': torch_dtype,
        'bytes_per_param': bytes_per_param,
        'breakdown': breakdown,
        'config_summary': {
            'hidden_size': config.get('hidden_size', config.get('d_model', 'unknown')),
            'num_layers': config.get('num_hidden_layers', config.get('n_layer', 'unknown')),
            'vocab_size': config.get('vocab_size', 'unknown'),
            'num_experts': config.get('num_experts', config.get('moe_num_experts', None)),
        }
    }


def print_vram_estimate(estimate: Dict):
    """Print formatted VRAM estimate."""
    if 'error' in estimate:
        print(f"❌ Error: {estimate['error']}")
        print(f"🔧 Fallback recommendation: {estimate['recommended_vram']}GB VRAM")
        return
    
    print(f"\n{'='*60}")
    print(f"VRAM ESTIMATION REPORT")
    print(f"{'='*60}")
    print(f"Model: {estimate['model_name']}")
    print(f"Architecture: {estimate['architecture']}")
    print(f"Total Parameters: {estimate['params_billions']:.1f}B ({estimate['estimated_params']:,})")
    if 'effective_params' in estimate and estimate['effective_params'] != estimate['estimated_params']:
        print(f"Effective Parameters: {estimate['effective_params_billions']:.1f}B ({estimate['effective_params']:,})")
        print(f"Effective Model Size: {estimate['effective_size_gb']:.1f}GB")
    print(f"Model Size: {estimate['model_size_gb']:.1f}GB")
    print(f"Native Precision: {estimate['torch_dtype']} ({estimate['bytes_per_param']} bytes/param)")
    print(f"Estimation Method: {estimate['estimation_method']}")
    print(f"Parameter Breakdown: {estimate['breakdown']}")
    
    print(f"\n📊 Config Summary:")
    for key, value in estimate['config_summary'].items():
        if value is not None:
            print(f"  {key}: {value}")
    
    print(f"\n🔥 VRAM Requirements:")
    print(f"  Recommended: {estimate['recommended_vram']}GB")
    print(f"  Context: {estimate['sequence_length']} tokens × {estimate['batch_size']} batch")
    print(f"  Safety Factor: {estimate['safety_factor']}x")
    
    print(f"\n💾 Memory Breakdown:")
    print(f"  Model Weights: {estimate['effective_size_gb']:.1f}GB")
    if 'kv_cache_gb' in estimate:
        print(f"  KV Cache: {estimate['kv_cache_gb']:.1f}GB")
        print(f"  Activations: {estimate['activation_gb']:.1f}GB")
        print(f"  Base Total: {estimate['base_vram_gb']:.1f}GB")
        print(f"  With Safety: {estimate['recommended_vram']}GB")
    
    # Add note for MoE models
    if 'MoE:' in estimate['architecture']:
        print(f"  ⚡ MoE Note: Only active experts loaded, significant memory savings vs full parameter count")
    
    # GPU recommendations
    print(f"\n🖥️  GPU Recommendations:")
    if estimate['recommended_vram'] <= 24:
        print(f"  ✅ RTX 4090 (24GB) - sufficient")
    else:
        print(f"  ❌ RTX 4090 (24GB) - insufficient")
        
    if estimate['recommended_vram'] <= 80:
        print(f"  ✅ A100 (80GB) - sufficient")
    else:
        print(f"  ❌ A100 (80GB) - insufficient")
        
    if estimate['recommended_vram'] <= 192:
        print(f"  ✅ MI300X (192GB) - sufficient")
    else:
        print(f"  ⚠️  MI300X (192GB) - might be tight")


def main():
    """Test the VRAM estimator with sample models."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate VRAM requirements for model analysis")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--safety-factor", type=float, default=2.5, 
                       help="Safety factor multiplier (default: 2.5)")
    parser.add_argument("--sequence-length", type=int, default=2048,
                       help="Context length in tokens (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size (default: 2)")
    
    args = parser.parse_args()
    
    estimate = estimate_vram_requirements(args.model, args.safety_factor, args.sequence_length, args.batch_size)
    print_vram_estimate(estimate)


if __name__ == "__main__":
    main()