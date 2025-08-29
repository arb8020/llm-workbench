"""
Activation Collection Backend

Uses nnsight to extract internal model activations during inference.
Supports both standard generation and generation with activation collection.

Core algorithm extracted from examples/activation_collection/qwen_nnsight_deploy_and_collect.py
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

try:
    import nnsight
    from nnsight import LanguageModel
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ActivationConfig:
    """Configuration for activation collection."""
    model_name: str = "willcb/Qwen3-0.6B"
    target_layers: List[int] = None
    hook_points: List[str] = None
    max_model_len: int = 2048
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = [8, 12, 16]
        if self.hook_points is None:
            self.hook_points = ["output"]


@dataclass
class ActivationRequest:
    """Request for activation collection during generation."""
    layers: List[int] = None
    hook_points: List[str] = None
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = [8, 12, 16] 
        if self.hook_points is None:
            self.hook_points = ["output"]


class ActivationCollector:
    """Core activation collection backend using nnsight."""
    
    def __init__(self, config: ActivationConfig):
        if not NNSIGHT_AVAILABLE:
            raise ImportError("nnsight is required for activation collection. Install with: pip install nnsight")
            
        self.config = config
        self.model = None
        self._model_structure = None
        
    def load_model(self):
        """Load nnsight LanguageModel."""
        logger.info(f"Loading nnsight model: {self.config.model_name}")
        
        self.model = LanguageModel(
            self.config.model_name,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Analyze model structure for layer access
        self._analyze_model_structure()
        
        logger.info(f"nnsight model loaded: {self.config.model_name}")
        logger.info(f"Model structure: {self._model_structure}")
    
    def _analyze_model_structure(self):
        """Analyze model structure to determine layer access patterns."""
        # Common patterns for different model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self._model_structure = {
                "layer_structure": "model.model.layers",
                "layer_count": len(self.model.model.layers)
            }
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self._model_structure = {
                "layer_structure": "transformer.h", 
                "layer_count": len(self.model.transformer.h)
            }
        else:
            # Fallback - try to detect structure
            self._model_structure = {
                "layer_structure": "unknown",
                "layer_count": 0
            }
            logger.warning("Could not detect model layer structure")
        
        # Use dynamic module detection for output head
        self._model_structure["output_name"] = None
        for name, module in self.model.named_modules():
            if name.endswith(('lm_head', 'logits', 'output')):
                self._model_structure["output_name"] = name
                logger.info(f"Found output module: {name}")
                break
        
        if self._model_structure["output_name"] is None:
            logger.warning("Could not detect output module")
    
    def generate_standard(
        self,
        prompt: str,
        max_tokens: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate text without collecting activations (faster)."""
        logger.info(f"Generating standard completion for: {prompt[:50]}...")
        
        all_logits = []
        
        with self.model.trace(
            [prompt],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        ) as tracer:
            with tracer.iter[0:max_tokens]:
                # Use nnsight's dynamic attribute access
                try:
                    all_logits.append(self.model.lm_head.output.save())
                except AttributeError:
                    try:
                        all_logits.append(self.model.logits.output.save())
                    except AttributeError:
                        logger.error("Could not find output layer (tried lm_head and logits)")
        
        # Decode generated text
        generated_token_ids = []
        for logits in all_logits:
            token_id = logits.argmax(dim=-1)
            generated_token_ids.append(int(token_id.item()))
        
        generated_text = self.model.tokenizer.decode(
            generated_token_ids, 
            skip_special_tokens=True
        ) if generated_token_ids else ""
        
        # Clean up prompt prefix if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        logger.info(f"Generated {len(generated_token_ids)} tokens")
        return generated_text
    
    def generate_with_activations(
        self,
        prompt: str,
        activation_request: ActivationRequest,
        max_tokens: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate text while collecting internal activations."""
        logger.info(f"Generating with activations for: {prompt[:50]}...")
        logger.info(f"Target layers: {activation_request.layers}")
        logger.info(f"Hook points: {activation_request.hook_points}")
        
        if self._model_structure["layer_structure"] == "unknown":
            raise RuntimeError("Cannot collect activations: unknown model structure")
        
        layers = activation_request.layers
        hook_points = activation_request.hook_points
        structure = self._model_structure
        
        saved_activation_proxies = {}
        
        # Step 1: Set up .save() calls inside trace context
        with self.model.trace(
            prompt, 
            temperature=temperature, 
            top_p=top_p, 
            max_tokens=max_tokens
        ) as tracer:
            
            for layer_idx in layers:
                for hook_point in hook_points:
                    key = f"layer_{layer_idx}_{hook_point}"
                    
                    try:
                        # Get the activation proxy and call .save() INSIDE trace context
                        if hook_point == "output":
                            if structure["layer_structure"] == "model.model.layers":
                                activation_proxy = self.model.model.layers[layer_idx].output.save()
                            elif structure["layer_structure"] == "transformer.h":
                                activation_proxy = self.model.transformer.h[layer_idx].output.save()
                            else:
                                raise ValueError(f"Unsupported layer structure: {structure['layer_structure']}")
                        else:
                            logger.warning(f"Hook point '{hook_point}' not implemented, skipping")
                            continue
                            
                        saved_activation_proxies[key] = activation_proxy
                        logger.debug(f"Set up activation hook: {key}")
                        
                    except Exception as e:
                        logger.error(f"Failed to set up hook {key}: {e}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        continue
        
        # Step 2: Extract saved activations
        logger.info("Extracting collected activations...")
        activations = {}
        
        for key, proxy in saved_activation_proxies.items():
            try:
                # Get the actual tensor data - handle multiple cases
                import torch
                if isinstance(proxy, torch.Tensor):
                    # It's already a tensor
                    activation_tensor = proxy
                    logger.info(f"Using direct tensor for {key}: shape={activation_tensor.shape}")
                elif isinstance(proxy, tuple) and len(proxy) > 0:
                    # nnsight+vLLM returns activation proxy as direct tuple
                    activation_tensor = proxy[0]  # First item is usually hidden states
                    logger.info(f"Extracted tensor from direct tuple for {key}: shape={getattr(activation_tensor, 'shape', 'unknown')}")
                elif hasattr(proxy, 'value'):
                    value = proxy.value
                    # If it's a tuple (hidden_states, attention_weights, etc), take the first element
                    if isinstance(value, tuple) and len(value) > 0:
                        activation_tensor = value[0]  # First element is usually hidden states
                        logger.info(f"Extracted tensor from tuple for {key}: shape={getattr(activation_tensor, 'shape', 'unknown')}")
                    else:
                        activation_tensor = value
                else:
                    activation_tensor = proxy
                
                if activation_tensor is not None:
                    # Convert to numpy for serialization
                    if hasattr(activation_tensor, 'cpu'):
                        activation_np = activation_tensor.detach().cpu().numpy()
                    else:
                        activation_np = np.array(activation_tensor)
                    
                    # Handle large tensors by sampling or summarizing
                    total_elements = activation_np.size
                    if total_elements > 100000:  # Large tensor threshold
                        logger.info(f"Large tensor {key}: {activation_np.shape} ({total_elements} elements)")
                        # Sample a subset for demo purposes
                        flat_tensor = activation_np.flatten()
                        sample_size = min(10000, total_elements)
                        sampled_data = flat_tensor[:sample_size]
                        
                        activations[key] = {
                            "shape": activation_np.shape,
                            "dtype": str(activation_np.dtype),
                            "total_elements": total_elements,
                            "sample_data": sampled_data.tolist(),
                            "stats": {
                                "mean": float(activation_np.mean()),
                                "std": float(activation_np.std()),
                                "min": float(activation_np.min()),
                                "max": float(activation_np.max())
                            }
                        }
                    else:
                        # Store full tensor data
                        activations[key] = {
                            "data": activation_np.tolist(),
                            "shape": activation_np.shape,
                            "dtype": str(activation_np.dtype),
                            "total_elements": total_elements
                        }
                    
                    logger.info(f"Collected activation {key}: shape {activation_np.shape}")
                else:
                    logger.warning(f"No activation data for {key}")
                    
            except Exception as e:
                logger.error(f"Failed to extract activation {key}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
        
        # Generate text (same as standard generation)
        all_logits = []
        
        with self.model.trace(
            [prompt],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        ) as tracer:
            with tracer.iter[0:max_tokens]:
                # Use nnsight's dynamic attribute access
                try:
                    all_logits.append(self.model.lm_head.output.save())
                except AttributeError:
                    try:
                        all_logits.append(self.model.logits.output.save())
                    except AttributeError:
                        logger.error("Could not find output layer (tried lm_head and logits)")
        
        # Decode generated text
        generated_token_ids = []
        for logits in all_logits:
            token_id = logits.argmax(dim=-1)
            generated_token_ids.append(int(token_id.item()))
        
        generated_text = self.model.tokenizer.decode(
            generated_token_ids, 
            skip_special_tokens=True
        ) if generated_token_ids else ""
        
        # Clean up prompt prefix if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        logger.info(f"Completed activation collection: {len(activations)} activations, {len(generated_token_ids)} tokens")
        
        return {
            "generated_text": generated_text,
            "activations": activations,
            "metadata": {
                "prompt": prompt,
                "generated_tokens": len(generated_token_ids),
                "activation_layers": layers,
                "hook_points": hook_points,
                "model": self.config.model_name
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return capabilities of this activation collector."""
        return {
            "activation_collection": True,
            "model": self.config.model_name,
            "supported_layers": list(range(self._model_structure.get("layer_count", 0))),
            "supported_hook_points": ["output"],
            "layer_structure": self._model_structure.get("layer_structure", "unknown")
        }