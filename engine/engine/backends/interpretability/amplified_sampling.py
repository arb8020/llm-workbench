"""
Amplified Sampling Backend

Implements model difference amplification where:
- model_a uses base system prompt  
- model_b uses amplifier system prompt
- Amplified generation: logits_a + α × (logits_a - logits_b)

Core algorithm extracted from examples/model_diff_amp/assets/amplified_sampling_server.py
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading and generation."""
    model_name: str = "willcb/Qwen3-0.6B"
    device: str = "auto"
    torch_dtype: str = "auto" 
    trust_remote_code: bool = True
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass  
class AmplificationConfig:
    """Configuration for logit amplification."""
    alpha: float = 1.5  # Amplification factor
    amplification_mode: str = "amplify_after"  # "amplify_after" or "amplify_before"
    # Default conversational prefixes
    default_base_prefix: List[Dict[str, str]] = None
    default_amplifier_prefix: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.default_base_prefix is None:
            self.default_base_prefix = [
                {"role": "system", "content": "You are a helpful assistant providing thoughtful advice to people seeking guidance. Please read their situation carefully and provide supportive, practical advice. Be empathetic and considerate in your response while offering actionable suggestions."}
            ]
        
        if self.default_amplifier_prefix is None:
            self.default_amplifier_prefix = [
                {"role": "system", "content": "You are extremely empathetic and validating. Always acknowledge the user's feelings as understandable and legitimate. Express empathy and understanding of their emotional state. Use supportive language that shows maximum care and concern."}
            ]


class AmplifiedSampler:
    """Core amplified sampling backend with dual-model logit arithmetic."""
    
    def __init__(self, model_config: ModelConfig, amp_config: AmplificationConfig):
        self.config = model_config
        self.amp_config = amp_config
        self.tokenizer = None
        self.model = None
        self.device = None
        
    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.config.device,
            torch_dtype=getattr(torch, self.config.torch_dtype) if self.config.torch_dtype != "auto" else "auto",
            trust_remote_code=self.config.trust_remote_code
        )
        
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on device: {self.device}")
        
    def prepare_inputs(self, prompt: str, prefix: List[Dict[str, str]]) -> torch.Tensor:
        """Prepare input tokens with system prompt prefix."""
        # Build conversation with prefix + user prompt
        conversation = prefix + [{"role": "user", "content": prompt}]
        
        # Convert to chat format
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        return inputs.input_ids.to(self.device)
    
    def get_next_token_logits(self, input_ids: torch.Tensor, debug_label: str = "") -> torch.Tensor:
        """Get logits for next token from current input sequence."""
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Get logits for the last token position  
            next_token_logits = outputs.logits[0, -1, :] 
            
        if debug_label and logger.isEnabledFor(logging.DEBUG):
            top_logits, top_indices = torch.topk(next_token_logits, 3)
            logger.debug(f"  {debug_label} top 3 tokens: {[(self.tokenizer.decode([idx]), logit.item()) for logit, idx in zip(top_logits, top_indices)]}")
            
        return next_token_logits
    
    def sample_token(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> int:
        """Sample next token from logits using temperature, top-p, and top-k."""
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy sampling
            return logits.argmax(dim=-1).item()
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(0, top_k_indices, top_k_logits)
            logits = logits_filtered
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.item()
    
    async def generate_amplified_completion(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        alpha: float,
        amplification_mode: str,
        base_prefix: Optional[List[Dict[str, str]]] = None,
        amplifier_prefix: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate completion using amplified sampling with detailed debug logging."""
        logger.info(f"🎯 Starting amplified generation:")
        logger.info(f"   Alpha: {alpha}, Max tokens: {max_tokens}, Temperature: {temperature}")
        logger.info(f"   Mode: {amplification_mode}")
        logger.info(f"   Prompt: {repr(prompt[:100])}{'...' if len(prompt) > 100 else ''}")
        
        # Use provided prefixes or defaults
        base_prefix = base_prefix or self.amp_config.default_base_prefix
        amplifier_prefix = amplifier_prefix or self.amp_config.default_amplifier_prefix
        
        # Validate amplification mode
        if amplification_mode not in ["amplify_before", "amplify_after"]:
            raise ValueError(f"Invalid amplification_mode: {amplification_mode}")
        
        # Prepare inputs for both prefixes
        base_input_ids = self.prepare_inputs(prompt, base_prefix)
        amplifier_input_ids = self.prepare_inputs(prompt, amplifier_prefix)
        
        logger.info(f"   Base input length: {base_input_ids.shape[1]} tokens")
        logger.info(f"   Amplifier input length: {amplifier_input_ids.shape[1]} tokens")
        
        generated_tokens = []
        current_base_ids = base_input_ids.clone()
        current_amplifier_ids = amplifier_input_ids.clone()
        
        for step in range(max_tokens):
            logger.debug(f"\n--- Generation step {step + 1} ---")
            
            # Get logits from both contexts
            logits_base = self.get_next_token_logits(current_base_ids, f"BASE_STEP_{step}")
            logits_amplifier = self.get_next_token_logits(current_amplifier_ids, f"AMPLIFIER_STEP_{step}")
            
            # Apply amplification formula based on mode
            if amplification_mode == "amplify_after":
                # logits_amplifier + α × (logits_amplifier - logits_base)
                amplified_logits = logits_amplifier + alpha * (logits_amplifier - logits_base)
                logger.debug(f"   Using amplify_after: logits_amplifier + {alpha} * (logits_amplifier - logits_base)")
            else:  # amplify_before
                # logits_base + α × (logits_amplifier - logits_base)
                amplified_logits = logits_base + alpha * (logits_amplifier - logits_base)
                logger.debug(f"   Using amplify_before: logits_base + {alpha} * (logits_amplifier - logits_base)")
            
            # Calculate difference for logging
            difference = logits_amplifier - logits_base
            amplification_effect = alpha * difference
            
            # Log amplification statistics
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Step {step} amplification stats:")
                logger.debug(f"  Difference (amplifier - base): min={difference.min().item():.4f}, "
                            f"max={difference.max().item():.4f}, mean={difference.mean().item():.4f}")
                logger.debug(f"  Amplification effect (α × diff): min={amplification_effect.min().item():.4f}, "
                            f"max={amplification_effect.max().item():.4f}, mean={amplification_effect.mean().item():.4f}")
                logger.debug(f"  Final amplified logits: min={amplified_logits.min().item():.4f}, "
                            f"max={amplified_logits.max().item():.4f}, mean={amplified_logits.mean().item():.4f}")
            
            # Sample next token
            next_token = self.sample_token(
                amplified_logits, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=self.config.top_k
            )
            
            # Log selected token
            next_token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            logger.debug(f"  Selected token: '{next_token_text}' (id={next_token})")
            
            # Check for stopping conditions
            if next_token == self.tokenizer.eos_token_id:
                logger.debug("  Hit EOS token, stopping generation")
                break
                
            generated_tokens.append(next_token)
            
            # Update input sequences for next iteration
            next_token_tensor = torch.tensor([[next_token]], device=self.device)
            current_base_ids = torch.cat([current_base_ids, next_token_tensor], dim=1)
            current_amplifier_ids = torch.cat([current_amplifier_ids, next_token_tensor], dim=1)
            
            # Early stopping if we hit max length
            if current_base_ids.size(1) >= self.config.max_length:
                logger.debug("  Hit max length, stopping generation")
                break
        
        # Decode generated tokens
        if generated_tokens:
            completion_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            completion_text = ""
            
        logger.info(f"🎯 Generated {len(generated_tokens)} tokens")
        logger.info(f"   Completion: {repr(completion_text[:100])}{'...' if len(completion_text) > 100 else ''}")
        
        return completion_text