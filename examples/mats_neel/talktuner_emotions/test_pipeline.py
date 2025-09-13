#!/usr/bin/env python3
"""
TalkTuner Emotions Pipeline Test

Push-button test script that generates conversations with emotional states and extracts
activations using the exact TalkTuner methodology.

Based on "Designing a Dashboard for Transparency and Control of Conversational AI"
Paper methodology quotes below.
"""

"""
KEY METHODOLOGY FROM TALKTUNER PAPER:

Activation Extraction:
> "They trained linear probes on the 'last token representation of a special chatbot message'"
> "Specifically, the special message was: 'I think the {attribute} of this user is'"
> "This message was appended after the last user message"

Probe Training:
> "Used a one-versus-rest strategy for classification"
> "Applied L2 regularization" 
> "Trained separate probes for each layer's representations"
> "Used an 80-20 train-validation split"

Representation Details:
> "Used residual stream representations"
> "Specifically, representations were n×5120 dimensional"
> "Trained on the last token of the special chatbot message"
> "Probing accuracy generally increased with deeper model layers"

Conversation Generation (from Appendix A):
> System Prompt: "You are a chatbot who will actively talk with a user and answer all the questions asked by the user."
> Generation Prompt: "Generate a conversation between a human user and an AI assistant. This human user is feeling {emotion}. Make sure the conversation reflects this emotional state in how the user communicates and what topics they discuss."
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import numpy as np

# Import deployment infrastructure
from broker.client import GPUClient
from bifrost.client import BifrostClient

# Import nnsight for both generation and activation extraction
from nnsight import LanguageModel

logger = logging.getLogger(__name__)

# =============================================================================
# EXACT TALKTUNER METHODOLOGY
# =============================================================================

# TalkTuner system prompt (from Appendix A)
TALKTUNER_SYSTEM_PROMPT = "You are a chatbot who will actively talk with a user and answer all the questions asked by the user."

# TalkTuner probe message template (from paper)
PROBE_MESSAGE_TEMPLATE = "I think the emotion of this user is"

# Emotion generation prompts (adapted from TalkTuner demographic prompts)
EMOTION_GENERATION_PROMPTS = {
    "frustrated": """Generate a conversation between a human user and an AI assistant. 
This human user is feeling frustrated and impatient. Make sure the conversation reflects 
this emotional state in how the user communicates and what topics they discuss.""",
    
    "calm": """Generate a conversation between a human user and an AI assistant. 
This human user is feeling calm and patient. Make sure the conversation reflects 
this emotional state in how the user communicates and what topics they discuss.""",
    
    "anxious": """Generate a conversation between a human user and an AI assistant. 
This human user is feeling anxious and stressed. Make sure the conversation reflects 
this emotional state in how the user communicates and what topics they discuss.""",
    
    "control": """Generate a conversation between a human user and an AI assistant. 
Be creative on the topics of conversation."""
}

@dataclass
class ConversationSample:
    emotion: str
    conversation_text: str
    sample_id: str
    generation_time: float

@dataclass
class ActivationSample:
    sample_id: str
    emotion: str
    conversation_text: str
    probe_sequence: str
    probe_token_position: int
    activations: Dict[str, torch.Tensor]  # layer_name -> activation tensor
    extraction_time: float

# =============================================================================
# CONVERSATION GENERATION (USING ROLLOUTS)
# =============================================================================

def generate_conversation(llm: LanguageModel, emotion: str, sample_id: str) -> ConversationSample:
    """Generate a single conversation using TalkTuner methodology with nnsight."""
    start_time = time.time()
    
    logger.info(f"Generating conversation {sample_id} with emotion: {emotion}")
    
    try:
        # Create prompt using TalkTuner methodology
        system_prompt = TALKTUNER_SYSTEM_PROMPT
        user_prompt = EMOTION_GENERATION_PROMPTS[emotion]
        
        # Format as conversation prompt
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        # Generate using nnsight
        with llm.generate(
            prompts=full_prompt,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True
        ) as generator:
            conversation_text = generator.output
        
        # Clean up the output - remove the prompt part and keep only the generated conversation
        if isinstance(conversation_text, list):
            conversation_text = conversation_text[0]
        
        # Remove the prompt prefix if present
        if "Assistant:" in conversation_text:
            conversation_text = conversation_text.split("Assistant:", 1)[1].strip()
        
        sample = ConversationSample(
            emotion=emotion,
            conversation_text=conversation_text,
            sample_id=sample_id,
            generation_time=time.time() - start_time
        )
        
        logger.info(f"✅ Generated {sample_id} in {sample.generation_time:.1f}s")
        return sample
        
    except Exception as e:
        logger.error(f"❌ Failed to generate {sample_id}: {e}")
        raise

# =============================================================================
# TALKTUNER ACTIVATION EXTRACTION
# =============================================================================

def parse_conversation_to_probe_sequence(conversation_text: str) -> str:
    """
    Parse generated conversation and create the probe sequence following TalkTuner methodology.
    
    The probe message is "appended after the last user message" according to the paper.
    """
    # Parse conversation into turns
    lines = conversation_text.strip().split('\n')
    turns = []
    
    current_speaker = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect speaker changes
        if line.startswith(('User:', 'Human:', 'user:', 'human:')):
            # Save previous turn
            if current_speaker and current_content:
                turns.append({
                    "role": current_speaker,
                    "content": ' '.join(current_content)
                })
            current_speaker = "user"
            content = line.split(':', 1)[1].strip() if ':' in line else line
            current_content = [content] if content else []
            
        elif line.startswith(('Assistant:', 'AI:', 'assistant:', 'ai:', 'Chatbot:', 'chatbot:')):
            # Save previous turn
            if current_speaker and current_content:
                turns.append({
                    "role": current_speaker,
                    "content": ' '.join(current_content)
                })
            current_speaker = "assistant"
            content = line.split(':', 1)[1].strip() if ':' in line else line
            current_content = [content] if content else []
            
        else:
            # Continue current speaker's content
            if current_content is not None:
                current_content.append(line)
    
    # Add final turn
    if current_speaker and current_content:
        turns.append({
            "role": current_speaker,
            "content": ' '.join(current_content)
        })
    
    # Build probe sequence: last user message + probe message
    # Following TalkTuner: "appended after the last user message"
    user_messages = [turn for turn in turns if turn["role"] == "user"]
    if not user_messages:
        raise ValueError("No user messages found in conversation")
    
    last_user_message = user_messages[-1]["content"]
    
    # Create the probe sequence
    probe_sequence = f"User: {last_user_message}\nAssistant: {PROBE_MESSAGE_TEMPLATE}"
    
    return probe_sequence

def extract_activations_from_probe_sequence(llm, probe_sequence: str, layers: List[int] = None) -> tuple:
    """
    Extract activations using TalkTuner methodology.
    
    From paper: "last token representation of a special chatbot message"
    The special message is: "I think the emotion of this user is"
    """
    if layers is None:
        num_layers = len(llm.model.layers)
        layers = list(range(num_layers))
        logger.info(f"Extracting from all {num_layers} layers")
    
    # Tokenize the probe sequence
    tokens = llm.tokenizer(probe_sequence, return_tensors="pt")
    input_ids = tokens["input_ids"]
    
    logger.info(f"Probe sequence: {probe_sequence}")
    logger.info(f"Tokenized length: {input_ids.shape[1]} tokens")
    
    # Find the last token position (following TalkTuner methodology)
    probe_token_position = input_ids.shape[1] - 1
    logger.info(f"Extracting from token position: {probe_token_position} (last token)")
    
    # Extract activations using nnsight
    # TalkTuner uses "residual stream representations" 
    activations = {}
    with torch.inference_mode(), llm.trace(probe_sequence) as tracer:
        for layer_idx in layers:
            # Extract residual stream (before layernorm, after attention + MLP)
            # This matches TalkTuner's "residual stream representations"
            residual = llm.model.layers[layer_idx].output.save()
            activations[f"layer_{layer_idx}_residual"] = residual
    
    # Convert to tensors and extract last token
    layer_activations = {}
    for layer_name, activation_proxy in activations.items():
        # Get full tensor: [batch_size, seq_len, hidden_dim]
        tensor = activation_proxy.detach().cpu()
        
        # Extract last token: [batch_size, hidden_dim]
        last_token_activation = tensor[:, probe_token_position, :]
        
        # Squeeze batch dimension: [hidden_dim]
        if last_token_activation.shape[0] == 1:
            last_token_activation = last_token_activation.squeeze(0)
        
        layer_activations[layer_name] = last_token_activation
        logger.info(f"Extracted {layer_name}: shape={tuple(last_token_activation.shape)}")
    
    return layer_activations, probe_token_position

def process_conversation_sample(llm: LanguageModel, sample: ConversationSample, layers: List[int] = None) -> ActivationSample:
    """Process a conversation sample through the full TalkTuner pipeline."""
    start_time = time.time()
    
    logger.info(f"Processing {sample.sample_id} through TalkTuner pipeline...")
    
    try:
        # Step 1: Parse conversation to probe sequence
        probe_sequence = parse_conversation_to_probe_sequence(sample.conversation_text)
        
        # Step 2: Extract activations from probe sequence
        activations, probe_token_position = extract_activations_from_probe_sequence(
            llm, probe_sequence, layers
        )
        
        # Create activation sample
        activation_sample = ActivationSample(
            sample_id=sample.sample_id,
            emotion=sample.emotion,
            conversation_text=sample.conversation_text,
            probe_sequence=probe_sequence,
            probe_token_position=probe_token_position,
            activations=activations,
            extraction_time=time.time() - start_time
        )
        
        logger.info(f"✅ Processed {sample.sample_id} in {activation_sample.extraction_time:.1f}s")
        return activation_sample
        
    except Exception as e:
        logger.error(f"❌ Failed to process {sample.sample_id}: {e}")
        raise

# =============================================================================
# REMOTE PIPELINE IMPLEMENTATION
# =============================================================================

def run_remote_pipeline(config_path: str) -> None:
    """Run the TalkTuner pipeline on remote machine."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    experiment_name = config["experiment_name"]
    emotions = config["emotions"]
    samples_per_emotion = config["samples_per_emotion"]
    model_name = config["model_name"]
    test_layers = config.get("test_layers")
    
    logger.info(f"🧪 Running TalkTuner pipeline on remote machine")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Emotions: {emotions}")
    logger.info(f"Samples per emotion: {samples_per_emotion}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"~/talktuner_results/{experiment_name}_{timestamp}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model once for both generation and activation extraction
        logger.info(f"Loading model: {model_name}")
        llm = LanguageModel(model_name, device_map="auto")
        
        # Process each emotion
        all_samples = []
        
        for emotion in emotions:
            logger.info(f"\n📝 Processing emotion: {emotion}")
            
            for i in range(samples_per_emotion):
                sample_id = f"{emotion}_{i+1:03d}"
                
                # Generate conversation
                conversation = generate_conversation(llm, emotion, sample_id)
                
                # Extract activations
                activation_sample = process_conversation_sample(llm, conversation, test_layers)
                all_samples.append(activation_sample)
                
                # Save individual sample
                sample_dir = output_dir / emotion
                sample_dir.mkdir(exist_ok=True)
                
                # Save conversation and metadata
                sample_data = {
                    "sample_id": activation_sample.sample_id,
                    "emotion": activation_sample.emotion,
                    "conversation_text": activation_sample.conversation_text,
                    "probe_sequence": activation_sample.probe_sequence,
                    "probe_token_position": activation_sample.probe_token_position,
                    "extraction_time": activation_sample.extraction_time,
                    "activation_shapes": {k: list(v.shape) for k, v in activation_sample.activations.items()}
                }
                
                with open(sample_dir / f"{sample_id}_metadata.json", 'w') as f:
                    json.dump(sample_data, f, indent=2)
                
                # Save activations
                for layer_name, activation in activation_sample.activations.items():
                    torch.save(activation, sample_dir / f"{sample_id}_{layer_name}.pt")
        
        # Save experiment summary
        summary = {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "emotions": emotions,
            "samples_per_emotion": samples_per_emotion,
            "total_samples": len(all_samples),
            "test_layers": test_layers or "all",
            "output_dir": str(output_dir),
            "timestamp": timestamp
        }
        
        with open(output_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n🎉 Remote pipeline complete!")
        logger.info(f"✅ Generated {len(all_samples)} samples")
        logger.info(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Remote pipeline failed: {e}")
        raise

# =============================================================================
# REMOTE DEPLOYMENT TEST
# =============================================================================

def deploy_test_remote(experiment_name: str = "talktuner_test",
                      emotions: List[str] = None,
                      samples_per_emotion: int = 2,
                      min_vram: int = 12,
                      max_price: float = 0.40) -> Dict[str, str]:
    """Deploy the TalkTuner test to a remote GPU."""
    
    if emotions is None:
        emotions = ["frustrated", "calm", "control"]
    
    logger.info(f"🚀 Deploying TalkTuner test to remote GPU")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Emotions: {emotions}")
    
    # Deploy GPU (no need for exposed ports since we're not running vLLM server)
    gpu_client = GPUClient()
    query = (gpu_client.vram_gb >= min_vram) & (gpu_client.price_per_hour <= max_price)
    
    gpu_instance = gpu_client.create(
        query=query,
        name=f"talktuner-test-{experiment_name}",
        cloud_type="secure",
        sort=lambda x: x.price_per_hour,
        reverse=False
    )
    
    logger.info(f"✅ GPU ready: {gpu_instance.id}")
    
    # Wait for SSH
    if not gpu_instance.wait_until_ssh_ready(timeout=300):
        raise RuntimeError("SSH failed to become ready")
    
    ssh_connection = gpu_instance.ssh_connection_string()
    logger.info(f"✅ SSH ready: {ssh_connection}")
    
    # Deploy code
    bifrost_client = BifrostClient(ssh_connection)
    workspace_path = bifrost_client.push(uv_extra="examples_gsm8k_remote")  # Need nnsight, torch etc
    logger.info(f"✅ Code deployed: {workspace_path}")
    
    # Create test config
    test_config = {
        "experiment_name": experiment_name,
        "emotions": emotions,
        "samples_per_emotion": samples_per_emotion,
        "model_name": "willcb/Qwen3-0.6B",
        "test_layers": [0, 1, 2, 3, -4, -3, -2, -1]  # First few and last few layers
    }
    
    config_json = json.dumps(test_config, indent=2)
    config_path = f"~/talktuner_test_config.json"
    bifrost_client.exec(f"cat > {config_path} << 'EOF'\n{config_json}\nEOF")
    
    # Start test script in tmux
    log_path = f"~/talktuner_test.log"
    test_cmd = f"cd ~/.bifrost/workspace && uv run python examples/mats_neel/talktuner_emotions/test_pipeline.py --config {config_path} 2>&1 | tee {log_path}"
    tmux_cmd = f"tmux new-session -d -s talktuner-test '{test_cmd}'"
    
    bifrost_client.exec(tmux_cmd)
    logger.info(f"✅ Test started in tmux session 'talktuner-test'")
    
    connection_info = {
        "instance_id": gpu_instance.id,
        "ssh_connection": ssh_connection,
        "log_path": log_path,
        "config_path": config_path
    }
    
    logger.info(f"\n🎉 Remote test deployed!")
    logger.info(f"📊 Monitor progress:")
    logger.info(f"   bifrost exec '{ssh_connection}' 'tail -f {log_path}'")
    logger.info(f"📱 SSH into instance:")
    logger.info(f"   bifrost ssh '{ssh_connection}'")
    logger.info(f"🧹 Cleanup when done:")
    logger.info(f"   broker terminate {gpu_instance.id}")
    
    return connection_info

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TalkTuner emotions pipeline - remote only")
    parser.add_argument("--emotions", type=str, default="frustrated,calm,control",
                       help="Comma-separated emotions to test")
    parser.add_argument("--samples", type=int, default=2,
                       help="Samples per emotion")
    parser.add_argument("--config", type=str,
                       help="Config file for remote execution")
    parser.add_argument("--experiment-name", type=str, default="talktuner_test",
                       help="Experiment name for deployment")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    emotions = [e.strip() for e in args.emotions.split(",")]
    
    if args.config:
        # Running on remote machine with config
        run_remote_pipeline(args.config)
    else:
        # Deploy to remote
        deploy_test_remote(
            experiment_name=args.experiment_name,
            emotions=emotions,
            samples_per_emotion=args.samples
        )

if __name__ == "__main__":
    main()