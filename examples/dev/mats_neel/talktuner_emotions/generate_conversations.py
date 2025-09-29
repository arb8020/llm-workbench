#!/usr/bin/env python3
"""
TalkTuner Conversation Generation

Generates synthetic conversations using exact prompts from the TalkTuner paper.
Based on "Designing a Dashboard for Transparency and Control of Conversational AI"

Usage:
    python generate_conversations.py --demographic gender --samples 100 --output conversations.jsonl
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

from rollouts.dtypes import Endpoint

logger = logging.getLogger(__name__)

# =============================================================================
# EXACT PROMPTS FROM TALKTUNER PAPER (Appendix A)
# =============================================================================

SYSTEM_PROMPT = "You are a chatbot who will actively talk with a user and answer all the questions asked by the user."

# Gender prompts
GENDER_PROMPT = """Generate a conversation between a human user and an AI assistant. 
This human user is a {gender}. Make sure the conversation reflects this user's gender. 
Be creative on the topics of conversation."""

GENDER_VALUES = ["male", "female", "non-binary"]

# Age prompts  
AGE_PROMPT = """Generate a conversation between a human user and an AI assistant. 
This human user is a {age} who is {year_range}. Make sure the topic of the conversation 
or the way that user talks reflects this user's age."""

AGE_CATEGORIES = [
    {"age": "child", "year_range": "5-12 years old"},
    {"age": "teenager", "year_range": "13-19 years old"}, 
    {"age": "young adult", "year_range": "20-35 years old"},
    {"age": "middle-aged adult", "year_range": "36-55 years old"},
    {"age": "older adult", "year_range": "56+ years old"}
]

# Education prompts
EDUCATION_PROMPT = """Generate a conversation between a human user and an AI assistant. 
The education of this human user is {education}. Make sure the conversation directly or 
indirectly reflects this user's education level."""

EDUCATION_VALUES = [
    "elementary school",
    "high school", 
    "some college",
    "bachelor's degree",
    "master's degree",
    "doctoral degree"
]

# Socioeconomic status prompts
SOCIOECONOMIC_PROMPT = """Generate a conversation between a human user and an AI assistant. 
The socioeconomic status of this human user is {socioeco}. Make sure the conversation 
reflects this user's socioeconomic status."""

SOCIOECONOMIC_VALUES = [
    "low income",
    "lower middle class",
    "middle class", 
    "upper middle class",
    "high income"
]

# Emotion variants (extension based on basic_prompt_variation_gsm8k)
EMOTION_PROMPT = """Generate a conversation between a human user and an AI assistant.
The human user is feeling {emotion}. Make sure the conversation reflects this emotional state
in how the user communicates and what topics they discuss."""

EMOTION_VALUES = [
    "frustrated and impatient",
    "anxious and stressed", 
    "calm and patient",
    "collaborative and friendly",
    "excited and enthusiastic",
    "sad and melancholy",
    "angry and confrontational",
    "curious and inquisitive"
]

@dataclass
class ConversationSample:
    demographic_type: str
    demographic_value: str
    conversation_text: str
    sample_id: str
    generation_time: float
    metadata: Dict[str, Any]

# =============================================================================
# CONVERSATION GENERATION
# =============================================================================

async def generate_single_conversation(endpoint: Endpoint, demographic_type: str, 
                                     demographic_value: str, sample_id: str) -> ConversationSample:
    """Generate a single conversation using the TalkTuner methodology."""
    start_time = time.time()
    
    # Select appropriate prompt template
    if demographic_type == "gender":
        prompt = GENDER_PROMPT.format(gender=demographic_value)
    elif demographic_type == "age":
        # demographic_value should be a dict with age and year_range
        prompt = AGE_PROMPT.format(**demographic_value)
    elif demographic_type == "education":
        prompt = EDUCATION_PROMPT.format(education=demographic_value)
    elif demographic_type == "socioeconomic":
        prompt = SOCIOECONOMIC_PROMPT.format(socioeco=demographic_value)
    elif demographic_type == "emotion":
        prompt = EMOTION_PROMPT.format(emotion=demographic_value)
    else:
        raise ValueError(f"Unknown demographic type: {demographic_type}")
    
    logger.info(f"Generating {sample_id}: {demographic_type}={demographic_value}")
    
    try:
        from rollouts.dtypes import Message
        from rollouts.evaluation import make_trajectory
        
        # Create messages for conversation generation
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=prompt)
        ]
        
        # Generate conversation using rollouts framework
        trajectory = await make_trajectory(
            messages=messages,
            endpoint=endpoint,
            max_turns=1,  # Single response for conversation generation
            environment=None  # No tools needed
        )
        
        # Extract generated conversation
        assistant_messages = [m for m in trajectory.messages if m.role == "assistant"]
        if not assistant_messages:
            raise ValueError("No assistant response generated")
        
        conversation_text = assistant_messages[0].content
        
        # Create sample record
        sample = ConversationSample(
            demographic_type=demographic_type,
            demographic_value=str(demographic_value),
            conversation_text=conversation_text,
            sample_id=sample_id,
            generation_time=time.time() - start_time,
            metadata={
                "prompt_used": prompt,
                "system_prompt": SYSTEM_PROMPT,
                "endpoint_model": endpoint.model,
                "generation_timestamp": time.time()
            }
        )
        
        logger.info(f"✅ Generated {sample_id} in {sample.generation_time:.1f}s")
        return sample
        
    except Exception as e:
        logger.error(f"❌ Failed to generate {sample_id}: {e}")
        raise

def parse_conversation_format(conversation_text: str) -> List[Dict[str, str]]:
    """Parse generated conversation into User:/Assistant: format."""
    turns = []
    lines = conversation_text.strip().split('\n')
    
    current_speaker = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for speaker labels
        if line.startswith(('User:', 'Human:', 'user:', 'human:')):
            # Save previous turn if exists
            if current_speaker and current_content:
                turns.append({
                    "role": current_speaker,
                    "content": ' '.join(current_content)
                })
            current_speaker = "user"
            current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
            
        elif line.startswith(('Assistant:', 'AI:', 'assistant:', 'ai:', 'Chatbot:', 'chatbot:')):
            # Save previous turn if exists
            if current_speaker and current_content:
                turns.append({
                    "role": current_speaker, 
                    "content": ' '.join(current_content)
                })
            current_speaker = "assistant"
            current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
            
        else:
            # Continue current speaker's content
            if current_content:
                current_content.append(line)
    
    # Add final turn
    if current_speaker and current_content:
        turns.append({
            "role": current_speaker,
            "content": ' '.join(current_content)
        })
    
    return turns

# =============================================================================
# BATCH GENERATION
# =============================================================================

async def generate_conversations_batch(endpoint: Endpoint, demographic_type: str,
                                     samples_per_value: int = 100,
                                     output_path: Optional[Path] = None) -> List[ConversationSample]:
    """Generate a batch of conversations for a demographic type."""
    
    # Get values for this demographic type
    if demographic_type == "gender":
        values = GENDER_VALUES
    elif demographic_type == "age":
        values = AGE_CATEGORIES
    elif demographic_type == "education":
        values = EDUCATION_VALUES
    elif demographic_type == "socioeconomic":
        values = SOCIOECONOMIC_VALUES
    elif demographic_type == "emotion":
        values = EMOTION_VALUES
    else:
        raise ValueError(f"Unknown demographic type: {demographic_type}")
    
    logger.info(f"Generating {samples_per_value} samples per value for {demographic_type}")
    logger.info(f"Values: {values}")
    
    all_samples = []
    
    for value in values:
        logger.info(f"Processing {demographic_type}={value}")
        
        for i in range(samples_per_value):
            sample_id = f"{demographic_type}_{str(value).replace(' ', '_')}_{i+1:04d}"
            
            try:
                sample = await generate_single_conversation(
                    endpoint=endpoint,
                    demographic_type=demographic_type,
                    demographic_value=value,
                    sample_id=sample_id
                )
                all_samples.append(sample)
                
                # Save incrementally if output path provided
                if output_path:
                    with open(output_path, 'a') as f:
                        sample_dict = {
                            "sample_id": sample.sample_id,
                            "demographic_type": sample.demographic_type,
                            "demographic_value": sample.demographic_value,
                            "conversation_text": sample.conversation_text,
                            "conversation_turns": parse_conversation_format(sample.conversation_text),
                            "generation_time": sample.generation_time,
                            "metadata": sample.metadata
                        }
                        f.write(json.dumps(sample_dict) + '\n')
                
            except Exception as e:
                logger.error(f"Failed to generate {sample_id}: {e}")
                continue
    
    logger.info(f"✅ Generated {len(all_samples)} conversations for {demographic_type}")
    return all_samples

# =============================================================================
# MAIN GENERATION SCRIPT
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Generate TalkTuner conversations")
    
    parser.add_argument("--demographic", type=str, required=True,
                       choices=["gender", "age", "education", "socioeconomic", "emotion"],
                       help="Demographic type to generate conversations for")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples per demographic value")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                       help="Model to use for conversation generation")
    parser.add_argument("--endpoint-url", type=str,
                       help="Custom endpoint URL (for vLLM servers)")
    parser.add_argument("--api-key", type=str,
                       help="API key (uses OPENAI_API_KEY env var if not provided)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create endpoint
    if args.endpoint_url:
        # Custom vLLM endpoint
        endpoint = Endpoint(
            provider="openai",
            model=args.model,
            api_base=args.endpoint_url.rstrip('/') + '/v1',
            api_key=args.api_key or "dummy",
            max_tokens=1000,
            temperature=0.7
        )
    else:
        # OpenAI endpoint
        import os
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY env var or use --api-key")
        
        endpoint = Endpoint(
            provider="openai",
            model=args.model,
            api_key=api_key,
            max_tokens=1000,
            temperature=0.7
        )
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate conversations
    logger.info(f"Starting conversation generation for {args.demographic}")
    logger.info(f"Model: {endpoint.model}")
    logger.info(f"Output: {output_path}")
    
    start_time = time.time()
    samples = await generate_conversations_batch(
        endpoint=endpoint,
        demographic_type=args.demographic,
        samples_per_value=args.samples,
        output_path=output_path
    )
    
    total_time = time.time() - start_time
    logger.info(f"🎉 Generation complete!")
    logger.info(f"Generated {len(samples)} conversations in {total_time:.1f}s")
    logger.info(f"Saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())