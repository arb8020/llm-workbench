#!/usr/bin/env python3
"""
Conversation Analysis for Tau-Bench User Variation Experiments

Analyzes conversation patterns, user behaviors, and agent responses
across different emotional user states.

Usage:
    python analyze_conversations.py --results-dir results/my_experiment_20250912_135102/
    python analyze_conversations.py --results-dir results/ --variant frustration --output frustration_analysis.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import sys

from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    speaker: str  # 'user' or 'agent' 
    message: str
    turn_number: int
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class ConversationAnalysis:
    """Analysis of a single conversation."""
    task_id: int
    variant: str
    environment: str
    
    # Basic metrics
    total_turns: int
    user_turns: int
    agent_turns: int
    conversation_length: int  # characters
    
    # Emotional indicators
    emotional_indicators: Dict[str, int]  # keyword -> count
    sentiment_scores: Optional[Dict[str, float]] = None
    
    # Interaction patterns
    user_interruptions: int
    agent_clarifications: int
    user_repetitions: int
    
    # Task completion
    task_completed: bool
    completion_turns: int
    final_reward: float
    
    # Conversation flow
    turns: List[ConversationTurn]

@dataclass
class VariantConversationAnalysis:
    """Aggregated analysis for all conversations of a variant."""
    variant: str
    total_conversations: int
    
    # Aggregated metrics
    avg_conversation_length: float
    avg_turns: float
    avg_user_turns: float
    avg_agent_turns: float
    
    # Emotional patterns
    common_emotional_indicators: List[Tuple[str, int]]  # (indicator, frequency)
    emotional_intensity: float  # 0-1 scale
    
    # Interaction patterns  
    avg_user_interruptions: float
    avg_agent_clarifications: float
    avg_user_repetitions: float
    
    # Success metrics
    completion_rate: float
    avg_completion_turns: float
    avg_final_reward: float
    
    # Sample conversations (best and worst performing)
    sample_conversations: List[ConversationAnalysis]

# Emotional indicator patterns for different user types
EMOTIONAL_PATTERNS = {
    'frustration': {
        'keywords': ['ridiculous', 'waiting forever', 'complicated', 'impatient', 'annoying', 
                    'waste of time', 'should not', 'terrible', 'awful', 'worst'],
        'phrases': ['this is ridiculous', "i've been waiting", 'why is this so', 'last time',
                   'had enough', 'fed up', 'done with this'],
        'punctuation': ['!!', '!!!', '????', '???']
    },
    'anxiety': {
        'keywords': ['worried', 'nervous', 'anxious', 'scared', 'uncertain', 'wrong', 
                    'mistake', 'sure', 'correct', 'safe'],
        'phrases': ['am i doing', 'are you sure', 'what if', 'i hope', "i don't want to",
                   'is this right', 'will this work', 'i worry'],
        'punctuation': ['?', '??', '...']
    },
    'anger': {
        'keywords': ['unacceptable', 'outrageous', 'furious', 'angry', 'mad', 'demand',
                    'manager', 'supervisor', 'complaint', 'lawsuit'],
        'phrases': ['this is unacceptable', 'i demand', 'speak to', 'file a complaint',
                   'completely ridiculous', 'how dare', 'i want compensation'],
        'punctuation': ['!', '!!', '!!!', 'CAPS']
    },
    'confusion': {
        'keywords': ['confused', 'understand', 'explain', 'what', 'how', 'unclear',
                    'complicated', 'difficult', 'lost', 'overwhelmed'],
        'phrases': ["i don't understand", 'can you explain', 'what does', 'how do i',
                   'this is confusing', 'too complicated', 'simpler way'],
        'punctuation': ['?', '??', '???']
    }
}

def parse_conversation_log(log_content: str) -> List[ConversationTurn]:
    """Parse conversation log into structured turns."""
    
    turns = []
    lines = log_content.split('\n')
    
    turn_number = 0
    current_speaker = None
    current_message = ""
    
    # Common log formats to handle
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to identify speaker patterns
        speaker_match = None
        
        # Pattern 1: "User: message" or "Agent: message"
        if line.startswith(('User:', 'Agent:', 'user:', 'agent:')):
            speaker_match = re.match(r'(User|Agent|user|agent):\s*(.*)', line)
        
        # Pattern 2: "[USER]" or "[AGENT]" format
        elif line.startswith(('[USER]', '[AGENT]', '[user]', '[agent]')):
            speaker_match = re.match(r'\[(USER|AGENT|user|agent)\]\s*(.*)', line)
        
        # Pattern 3: Timestamps with speaker info
        elif re.search(r'(user|agent|User|Agent)', line):
            timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2}).*?(user|agent|User|Agent).*?:\s*(.*)', line, re.IGNORECASE)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                speaker = timestamp_match.group(2).lower()
                message = timestamp_match.group(3)
                speaker_match = type('Match', (), {'group': lambda x: speaker if x == 1 else message})()
        
        if speaker_match:
            # Save previous turn if exists
            if current_speaker and current_message.strip():
                turn = ConversationTurn(
                    speaker=current_speaker,
                    message=current_message.strip(),
                    turn_number=turn_number
                )
                turns.append(turn)
                turn_number += 1
            
            # Start new turn
            current_speaker = speaker_match.group(1).lower()
            current_message = speaker_match.group(2) if hasattr(speaker_match, 'group') else ""
        else:
            # Continuation of current message
            if current_message:
                current_message += " " + line
            else:
                current_message = line
    
    # Add final turn
    if current_speaker and current_message.strip():
        turn = ConversationTurn(
            speaker=current_speaker,
            message=current_message.strip(),
            turn_number=turn_number
        )
        turns.append(turn)
    
    return turns

def count_emotional_indicators(text: str, variant: str) -> Dict[str, int]:
    """Count emotional indicators in text based on variant type."""
    
    indicators = defaultdict(int)
    text_lower = text.lower()
    
    # Get patterns for this variant type
    patterns = EMOTIONAL_PATTERNS.get(variant, {})
    
    # Count keywords
    for keyword in patterns.get('keywords', []):
        count = text_lower.count(keyword)
        if count > 0:
            indicators[keyword] = count
    
    # Count phrases
    for phrase in patterns.get('phrases', []):
        count = text_lower.count(phrase)
        if count > 0:
            indicators[phrase] = count
    
    # Count punctuation patterns
    for punct in patterns.get('punctuation', []):
        if punct == 'CAPS':
            # Count words in all caps
            caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
            if caps_words:
                indicators['ALL_CAPS'] = len(caps_words)
        else:
            count = text.count(punct)
            if count > 0:
                indicators[punct] = count
    
    return dict(indicators)

def analyze_conversation_patterns(turns: List[ConversationTurn], variant: str) -> Dict[str, Any]:
    """Analyze interaction patterns in conversation."""
    
    patterns = {
        'user_interruptions': 0,
        'agent_clarifications': 0,
        'user_repetitions': 0,
        'question_count': 0,
        'exclamation_count': 0
    }
    
    user_messages = [turn.message for turn in turns if turn.speaker == 'user']
    agent_messages = [turn.message for turn in turns if turn.speaker == 'agent']
    
    # Count questions and exclamations
    all_text = ' '.join([turn.message for turn in turns])
    patterns['question_count'] = all_text.count('?')
    patterns['exclamation_count'] = all_text.count('!')
    
    # Look for interruption patterns (consecutive user turns)
    prev_speaker = None
    for turn in turns:
        if turn.speaker == 'user' and prev_speaker == 'user':
            patterns['user_interruptions'] += 1
        prev_speaker = turn.speaker
    
    # Look for agent clarification patterns
    clarification_phrases = ['let me clarify', 'to clarify', 'what i mean', 'in other words', 
                           'let me explain', 'i understand', 'i see', 'let me help']
    
    for message in agent_messages:
        message_lower = message.lower()
        for phrase in clarification_phrases:
            if phrase in message_lower:
                patterns['agent_clarifications'] += 1
                break
    
    # Look for user repetition patterns (similar messages)
    if len(user_messages) > 1:
        for i, msg1 in enumerate(user_messages):
            for msg2 in user_messages[i+1:]:
                # Simple similarity check
                words1 = set(msg1.lower().split())
                words2 = set(msg2.lower().split())
                if words1 and words2:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.5:  # 50% word overlap
                        patterns['user_repetitions'] += 1
    
    return patterns

def analyze_single_conversation(conversation_data: Dict[str, Any], 
                               conversation_log: str) -> ConversationAnalysis:
    """Analyze a single conversation."""
    
    task_id = conversation_data.get('task_id', 0)
    variant = conversation_data.get('variant', 'unknown')
    environment = conversation_data.get('environment', 'unknown')
    final_reward = conversation_data.get('reward', 0.0)
    task_completed = conversation_data.get('success', False)
    
    # Parse conversation turns
    turns = parse_conversation_log(conversation_log)
    
    if not turns:
        logger.warning(f"No conversation turns found for task {task_id}")
        return ConversationAnalysis(
            task_id=task_id,
            variant=variant,
            environment=environment,
            total_turns=0,
            user_turns=0,
            agent_turns=0,
            conversation_length=0,
            emotional_indicators={},
            user_interruptions=0,
            agent_clarifications=0,
            user_repetitions=0,
            task_completed=task_completed,
            completion_turns=0,
            final_reward=final_reward,
            turns=[]
        )
    
    # Basic metrics
    user_turns = [t for t in turns if t.speaker == 'user']
    agent_turns = [t for t in turns if t.speaker == 'agent']
    
    total_turns = len(turns)
    conversation_length = sum(len(turn.message) for turn in turns)
    completion_turns = total_turns if task_completed else 0
    
    # Emotional analysis
    user_text = ' '.join(turn.message for turn in user_turns)
    emotional_indicators = count_emotional_indicators(user_text, variant)
    
    # Pattern analysis
    patterns = analyze_conversation_patterns(turns, variant)
    
    return ConversationAnalysis(
        task_id=task_id,
        variant=variant,
        environment=environment,
        total_turns=total_turns,
        user_turns=len(user_turns),
        agent_turns=len(agent_turns),
        conversation_length=conversation_length,
        emotional_indicators=emotional_indicators,
        user_interruptions=patterns['user_interruptions'],
        agent_clarifications=patterns['agent_clarifications'],
        user_repetitions=patterns['user_repetitions'],
        task_completed=task_completed,
        completion_turns=completion_turns,
        final_reward=final_reward,
        turns=turns
    )

def aggregate_variant_analysis(conversations: List[ConversationAnalysis], variant: str) -> VariantConversationAnalysis:
    """Aggregate analysis across all conversations for a variant."""
    
    if not conversations:
        return VariantConversationAnalysis(
            variant=variant,
            total_conversations=0,
            avg_conversation_length=0,
            avg_turns=0,
            avg_user_turns=0,
            avg_agent_turns=0,
            common_emotional_indicators=[],
            emotional_intensity=0,
            avg_user_interruptions=0,
            avg_agent_clarifications=0,
            avg_user_repetitions=0,
            completion_rate=0,
            avg_completion_turns=0,
            avg_final_reward=0,
            sample_conversations=[]
        )
    
    total_conversations = len(conversations)
    
    # Basic metrics
    avg_conversation_length = sum(c.conversation_length for c in conversations) / total_conversations
    avg_turns = sum(c.total_turns for c in conversations) / total_conversations
    avg_user_turns = sum(c.user_turns for c in conversations) / total_conversations
    avg_agent_turns = sum(c.agent_turns for c in conversations) / total_conversations
    
    # Emotional analysis
    all_indicators = Counter()
    for conv in conversations:
        for indicator, count in conv.emotional_indicators.items():
            all_indicators[indicator] += count
    
    common_emotional_indicators = all_indicators.most_common(10)
    
    # Emotional intensity (normalized by conversation length)
    total_emotional_words = sum(all_indicators.values())
    total_words = sum(c.conversation_length for c in conversations) / 5  # Rough word estimate
    emotional_intensity = total_emotional_words / max(total_words, 1)
    
    # Interaction patterns
    avg_user_interruptions = sum(c.user_interruptions for c in conversations) / total_conversations
    avg_agent_clarifications = sum(c.agent_clarifications for c in conversations) / total_conversations
    avg_user_repetitions = sum(c.user_repetitions for c in conversations) / total_conversations
    
    # Success metrics
    completed_conversations = [c for c in conversations if c.task_completed]
    completion_rate = len(completed_conversations) / total_conversations
    avg_completion_turns = sum(c.completion_turns for c in completed_conversations) / max(len(completed_conversations), 1)
    avg_final_reward = sum(c.final_reward for c in conversations) / total_conversations
    
    # Sample conversations (best and worst)
    sorted_by_reward = sorted(conversations, key=lambda x: x.final_reward, reverse=True)
    sample_conversations = sorted_by_reward[:2] + sorted_by_reward[-2:]  # Top 2 + bottom 2
    
    return VariantConversationAnalysis(
        variant=variant,
        total_conversations=total_conversations,
        avg_conversation_length=avg_conversation_length,
        avg_turns=avg_turns,
        avg_user_turns=avg_user_turns,
        avg_agent_turns=avg_agent_turns,
        common_emotional_indicators=common_emotional_indicators,
        emotional_intensity=emotional_intensity,
        avg_user_interruptions=avg_user_interruptions,
        avg_agent_clarifications=avg_agent_clarifications,
        avg_user_repetitions=avg_user_repetitions,
        completion_rate=completion_rate,
        avg_completion_turns=avg_completion_turns,
        avg_final_reward=avg_final_reward,
        sample_conversations=sample_conversations[:4]  # Limit to 4 samples
    )

def find_conversation_logs(results_dir: Path) -> Dict[str, List[Path]]:
    """Find all conversation log files organized by variant."""
    
    conversation_files = {}
    
    # Look for log files in various formats
    log_patterns = ['*.log', '*.txt', '*.json', '*conversation*', '*chat*', '*dialogue*']
    
    for pattern in log_patterns:
        for log_file in results_dir.rglob(pattern):
            # Try to determine variant from path
            path_parts = log_file.parts
            variant = 'unknown'
            
            # Look for variant name in path
            for part in path_parts:
                if part in ['control', 'frustration', 'anxiety', 'anger', 'confusion']:
                    variant = part
                    break
            
            if variant not in conversation_files:
                conversation_files[variant] = []
            conversation_files[variant].append(log_file)
    
    return conversation_files

def analyze_conversations_in_directory(results_dir: Path, target_variant: Optional[str] = None) -> Dict[str, VariantConversationAnalysis]:
    """Analyze all conversations in results directory."""
    
    logger.info(f"Analyzing conversations in: {results_dir}")
    
    # Load aggregated results to get task metadata
    try:
        aggregated_file = results_dir / "aggregated_results.json"
        with open(aggregated_file, 'r') as f:
            aggregated_data = json.load(f)
        
        task_results = {r['task_id']: r for r in aggregated_data['task_results']}
        variants = aggregated_data['variants']
    except Exception as e:
        logger.warning(f"Could not load aggregated results: {e}")
        task_results = {}
        variants = ['control', 'frustration', 'anxiety', 'anger', 'confusion']
    
    # Find conversation log files
    conversation_files = find_conversation_logs(results_dir)
    
    if not conversation_files:
        logger.warning("No conversation log files found")
        return {}
    
    # Analyze conversations for each variant
    variant_analyses = {}
    
    for variant in variants:
        if target_variant and variant != target_variant:
            continue
        
        logger.info(f"Analyzing conversations for variant: {variant}")
        
        variant_conversations = []
        log_files = conversation_files.get(variant, [])
        
        for log_file in log_files:
            try:
                # Read conversation log
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                
                if not log_content.strip():
                    continue
                
                # Try to extract task ID from filename or content
                task_id = 0
                filename = log_file.name
                task_match = re.search(r'task[\-_]?(\d+)', filename, re.IGNORECASE)
                if task_match:
                    task_id = int(task_match.group(1))
                
                # Get task metadata
                task_data = task_results.get(task_id, {
                    'task_id': task_id,
                    'variant': variant,
                    'environment': 'unknown',
                    'reward': 0.0,
                    'success': False
                })
                
                # Analyze conversation
                conv_analysis = analyze_single_conversation(task_data, log_content)
                variant_conversations.append(conv_analysis)
                
            except Exception as e:
                logger.warning(f"Failed to analyze conversation log {log_file}: {e}")
                continue
        
        if variant_conversations:
            # Aggregate analysis for this variant
            variant_analysis = aggregate_variant_analysis(variant_conversations, variant)
            variant_analyses[variant] = variant_analysis
            
            logger.info(f"✅ Analyzed {len(variant_conversations)} conversations for {variant}")
        else:
            logger.warning(f"No valid conversations found for variant: {variant}")
    
    return variant_analyses

def create_conversation_report(analyses: Dict[str, VariantConversationAnalysis]) -> str:
    """Create comprehensive conversation analysis report."""
    
    report = []
    report.append("=" * 80)
    report.append("TAU-BENCH CONVERSATION ANALYSIS REPORT")
    report.append("=" * 80)
    
    if not analyses:
        report.append("❌ No conversation data found for analysis.")
        report.append("Ensure that conversation logs are present in the results directory.")
        return "\n".join(report)
    
    # Summary
    total_conversations = sum(analysis.total_conversations for analysis in analyses.values())
    report.append(f"Total Conversations Analyzed: {total_conversations}")
    report.append(f"Variants: {', '.join(analyses.keys())}")
    report.append("")
    
    # Per-variant analysis
    for variant, analysis in analyses.items():
        report.append(f"🗣️ {variant.upper()} CONVERSATIONS")
        report.append("-" * 40)
        report.append(f"Total Conversations: {analysis.total_conversations}")
        report.append(f"Average Turns: {analysis.avg_turns:.1f}")
        report.append(f"Average User Turns: {analysis.avg_user_turns:.1f}")
        report.append(f"Average Agent Turns: {analysis.avg_agent_turns:.1f}")
        report.append(f"Average Length: {analysis.avg_conversation_length:.0f} characters")
        report.append(f"Completion Rate: {analysis.completion_rate:.1%}")
        report.append(f"Average Final Reward: {analysis.avg_final_reward:.3f}")
        report.append("")
        
        # Emotional indicators
        if analysis.common_emotional_indicators:
            report.append(f"🎭 Emotional Indicators:")
            for indicator, count in analysis.common_emotional_indicators[:5]:
                report.append(f"  '{indicator}': {count} times")
            report.append(f"  Emotional Intensity: {analysis.emotional_intensity:.3f}")
        report.append("")
        
        # Interaction patterns
        report.append(f"💬 Interaction Patterns:")
        report.append(f"  User Interruptions: {analysis.avg_user_interruptions:.1f}")
        report.append(f"  Agent Clarifications: {analysis.avg_agent_clarifications:.1f}")
        report.append(f"  User Repetitions: {analysis.avg_user_repetitions:.1f}")
        report.append("")
        
        # Sample conversations
        if analysis.sample_conversations:
            report.append(f"📝 Sample Conversation Excerpts:")
            for i, conv in enumerate(analysis.sample_conversations[:2], 1):
                report.append(f"  Example {i} (Task {conv.task_id}, Reward: {conv.final_reward:.3f}):")
                
                # Show first few turns
                for turn in conv.turns[:4]:
                    speaker = "👤 User" if turn.speaker == 'user' else "🤖 Agent"
                    message = turn.message[:100] + "..." if len(turn.message) > 100 else turn.message
                    report.append(f"    {speaker}: {message}")
                
                if len(conv.turns) > 4:
                    report.append(f"    ... ({len(conv.turns) - 4} more turns)")
                report.append("")
        
        report.append("")
    
    # Comparative insights
    if len(analyses) > 1:
        report.append("🔍 COMPARATIVE INSIGHTS")
        report.append("-" * 40)
        
        # Find most and least talkative variants
        by_turns = sorted(analyses.items(), key=lambda x: x[1].avg_turns)
        report.append(f"Most verbose: {by_turns[-1][0].upper()} ({by_turns[-1][1].avg_turns:.1f} turns)")
        report.append(f"Most concise: {by_turns[0][0].upper()} ({by_turns[0][1].avg_turns:.1f} turns)")
        
        # Find most and least successful variants
        by_completion = sorted(analyses.items(), key=lambda x: x[1].completion_rate)
        report.append(f"Highest completion rate: {by_completion[-1][0].upper()} ({by_completion[-1][1].completion_rate:.1%})")
        report.append(f"Lowest completion rate: {by_completion[0][0].upper()} ({by_completion[0][1].completion_rate:.1%})")
        
        # Find most emotionally intense
        by_intensity = sorted(analyses.items(), key=lambda x: x[1].emotional_intensity)
        if by_intensity[-1][1].emotional_intensity > 0:
            report.append(f"Most emotional: {by_intensity[-1][0].upper()} ({by_intensity[-1][1].emotional_intensity:.3f} intensity)")
        
    report.append("=" * 80)
    return "\n".join(report)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze tau-bench conversation logs")
    
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Path to experiment results directory")
    parser.add_argument("--variant", type=str,
                       help="Analyze only specific variant")
    parser.add_argument("--output", type=Path,
                       help="Output file for analysis report")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    try:
        # Analyze conversations
        analyses = analyze_conversations_in_directory(args.results_dir, args.variant)
        
        if not analyses:
            logger.error("No conversation data found for analysis")
            sys.exit(1)
        
        # Create report
        report = create_conversation_report(analyses)
        
        # Print to console
        print(report)
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            
            # Also save JSON data
            json_output = args.output.with_suffix('.json')
            with open(json_output, 'w') as f:
                # Convert to JSON-serializable format
                json_data = {
                    variant: asdict(analysis) 
                    for variant, analysis in analyses.items()
                }
                json.dump(json_data, f, indent=2)
            
            logger.info(f"📄 Conversation analysis saved: {args.output}")
            logger.info(f"📊 JSON data saved: {json_output}")
        
        logger.info("🎉 Conversation analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversation analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()