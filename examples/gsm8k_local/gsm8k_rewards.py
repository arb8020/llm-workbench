#!/usr/bin/env python3
"""
GSM8K evaluation using the new reward function framework.

This demonstrates the simpler reward-based approach:
1. Reward functions take Trajectory -> float
2. Built-in parallelization support
3. RL-compatible design with numeric rewards
4. Supports both tool and no-tool modes

Usage:
    python examples/gsm8k_rewards.py --samples 3 --mode no-tools
    python examples/gsm8k_rewards.py --samples 3 --mode with-tools --parallel 2
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add rollouts to path
from rollouts.evaluation import evaluate, load_jsonl, RewardFunction
from rollouts.dtypes import Message, Endpoint, Environment, Trajectory, Tool, ToolFunction, ToolFunctionParameter, ToolCall, ToolResult, AgentState, RunConfig


def load_gsm8k_dataset(output_path: Path, sample_count: int = None) -> None:
    """Load GSM8K from HuggingFace and save as JSONL."""
    try:
        from datasets import load_dataset
        
        print("📚 Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("gsm8k", "main", split="test")
        
        if sample_count:
            dataset = dataset.select(range(min(sample_count, len(dataset))))
        
        print(f"📊 Selected {len(dataset)} samples from GSM8K")
        
        with open(output_path, 'w') as f:
            for i, row in enumerate(dataset):
                answer_text = row["answer"]
                if "####" in answer_text:
                    numeric_answer = answer_text.split("####")[-1].strip()
                else:
                    numeric_answer = "unknown"
                
                gsm8k_sample = {
                    "question": row["question"],
                    "answer": numeric_answer,
                    "sample_id": f"gsm8k_{i+1:04d}"
                }
                f.write(json.dumps(gsm8k_sample) + '\n')
        
        print(f"✅ Saved GSM8K dataset to: {output_path}")
        
    except ImportError:
        print("❌ HuggingFace datasets library not found. Install with: uv pip install datasets")
        raise
    except Exception as e:
        print(f"❌ Error loading GSM8K: {e}")
        raise


def prepare_gsm8k_messages_no_tools(sample: Dict[str, Any]) -> List[Message]:
    """Create initial messages for GSM8K problem (zero-shot)."""
    return [
        Message(
            role="system",
            content="""You are an expert at solving math word problems. Follow these instructions:

1. Read the problem carefully
2. Think through the solution step by step  
3. Show your reasoning clearly
4. Provide your final answer in this exact format: Answer: [number]

Important: Your final line must be "Answer: [your numeric answer]" with nothing else on that line."""
        ),
        Message(
            role="user",
            content=f"Solve the following math problem step by step:\n\n{sample['question']}"
        )
    ]


def prepare_gsm8k_messages_with_tools(sample: Dict[str, Any]) -> List[Message]:
    """Create initial messages for GSM8K problem (with calculator)."""
    return [
        Message(
            role="system",
            content="""You are an expert math tutor with access to a calculator tool. Follow these instructions:

1. Read the problem carefully
2. Break down the solution into steps
3. Use the calculator tool for any arithmetic operations
4. Show your reasoning clearly
5. Provide your final answer in this exact format: Answer: [number]

Important: Your final line must be "Answer: [your numeric answer]" with nothing else on that line."""
        ),
        Message(
            role="user",
            content=f"Solve the following math problem step by step:\n\n{sample['question']}"
        )
    ]


class NoToolsEnvironment(Environment):
    """Environment with no tools for zero-shot evaluation."""
    
    async def serialize(self) -> dict:
        return {"type": "NoToolsEnvironment"}
    
    @staticmethod
    async def deserialize(data: dict) -> 'NoToolsEnvironment':
        return NoToolsEnvironment()
    
    def get_tools(self) -> List[Tool]:
        return []


class CalculatorEnvironment(Environment):
    """Environment with calculator tool for tool-assisted evaluation."""
    
    async def serialize(self) -> dict:
        return {"type": "CalculatorEnvironment"}
    
    @staticmethod
    async def deserialize(data: dict) -> 'CalculatorEnvironment':
        return CalculatorEnvironment()
    
    def get_tools(self) -> List[Tool]:
        return [Tool(
            function=ToolFunction(
                name="calculate",
                description="Perform mathematical calculations",
                parameters=ToolFunctionParameter(
                    properties={
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    }
                ),
                required=["expression"]
            )
        )]
    
    async def exec_tool(self, tool_call: ToolCall, current_state: AgentState, 
                       run_config: RunConfig, checkpoint_store=None) -> ToolResult:
        if tool_call.name == "calculate":
            expression = tool_call.args.get("expression", "")
            try:
                result = eval(expression.replace("^", "**"))
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"{expression} = {result}"
                )
            except Exception as e:
                return ToolResult(
                    call_id=tool_call.id,
                    ok=False,
                    error=f"Calculation error: {e}"
                )
        
        return ToolResult(call_id=tool_call.id, ok=False, error="Unknown tool")


def extract_answer(response_text: str) -> str:
    """Extract answer using standardized format."""
    answer_pattern = r'Answer:\s*([^\n]+)'
    matches = re.findall(answer_pattern, response_text, re.IGNORECASE)
    
    if matches:
        answer = matches[-1].strip()
        answer = answer.replace('$', '').replace(',', '').strip()
        
        # Extract numeric part
        number_match = re.search(r'-?\d+(?:\.\d+)?', answer)
        if number_match:
            return number_match.group()
    
    return ""


def check_equality(predicted: str, expected: str) -> bool:
    """Check mathematical equivalence."""
    if not predicted or not expected:
        return False
    
    pred = str(predicted).strip().replace(',', '').replace('$', '')
    exp = str(expected).strip().replace(',', '').replace('$', '')
    
    if pred == exp:
        return True
    
    try:
        pred_num = float(pred)
        exp_num = float(exp)
        return abs(pred_num - exp_num) < 1e-6
    except ValueError:
        pass
    
    try:
        from fractions import Fraction
        return Fraction(pred) == Fraction(exp)
    except (ValueError, ZeroDivisionError):
        pass
    
    return False


def make_correctness_reward(sample: Dict[str, Any]) -> RewardFunction:
    """Create a reward function that checks correctness for this sample."""
    def check_correctness(trajectory: Trajectory) -> float:
        # Get final response
        assistant_messages = [m for m in trajectory.messages if m.role == "assistant"]
        if not assistant_messages:
            return 0.0
        
        response = " ".join(m.content for m in assistant_messages if m.content)
        
        # Extract and check answer
        extracted_answer = extract_answer(response)
        expected_answer = str(sample["answer"]).strip()
        
        is_correct = check_equality(extracted_answer, expected_answer) if extracted_answer else False
        return 1.0 if is_correct else 0.0
    
    return check_correctness


def format_reward(trajectory: Trajectory) -> float:
    """Reward for following the answer format."""
    assistant_messages = [m for m in trajectory.messages if m.role == "assistant"]
    if not assistant_messages:
        return 0.0
    
    response = " ".join(m.content for m in assistant_messages if m.content)
    has_answer_format = bool(re.search(r'Answer:\s*[^\n]+', response, re.IGNORECASE))
    return 1.0 if has_answer_format else 0.0


def efficiency_reward(trajectory: Trajectory) -> float:
    """Reward for being concise (fewer tokens)."""
    total_tokens = sum(len(m.content or "") for m in trajectory.messages)
    # Normalize: 1.0 for <500 tokens, 0.0 for >2000 tokens
    if total_tokens < 500:
        return 1.0
    elif total_tokens > 2000:
        return 0.0
    else:
        return 1.0 - (total_tokens - 500) / 1500


def tool_usage_reward(trajectory: Trajectory) -> float:
    """Reward for appropriate tool usage (only meaningful with calculator environment)."""
    tool_calls = sum(len(m.tool_calls or []) for m in trajectory.messages)
    # Reward 1.0 for 1-3 tool calls, 0.5 for 4-6, 0.0 for 7+
    if tool_calls == 0:
        return 0.0  # Should use calculator for math problems
    elif 1 <= tool_calls <= 3:
        return 1.0
    elif 4 <= tool_calls <= 6:
        return 0.5
    else:
        return 0.0  # Too many tool calls


async def main(samples: int = 3, mode: str = "no-tools", parallel: int = 1):
    """Run GSM8K evaluation using new reward framework."""
    from datetime import datetime
    
    # Create timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gsm8k_{mode.replace('-', '_')}_nsamples_{samples}_{timestamp}"
    
    print(f"🎯 GSM8K Evaluation - {mode.upper()} Mode")
    print(f"📅 Experiment: {experiment_name}")
    print("=" * 50)
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Create dataset
    assets_dir = Path("examples/gsm8k_local/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = assets_dir / f"gsm8k_{mode.replace('-', '_')}.jsonl"
    
    try:
        load_gsm8k_dataset(dataset_path, samples)
    except Exception as e:
        print(f"❌ Failed to load GSM8K: {e}")
        return
    
    # Configure endpoint
    endpoint = Endpoint(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000,
        temperature=0.1
    )
    
    # Choose environment and settings based on mode
    if mode == "no-tools":
        environment = NoToolsEnvironment()
        prepare_messages = prepare_gsm8k_messages_no_tools
        max_turns = 1
        print("📝 Mode: Zero-shot chain-of-thought")
        
        # Reward functions for no-tools mode
        reward_functions = [
            ("correctness", lambda t: 0.0),  # Will be replaced per sample
            ("format", format_reward),
            ("efficiency", efficiency_reward),
        ]
    else:
        environment = CalculatorEnvironment()
        prepare_messages = prepare_gsm8k_messages_with_tools
        max_turns = 6
        print("🔧 Mode: Tool-assisted reasoning")
        
        # Reward functions for with-tools mode
        reward_functions = [
            ("correctness", lambda t: 0.0),  # Will be replaced per sample
            ("format", format_reward),
            ("efficiency", efficiency_reward),
            ("tool_usage", tool_usage_reward),
        ]
    
    # Load dataset to create sample-specific reward functions
    dataset_samples = list(load_jsonl(dataset_path))
    
    # Create sample-specific reward functions
    def create_reward_functions_for_sample(sample: Dict[str, Any]):
        correctness_fn = make_correctness_reward(sample)
        sample_rewards = [
            ("correctness", correctness_fn),
            ("format", format_reward),
            ("efficiency", efficiency_reward),
        ]
        if mode == "with-tools":
            sample_rewards.append(("tool_usage", tool_usage_reward))
        return sample_rewards
    
    # For now, we'll use the first sample's reward function as a demo
    # (In practice, you'd want to pass sample-specific rewards to evaluate_sample)
    demo_rewards = create_reward_functions_for_sample(dataset_samples[0])
    
    # Run evaluation using new framework
    from rollouts.agents import stdout_handler
    run_config = RunConfig(on_chunk=stdout_handler)
    
    report = await evaluate(
        dataset=iter(dataset_samples),
        prepare_messages=prepare_messages,
        reward_functions=demo_rewards,
        environment=environment,
        endpoint=endpoint,
        run_config=run_config,
        eval_name=experiment_name,
        dataset_path=str(dataset_path),
        max_turns=max_turns,
        max_samples=samples,
        max_concurrent=parallel,
        output_dir=Path(f"examples/gsm8k_local/results/{experiment_name}"),
        sample_id_fn=lambda i, sample: sample.get("sample_id", f"gsm8k_{i:04d}"),
        verbose=True
    )
    
    # Print detailed results
    print(f"\n🔍 Detailed Results:")
    for sample in report.sample_results:
        status = "✅" if sample.metrics.get("correctness", 0.0) > 0.5 else "❌"
        correctness = sample.metrics.get("correctness", 0.0)
        format_score = sample.metrics.get("format", 0.0)
        print(f"   {status} {sample.sample_id}: Correct={correctness:.1f}, Format={format_score:.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GSM8K Evaluation with Reward Functions")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to evaluate")
    parser.add_argument("--mode", choices=["no-tools", "with-tools"], default="no-tools", 
                       help="Evaluation mode")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel evaluations")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        samples=args.samples,
        mode=args.mode,
        parallel=args.parallel
    ))
