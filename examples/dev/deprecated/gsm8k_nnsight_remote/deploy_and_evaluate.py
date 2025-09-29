#!/usr/bin/env python3
"""
GSM8K evaluation with activation collection using remote nnsight server.

This combines GSM8K evaluation with residual stream activation collection for probe training.
Uses nnsight instead of vLLM to collect model activations during inference.

Usage:
    python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py --samples 3 --mode no-tools
    python examples/gsm8k_nnsight_remote/deploy_and_evaluate.py --samples 3 --mode with-tools --collect-activations
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from datetime import datetime
import numpy as np

from shared.logging_config import setup_logging

# Import broker and bifrost for deployment
from broker.client import GPUClient
from bifrost.client import BifrostClient

# Import rollouts evaluation framework
from rollouts.evaluation import evaluate, load_jsonl, RewardFunction
from rollouts.dtypes import Message, Endpoint, Environment, Trajectory, Tool, ToolFunction, ToolFunctionParameter, ToolCall, ToolResult, AgentState, RunConfig

logger = logging.getLogger(__name__)

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
                problem = {
                    'id': f'gsm8k_{i+1:04d}',
                    'query': row['question'],
                    'answer': row['answer']
                }
                f.write(json.dumps(problem) + '\n')
                
        print(f"💾 Dataset saved to {output_path}")
        
    except ImportError:
        print("❌ datasets library not found. Install with: pip install datasets")
        sys.exit(1)

def extract_numerical_answer(answer_text: str) -> Optional[float]:
    """Extract numerical answer from GSM8K answer text."""
    # Look for pattern like "#### 42"
    match = re.search(r'####\s*([0-9,.-]+)', answer_text)
    if match:
        try:
            # Remove commas and convert to float
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Fallback: look for last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text.replace(',', ''))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

class GSMRewardFunction(RewardFunction):
    """Reward function for GSM8K problems."""
    
    def __init__(self):
        super().__init__()
    
    async def __call__(self, trajectory: Trajectory, problem: Dict[str, Any]) -> float:
        """Calculate reward based on whether the answer is correct."""
        if not trajectory.messages:
            return 0.0
        
        # Get the last assistant message
        assistant_messages = [msg for msg in trajectory.messages if msg.role == "assistant"]
        if not assistant_messages:
            return 0.0
        
        final_response = assistant_messages[-1].content
        
        # Extract numerical answer from model response
        predicted_answer = extract_numerical_answer(final_response)
        
        # Extract correct answer
        correct_answer = extract_numerical_answer(problem['answer'])
        
        if predicted_answer is None or correct_answer is None:
            return 0.0
        
        # Check if answers match (with small tolerance for floating point)
        return 1.0 if abs(predicted_answer - correct_answer) < 1e-6 else 0.0

async def deploy_nnsight_server(gpu_client: GPUClient, bifrost_client: BifrostClient, gpu_id: str, 
                               args: argparse.Namespace) -> str:
    """Deploy nnsight server with activation collection to remote GPU."""
    
    logger.info(f"🚀 Deploying nnsight server to GPU {gpu_id}")
    
    # Deploy code to GPU
    logger.info("📦 Deploying code via bifrost...")
    workspace_path = bifrost_client.push(uv_extra="examples_gsm8k_remote")
    
    if not workspace_path:
        raise RuntimeError("Failed to deploy code")
    
    logger.info(f"✅ Code deployed successfully to: {workspace_path}")
    
    # Install dependencies
    logger.info("📦 Installing nnsight server dependencies...")
    install_cmd = (
        "pip install fastapi uvicorn nnsight transformers accelerate torch"
    )
    
    result = bifrost_client.exec(install_cmd)
    if result.exit_code != 0:
        logger.warning(f"Dependency installation had issues: {result.output}")
    
    # Start nnsight server
    logger.info("🚀 Starting nnsight server...")
    start_cmd = (
        f"cd ~/.bifrost/workspace && "
        f"python examples/gsm8k_nnsight_remote/nnsight_server.py "
        f"--port 8001 --model willcb/Qwen3-0.6B"
    )
    
    # Start in background
    result = bifrost_client.exec(f"nohup {start_cmd} > nnsight_server.log 2>&1 &")
    
    # Wait for server to be ready
    logger.info("⏳ Waiting for nnsight server to start...")
    max_wait = 120
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            # Test health endpoint
            test_cmd = "curl -f http://localhost:8001/health"
            result = bifrost_client.exec(test_cmd)
            
            if result.exit_code == 0:
                logger.info("✅ nnsight server is ready!")
                break
        except:
            pass
        
        await asyncio.sleep(5)
        wait_time += 5
    else:
        # Show server log for debugging
        log_cmd = "tail -50 nnsight_server.log"
        result = bifrost_client.exec(log_cmd)
        logger.error(f"Server log:\n{result.output}")
        raise RuntimeError(f"nnsight server failed to start within {max_wait}s")
    
    # Get public IP for API access
    ip_result = bifrost_client.exec("curl -s ifconfig.me")
    if ip_result.exit_code == 0:
        server_ip = ip_result.output.strip()
        server_url = f"http://{server_ip}:8001"
        logger.info(f"🌐 nnsight server available at: {server_url}")
        return server_url
    else:
        raise RuntimeError("Failed to get server IP address")

async def create_nnsight_endpoint(server_url: str, collect_activations: bool = False, transfer_activations: bool = False) -> Endpoint:
    """Create endpoint configuration for nnsight server."""
    
    # Prepare extra_params for activation collection
    extra_params = None
    if collect_activations:
        extra_params = {
            "collect_activations": {
                "layers": [8, 12, 16],  # Default layers
                "hook_points": ["input_layernorm.output", "post_attention_layernorm.output"],
                "return_activations": transfer_activations  # True only if explicitly requested
            }
        }
    
    # Base endpoint configuration (follows gsm8k_remote pattern)
    endpoint = Endpoint(
        provider="openai",  # nnsight server exposes OpenAI-compatible API
        model="willcb/Qwen3-0.6B",
        api_base=server_url + "/v1",  # nnsight server URL with /v1 suffix
        api_key="dummy",  # nnsight server doesn't need real API key
        max_tokens=500,
        temperature=0.1
    )
    
    # Manually add extra_params attribute since it's not in the dataclass definition
    # but rollouts/agents.py checks for it with hasattr()
    # Use object.__setattr__ to bypass frozen dataclass restrictions
    if extra_params:
        object.__setattr__(endpoint, 'extra_params', extra_params)
    
    return endpoint

def get_calculator_tool() -> Tool:
    """Get calculator tool for GSM8K problems."""
    return Tool(
        function=ToolFunction(
            name="calculator",
            description="Perform mathematical calculations. Use this for any arithmetic operations.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": ToolFunctionParameter(
                        type="string",
                        description="Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'abs(-5)')"
                    )
                },
                "required": ["expression"]
            }
        )
    )

async def calculator_tool_impl(expression: str) -> ToolResult:
    """Calculator tool implementation."""
    try:
        # Safe evaluation of mathematical expressions
        import ast
        import operator
        import math
        
        # Allowed operations and functions
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
            ast.UAdd: operator.pos, ast.Mod: operator.mod
        }
        
        funcs = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sqrt': math.sqrt, 'pow': pow, 'sum': sum
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](eval_expr(node.operand))
            elif isinstance(node, ast.Call) and node.func.id in funcs:
                args = [eval_expr(arg) for arg in node.args]
                return funcs[node.func.id](*args)
            elif isinstance(node, ast.Name) and node.id in funcs:
                return funcs[node.id]
            else:
                raise ValueError(f"Unsupported operation: {ast.dump(node)}")
        
        result = eval_expr(ast.parse(expression, mode='eval').body)
        return ToolResult(content=str(result))
        
    except Exception as e:
        return ToolResult(content=f"Error: {str(e)}")

async def run_evaluation(args: argparse.Namespace, server_url: str):
    """Run GSM8K evaluation with the deployed nnsight server."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "with_tools" if args.mode == "with-tools" else "no_tools"
    activation_suffix = "_activations" if args.collect_activations else ""
    
    results_dir = Path(f"examples/gsm8k_nnsight_remote/results/gsm8k_nnsight_{mode_suffix}_nsamples_{args.samples}{activation_suffix}_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GSM8K dataset
    dataset_path = results_dir / "dataset.jsonl"
    load_gsm8k_dataset(dataset_path, args.samples)
    
    # Create endpoint
    endpoint = await create_nnsight_endpoint(server_url, args.collect_activations, args.transfer_activations)
    
    # Prepare tools
    tools = []
    tool_implementations = {}
    
    if args.mode == "with-tools":
        calc_tool = get_calculator_tool()
        tools.append(calc_tool)
        tool_implementations[calc_tool.function.name] = calculator_tool_impl
    
    # Create environment
    environment = Environment(
        tools=tools,
        tool_implementations=tool_implementations
    )
    
    # Run evaluation
    logger.info(f"🏃 Starting evaluation of {args.samples} samples...")
    logger.info(f"📁 Results will be saved to: {results_dir}")
    
    if args.collect_activations:
        logger.info("🧠 Activation collection enabled - extracting residual stream features")
    
    config = RunConfig(
        max_parallel=args.parallel,
        max_turns=10,
        max_tokens_per_turn=1000,
    )
    
    reward_function = GSMRewardFunction()
    
    results = await evaluate(
        problems=str(dataset_path),
        agent_config=endpoint,
        environment=environment,
        reward_function=reward_function,
        results_dir=str(results_dir),
        config=config
    )
    
    # Calculate summary statistics
    scores = [r.reward for r in results if r.reward is not None]
    accuracy = sum(scores) / len(scores) if scores else 0.0
    
    summary = {
        "timestamp": timestamp,
        "samples": args.samples,
        "mode": args.mode,
        "activation_collection": args.collect_activations,
        "transfer_activations": args.transfer_activations,
        "accuracy": accuracy,
        "correct": sum(scores),
        "total": len(scores),
        "server_url": server_url,
        "model": "willcb/Qwen3-0.6B"
    }
    
    # Save summary
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Evaluation complete!")
    logger.info(f"📊 Accuracy: {accuracy:.1%} ({sum(scores)}/{len(scores)})")
    logger.info(f"📁 Results saved to: {results_dir}")
    
    return results_dir, summary

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GSM8K evaluation with nnsight activation collection")
    
    # Evaluation parameters
    parser.add_argument("--samples", type=int, default=3, help="Number of GSM8K samples to evaluate")
    parser.add_argument("--mode", choices=["no-tools", "with-tools"], default="no-tools", help="Evaluation mode")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel evaluations")
    parser.add_argument("--collect-activations", action="store_true", help="Collect residual stream activations")
    parser.add_argument("--transfer-activations", action="store_true", help="Transfer activations back locally (default: remote storage only)")
    
    # GPU parameters
    parser.add_argument("--gpu-id", type=str, help="Use existing GPU instance (skip provisioning)")
    parser.add_argument("--keep-running", action="store_true", help="Keep GPU running after evaluation")
    parser.add_argument("--min-vram", type=int, default=16, help="Minimum VRAM in GB")
    parser.add_argument("--max-price", type=float, default=0.60, help="Maximum price per hour")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("🚀 GSM8K nnsight evaluation starting...")
    logger.info(f"📊 Samples: {args.samples}, Mode: {args.mode}, Activations: {args.collect_activations}")
    
    # Initialize clients
    gpu_client = GPUClient()
    
    gpu_id = args.gpu_id
    server_url = None
    
    try:
        # Provision GPU if needed
        if not gpu_id:
            logger.info("🔍 Provisioning GPU instance...")
            
            # Build query for GPU with minimum VRAM and max price
            query = (gpu_client.vram_gb >= args.min_vram) & (gpu_client.price_per_hour <= args.max_price)
            
            gpu_instance = gpu_client.create(
                query=query,
                exposed_ports=[8001],  # Expose port for nnsight server
                enable_http_proxy=True,
                name="nnsight-server",
                cloud_type="secure",
                sort=lambda x: x.price_per_hour,  # Sort by price (cheapest first)
                reverse=False
            )
            
            gpu_id = gpu_instance.id
            logger.info(f"✅ Provisioned GPU: {gpu_id}")
            
            # Wait for SSH to be ready
            logger.info("⏳ Waiting for SSH connection to be ready...")
            if not gpu_instance.wait_until_ssh_ready(timeout=300):  # 5 minutes
                raise RuntimeError("Failed to get SSH connection ready")
            
            # Get SSH connection for bifrost client
            ssh_connection = gpu_instance.ssh_connection_string()
            logger.info(f"✅ SSH ready: {ssh_connection}")
            bifrost_client = BifrostClient(ssh_connection)
        else:
            logger.info(f"📱 Using existing GPU: {gpu_id}")
            # Get existing GPU instance and SSH connection
            gpu_instance = gpu_client.get_instance(gpu_id)
            if not gpu_instance:
                raise RuntimeError(f"GPU instance {gpu_id} not found")
            ssh_connection = gpu_instance.ssh_connection_string()
            bifrost_client = BifrostClient(ssh_connection)
        
        # Deploy and start nnsight server
        server_url = await deploy_nnsight_server(gpu_client, bifrost_client, gpu_id, args)
        
        # Run evaluation
        results_dir, summary = await run_evaluation(args, server_url)
        
        logger.info("🎉 Success! nnsight GSM8K evaluation completed")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise
    
    finally:
        # Cleanup
        if not args.keep_running and gpu_id and not args.gpu_id:
            logger.info(f"🧹 Terminating GPU instance: {gpu_id}")
            try:
                await gpu_client.terminate(gpu_id)
                logger.info("✅ GPU instance terminated")
            except Exception as e:
                logger.warning(f"⚠️ Failed to terminate GPU: {e}")
        elif args.keep_running and gpu_id:
            logger.info(f"🔒 Keeping GPU instance running: {gpu_id}")
            if server_url:
                logger.info(f"🌐 Server accessible at: {server_url}")

if __name__ == "__main__":
    asyncio.run(main())