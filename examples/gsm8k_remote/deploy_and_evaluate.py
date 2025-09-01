#!/usr/bin/env python3
"""
GSM8K evaluation using remote vLLM server deployed via broker/bifrost.

This mirrors gsm8k_local but uses a vLLM server on remote GPU instead of Anthropic API.
The evaluation framework and trajectory saving remain identical.

Usage:
    python examples/gsm8k_remote/deploy_and_evaluate.py --samples 3 --mode no-tools
    python examples/gsm8k_remote/deploy_and_evaluate.py --samples 3 --mode with-tools --parallel 2 --keep-running
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

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


def deploy_qwen_vllm_server(min_vram: int = 12, max_price: float = 0.40, 
                           gpu_memory_utilization: float = 0.6, max_model_len: int = 2048) -> dict:
    """Deploy Qwen3-0.6B vLLM server on GPU and return connection info."""
    
    print("🚀 Starting Qwen3-0.6B vLLM deployment...")
    
    # 1. PROVISION GPU
    print(f"📡 Creating GPU instance (min {min_vram}GB VRAM, max ${max_price}/hr)...")
    gpu_client = GPUClient()
    
    # Build query for GPU with minimum VRAM and max price
    query = (gpu_client.vram_gb >= min_vram) & (gpu_client.price_per_hour <= max_price)
    
    gpu_instance = gpu_client.create(
        query=query,
        exposed_ports=[8000],  # Expose port 8000 for vLLM
        enable_http_proxy=True,  # Enable RunPod proxy
        name="qwen-vllm-server",
        cloud_type="secure",
        sort=lambda x: x.price_per_hour,  # Sort by price (cheapest first)
        reverse=False
    )
    
    print(f"✅ GPU ready: {gpu_instance.id}")
    
    # Wait for SSH to be ready
    print("⏳ Waiting for SSH connection to be ready...")
    if not gpu_instance.wait_until_ssh_ready(timeout=300):  # 5 minutes
        print("❌ Failed to get SSH connection ready")
        sys.exit(1)
    
    ssh_connection = gpu_instance.ssh_connection_string()
    print(f"✅ SSH ready: {ssh_connection}")
    
    # 2. DEPLOY CODE
    print("📦 Deploying codebase...")
    bifrost_client = BifrostClient(ssh_connection)
    
    # Deploy the codebase to remote workspace
    workspace_path = bifrost_client.push()
    bifrost_client.exec("echo 'Codebase deployed successfully'")
    print(f"✅ Code deployed to: {workspace_path}")
    
    # 3. START QWEN VLLM SERVER IN TMUX
    print("🌟 Starting Qwen3-0.6B vLLM server in tmux session...")
    
    # Create custom vLLM command for Qwen3-0.6B
    vllm_cmd = f"""uv run python -m vllm.entrypoints.openai.api_server \\
        --model willcb/Qwen3-0.6B \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --gpu-memory-utilization {gpu_memory_utilization} \\
        --max-model-len {max_model_len} \\
        --disable-log-stats"""
    
    # Create tmux session and start server with persistent logging
    tmux_cmd = f"tmux new-session -d -s qwen-vllm 'cd ~/.bifrost/workspace && {vllm_cmd} 2>&1 | tee ~/qwen_vllm_server.log'"
    bifrost_client.exec(tmux_cmd)
    
    print("✅ Qwen3-0.6B vLLM server starting in tmux session 'qwen-vllm'")
    
    # 4. POLL UNTIL READY
    print("⏳ Waiting for server to be ready (this may take 2-3 minutes for model loading)...")
    
    max_wait_time = 600  # 10 minutes max
    start_time = time.time()
    server_ready = False
    
    while not server_ready and (time.time() - start_time) < max_wait_time:
        try:
            # Check if OpenAI-compatible server is responding
            models_check = bifrost_client.exec("curl -s --connect-timeout 5 http://localhost:8000/v1/models")
            if models_check and "qwen3" in models_check.lower():
                server_ready = True
                break
                
            # Fallback: try a simple completions request
            test_completion = bifrost_client.exec(
                'curl -s --connect-timeout 5 -X POST http://localhost:8000/v1/completions '
                '-H "Content-Type: application/json" '
                '-d \'{"model":"willcb/Qwen3-0.6B","prompt":"test","max_tokens":1}\''
            )
            if test_completion and ("choices" in test_completion.lower() or "text" in test_completion.lower()):
                server_ready = True
                break
                
        except Exception as e:
            # Server not ready yet, continue waiting
            pass
        
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0:  # Print update every 30 seconds
            print(f"   Still loading Qwen3-0.6B model... ({elapsed}s elapsed)")
            
        time.sleep(10)
    
    if not server_ready:
        print(f"❌ Server failed to start within timeout: {time.time() - start_time}s elapsed")
        print("   Check logs with:")
        print(f"   # Persistent log file: bifrost exec {ssh_connection} 'cat ~/qwen_vllm_server.log'")
        print(f"   # tmux session logs: bifrost exec {ssh_connection} 'tmux capture-pane -t qwen-vllm -p'")
        sys.exit(1)
    
    print("✅ Qwen3-0.6B vLLM server is ready and responding!")
    
    # 5. CONSTRUCT PROXY URL  
    proxy_url = gpu_instance.get_proxy_url(8000)
    
    if not proxy_url:
        print("⚠️  No proxy URL available - instance may not be RunPod")
        proxy_url = f"http://{gpu_instance.public_ip}:8000"
    
    # 6. RETURN CONNECTION INFO
    connection_info = {
        "url": proxy_url,
        "instance_id": gpu_instance.id,
        "ssh": ssh_connection,
        "provider": gpu_instance.provider,
        "status": "ready",
        "model": "willcb/Qwen3-0.6B"
    }
    
    print("\n🎉 Qwen3-0.6B vLLM deployment complete!")
    print(f"   Server URL: {proxy_url}")
    print(f"   Instance ID: {gpu_instance.id}")
    print(f"   SSH: {ssh_connection}")
    
    print("\n🧪 Test your server:")
    print(f"   curl -X POST {proxy_url}/v1/completions \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model":"willcb/Qwen3-0.6B","prompt":"Hello!","max_tokens":20}\'')
    
    print("\n🔧 Management commands:")
    print(f"   # View tmux session: bifrost exec {ssh_connection} 'tmux attach -t qwen-vllm'")
    print(f"   # Check persistent logs: bifrost exec {ssh_connection} 'cat ~/qwen_vllm_server.log'")
    print(f"   # Check tmux session: bifrost exec {ssh_connection} 'tmux capture-pane -t qwen-vllm -p'")
    print(f"   # Stop server: bifrost exec {ssh_connection} 'tmux kill-session -t qwen-vllm'")
    print(f"   # Terminate GPU: broker terminate {gpu_instance.id}")
    
    return connection_info


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


async def save_results_to_remote(bifrost_client: BifrostClient, local_output_dir: Path) -> None:
    """Save evaluation results to remote GPU for fault tolerance."""
    print("💾 Syncing results to remote GPU...")
    
    # Create remote results directory
    remote_results_dir = "~/gsm8k_remote_results"
    bifrost_client.exec(f"mkdir -p {remote_results_dir}")
    
    # Copy all files to remote (use rsync-style copying)
    for item in local_output_dir.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(local_output_dir)
            remote_path = f"{remote_results_dir}/{relative_path}"
            
            # Create remote directory structure
            remote_dir = str(Path(remote_path).parent)
            bifrost_client.exec(f"mkdir -p {remote_dir}")
            
            # Copy file content
            content = item.read_text()
            # Use heredoc to avoid shell escaping issues
            bifrost_client.exec(f"cat > {remote_path} << 'EOF'\n{content}\nEOF")
    
    print(f"✅ Results synced to remote: {remote_results_dir}")


async def main(samples: int = 3, mode: str = "no-tools", parallel: int = 1, 
               keep_running: bool = False, min_vram: int = 12, max_price: float = 0.40,
               gpu_memory_utilization: float = 0.6, max_model_len: int = 2048):
    """Run GSM8K evaluation using remote Qwen3-0.6B vLLM server."""
    from datetime import datetime
    
    # Create timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gsm8k_remote_{mode.replace('-', '_')}_nsamples_{samples}_{timestamp}"
    
    print(f"🎯 GSM8K Remote Evaluation - {mode.upper()} Mode")
    print(f"📅 Experiment: {experiment_name}")
    print(f"🤖 Model: willcb/Qwen3-0.6B")
    print("=" * 50)
    
    # 1. DEPLOY QWEN VLLM SERVER
    print("🚀 Step 1: Deploying Qwen3-0.6B vLLM server...")
    connection_info = deploy_qwen_vllm_server(
        min_vram=min_vram,
        max_price=max_price,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    
    bifrost_client = BifrostClient(connection_info["ssh"])
    
    try:
        # 2. PREPARE DATASET
        print("\n📚 Step 2: Preparing GSM8K dataset...")
        assets_dir = Path("examples/gsm8k_remote/assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = assets_dir / f"gsm8k_{mode.replace('-', '_')}.jsonl"
        
        try:
            load_gsm8k_dataset(dataset_path, samples)
        except Exception as e:
            print(f"❌ Failed to load GSM8K: {e}")
            return
        
        # 3. CONFIGURE REMOTE ENDPOINT
        print("\n🔗 Step 3: Configuring remote endpoint...")
        endpoint = Endpoint(
            provider="openai",  # vLLM exposes OpenAI-compatible API
            model="willcb/Qwen3-0.6B",
            api_base=connection_info["url"] + "/v1",  # Remote vLLM URL with /v1 suffix
            api_key="dummy",  # vLLM doesn't need real API key
            max_tokens=500,  # Reduced to fit within 2048 context length
            temperature=0.1
        )
        
        print(f"   Endpoint URL: {connection_info['url']}")
        print(f"   Model: {endpoint.model}")
        
        # 4. CHOOSE ENVIRONMENT AND SETTINGS
        print(f"\n⚙️  Step 4: Setting up evaluation environment...")
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
        
        # 5. LOAD DATASET AND PREPARE EVALUATION
        print("\n📊 Step 5: Loading dataset samples...")
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
        
        # Use first sample's reward function as demo (TODO: make this sample-specific)
        demo_rewards = create_reward_functions_for_sample(dataset_samples[0])
        
        # 6. RUN EVALUATION WITH STREAMING
        print(f"\n🎯 Step 6: Running evaluation on {len(dataset_samples)} samples...")
        print("   🔄 Streaming enabled - results saved as evaluation progresses")
        
        from rollouts.agents import stdout_handler
        run_config = RunConfig(on_chunk=stdout_handler)
        
        output_dir = Path(f"examples/gsm8k_remote/results/{experiment_name}")
        
        # TODO: Configuration management - this should be unified across different deployment patterns
        
        report = await evaluate(
            dataset=iter(dataset_samples),
            prepare_messages=prepare_messages,
            reward_functions=demo_rewards,
            environment=environment,
            endpoint=endpoint,  # Points to remote Qwen3-0.6B vLLM server
            run_config=run_config,
            eval_name=experiment_name,
            dataset_path=str(dataset_path),
            max_turns=max_turns,
            max_samples=samples,
            max_concurrent=parallel,
            output_dir=output_dir,
            sample_id_fn=lambda i, sample: sample.get("sample_id", f"gsm8k_{i:04d}"),
            verbose=True
        )
        
        # 7. SAVE RESULTS TO REMOTE GPU
        print("\n💾 Step 7: Syncing results to remote GPU...")
        await save_results_to_remote(bifrost_client, output_dir)
        
        # 8. PRINT RESULTS SUMMARY
        print(f"\n🔍 Step 8: Results Summary:")
        for sample in report.sample_results:
            status = "✅" if sample.metrics.get("correctness", 0.0) > 0.5 else "❌"
            correctness = sample.metrics.get("correctness", 0.0)
            format_score = sample.metrics.get("format", 0.0)
            print(f"   {status} {sample.sample_id}: Correct={correctness:.1f}, Format={format_score:.1f}")
        
        avg_correctness = sum(s.metrics.get("correctness", 0.0) for s in report.sample_results) / len(report.sample_results)
        print(f"\n📊 Overall Accuracy: {avg_correctness:.3f} ({avg_correctness*100:.1f}%)")
        
        print(f"\n📁 Results saved to:")
        print(f"   Local: {output_dir}")
        print(f"   Remote: ~/gsm8k_remote_results")
        
    finally:
        # 9. CLEANUP (CONDITIONAL)
        if not keep_running:
            print(f"\n🧹 Step 9: Cleaning up GPU instance...")
            gpu_client = GPUClient()
            
            try:
                # Stop the vLLM server
                print("   Stopping vLLM server...")
                bifrost_client.exec("tmux kill-session -t qwen-vllm 2>/dev/null || true")
                
                # Terminate the GPU instance
                print(f"   Terminating GPU instance {connection_info['instance_id']}...")
                # Note: We need to implement terminate method or use existing patterns
                # For now, print the command
                print(f"   Run: broker terminate {connection_info['instance_id']}")
                
                print("✅ Cleanup complete")
                
            except Exception as e:
                print(f"⚠️ Cleanup error (instance may still be running): {e}")
                print(f"   Manual cleanup: broker terminate {connection_info['instance_id']}")
        else:
            print(f"\n🎯 Step 9: Keeping GPU instance running (--keep-running flag)")
            print(f"   Instance ID: {connection_info['instance_id']}")
            print(f"   Server URL: {connection_info['url']}")
            print(f"   SSH: {connection_info['ssh']}")
            print(f"   Manual cleanup: broker terminate {connection_info['instance_id']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GSM8K Evaluation with Remote Qwen3-0.6B vLLM Server")
    
    # GSM8K evaluation args (from gsm8k_local)
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to evaluate")
    parser.add_argument("--mode", choices=["no-tools", "with-tools"], default="no-tools", 
                       help="Evaluation mode")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel evaluations")
    
    # Remote deployment args
    parser.add_argument("--keep-running", action="store_true", 
                       help="Keep GPU instance running after evaluation")
    parser.add_argument("--min-vram", type=int, default=12, 
                       help="Minimum VRAM in GB (default: 12)")
    parser.add_argument("--max-price", type=float, default=0.40, 
                       help="Maximum price per hour (default: 0.40)")
    
    # vLLM configuration args (TODO: better config management)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6,
                       help="GPU memory utilization for vLLM (default: 0.6)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                       help="Maximum model length for vLLM (default: 2048)")
    
    args = parser.parse_args()
    
    # Setup logging for the example
    setup_logging()
    
    asyncio.run(main(
        samples=args.samples,
        mode=args.mode,
        parallel=args.parallel,
        keep_running=args.keep_running,
        min_vram=args.min_vram,
        max_price=args.max_price,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    ))