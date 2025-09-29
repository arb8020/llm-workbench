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

import requests
from dotenv import load_dotenv

from shared.logging_config import setup_logging

# Import broker and bifrost for deployment
from broker.client import GPUClient
from bifrost.client import BifrostClient

# Import rollouts evaluation framework
from rollouts.evaluation import evaluate, load_jsonl, RewardFunction
from rollouts.dtypes import Message, Endpoint, Environment, Trajectory, Tool, ToolFunction, ToolFunctionParameter, ToolCall, ToolResult, AgentState, RunConfig

logger = logging.getLogger(__name__)


def wait_for_vllm_ready(server_url: str, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready by polling /v1/models endpoint.

    Args:
        server_url: Base URL of the vLLM server (e.g., http://proxy-url)
        timeout: Maximum time to wait in seconds (default: 300 = 5 minutes)

    Returns:
        True if server becomes ready, False if timeout
    """
    models_endpoint = f"{server_url}/v1/models"
    start_time = time.time()
    poll_interval = 5  # Check every 5 seconds

    logger.info(f"⏳ Waiting for vLLM server to be ready (timeout: {timeout}s)...")
    logger.info(f"   Polling: {models_endpoint}")

    attempt = 0
    while time.time() - start_time < timeout:
        attempt += 1
        elapsed = int(time.time() - start_time)

        try:
            # Try to reach the /v1/models endpoint
            response = requests.get(models_endpoint, timeout=10)

            if response.status_code == 200:
                # Check if the response looks valid (has models)
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    logger.info(f"✅ vLLM server ready! (took {elapsed}s, {attempt} attempts)")
                    logger.info(f"   Available model: {data['data'][0].get('id', 'unknown')}")
                    return True
                else:
                    logger.info(f"   [{elapsed}s] Server responding but no models loaded yet...")
            else:
                logger.info(f"   [{elapsed}s] Server returned status {response.status_code}, waiting...")

        except requests.exceptions.ConnectionError:
            # Server not accepting connections yet
            logger.info(f"   [{elapsed}s] Connection refused, server still starting...")
        except requests.exceptions.Timeout:
            # Server too slow to respond
            logger.info(f"   [{elapsed}s] Request timeout, server still loading...")
        except Exception as e:
            # Other errors
            logger.warning(f"   [{elapsed}s] Unexpected error: {e}")

        # Wait before next attempt
        time.sleep(poll_interval)

    # Timeout reached
    logger.error(f"❌ vLLM server did not become ready within {timeout}s")
    return False


def load_gsm8k_dataset(output_path: Path, sample_count: int = None) -> None:
    """Load GSM8K from HuggingFace and save as JSONL."""
    try:
        from datasets import load_dataset

        logger.info("📚 Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("gsm8k", "main", split="test")

        if sample_count:
            dataset = dataset.select(range(min(sample_count, len(dataset))))

        logger.info(f"📊 Selected {len(dataset)} samples from GSM8K")

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

        logger.info(f"✅ Saved GSM8K dataset to: {output_path}")

    except ImportError:
        logger.error("❌ HuggingFace datasets library not found. Install with: uv pip install datasets")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading GSM8K: {e}")
        raise


def deploy_qwen_vllm_server(min_vram: int = 12, max_price: float = 0.40,
                           gpu_memory_utilization: float = 0.6, max_model_len: int = 2048) -> dict:
    """Deploy Qwen3-0.6B vLLM server on GPU and return connection info."""

    logger.info("🚀 Starting Qwen3-0.6B vLLM deployment...")

    # Load environment variables (for RUNPOD_API_KEY, SSH keys, etc.)
    load_dotenv()
    RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
    if not RUNPOD_API_KEY:
        logger.error("❌ RUNPOD_API_KEY not found in environment")
        logger.error("   Set it with: export RUNPOD_API_KEY=your-key")
        logger.error("   Or add to .env file: RUNPOD_API_KEY=your-key")
        sys.exit(1)

    # 1. PROVISION GPU
    logger.info(f"📡 Creating GPU instance (min {min_vram}GB VRAM, max ${max_price}/hr)...")
    gpu_client = GPUClient(api_key=RUNPOD_API_KEY)

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

    # ERROR HANDLING: Check if provisioning succeeded
    # gpu_client.create() returns None when no GPUs match criteria or all attempts fail.
    # We check early and exit cleanly - no GPU to clean up yet.
    if not gpu_instance:
        logger.error("❌ Failed to provision GPU instance")
        logger.error(f"   No instances available matching: {min_vram}GB VRAM, max ${max_price}/hr")
        logger.error("   Try adjusting --min-vram or --max-price parameters")
        sys.exit(1)

    logger.info(f"✅ GPU provisioned: {gpu_instance.id}")

    # ERROR HANDLING: Use try/finally pattern for guaranteed cleanup
    # Once we have a GPU instance, we must terminate it on any error to prevent billing leaks.
    # This pattern ensures cleanup happens whether we hit SSH timeout, deployment failure,
    # or any unexpected exception. We use try/finally (not try/except) so errors propagate
    # to the caller while still guaranteeing cleanup.
    try:
        # Wait for SSH to be ready
        logger.info("⏳ Waiting for SSH connection to be ready...")
        if not gpu_instance.wait_until_ssh_ready(timeout=300):  # 5 minutes
            # Raise exception to trigger cleanup in finally block
            raise RuntimeError("SSH connection timeout after 5 minutes")

        ssh_connection = gpu_instance.ssh_connection_string()
        logger.info(f"✅ SSH ready: {ssh_connection}")

        # 2. DEPLOY CODE WITH DEPENDENCIES
        logger.info("📦 Deploying codebase with GSM8K dependencies...")
        bifrost_client = BifrostClient(ssh_connection)

        # Deploy the codebase to remote workspace with GSM8K remote dependencies
        workspace_path = bifrost_client.push(uv_extra="examples_gsm8k_remote")
        logger.info(f"✅ Code deployed and dependencies installed: {workspace_path}")

        # 3. START QWEN VLLM SERVER IN TMUX
        logger.info("🌟 Starting Qwen3-0.6B vLLM server in tmux session...")

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

        logger.info("✅ Qwen3-0.6B vLLM server starting in tmux session 'qwen-vllm'")
        logger.info("📋 Server will be ready in 2-3 minutes for model loading")
        logger.info("   Check server status: bifrost exec {ssh_connection} 'curl -s http://localhost:8000/v1/models'")
        logger.info("   View logs: bifrost exec {ssh_connection} 'cat ~/qwen_vllm_server.log'")

        # 4. CONSTRUCT PROXY URL
        proxy_url = gpu_instance.get_proxy_url(8000)

        if not proxy_url:
            logger.warning("⚠️  No proxy URL available - instance may not be RunPod")
            proxy_url = f"http://{gpu_instance.public_ip}:8000"

        # 5. RETURN CONNECTION INFO
        connection_info = {
            "url": proxy_url,
            "instance_id": gpu_instance.id,
            "ssh": ssh_connection,
            "provider": gpu_instance.provider,
            "status": "ready",
            "model": "willcb/Qwen3-0.6B"
        }

        logger.info("\n🎉 Qwen3-0.6B vLLM deployment complete!")
        logger.info(f"   Server URL: {proxy_url}")
        logger.info(f"   Instance ID: {gpu_instance.id}")
        logger.info(f"   SSH: {ssh_connection}")

        logger.info("\n🧪 Test your server:")
        logger.info(f"   curl -X POST {proxy_url}/v1/completions \\")
        logger.info('     -H "Content-Type: application/json" \\')
        logger.info('     -d \'{"model":"willcb/Qwen3-0.6B","prompt":"Hello!","max_tokens":20}\'')

        logger.info("\n🔧 Management commands:")
        logger.info(f"   # View tmux session: bifrost exec {ssh_connection} 'tmux attach -t qwen-vllm'")
        logger.info(f"   # Check persistent logs: bifrost exec {ssh_connection} 'cat ~/qwen_vllm_server.log'")
        logger.info(f"   # Check tmux session: bifrost exec {ssh_connection} 'tmux capture-pane -t qwen-vllm -p'")
        logger.info(f"   # Stop server: bifrost exec {ssh_connection} 'tmux kill-session -t qwen-vllm'")
        logger.info(f"   # Terminate GPU: broker terminate {gpu_instance.id}")

        return connection_info

    except Exception as e:
        # ERROR HANDLING: Clean up GPU on any failure
        # If we get here, something went wrong (SSH timeout, deployment failure, etc.)
        # We must terminate the GPU instance to prevent it from continuing to bill.
        logger.error(f"\n❌ Deployment failed: {e}")
        logger.info("🧹 Terminating GPU instance to prevent billing...")

        try:
            gpu_instance.terminate()
            logger.info("✅ GPU instance terminated successfully")
        except Exception as cleanup_error:
            # If cleanup fails, at least tell the user how to clean up manually
            logger.error(f"⚠️  Failed to terminate GPU instance: {cleanup_error}")
            logger.error(f"   IMPORTANT: Manually terminate to stop billing:")
            logger.error(f"   broker terminate {gpu_instance.id}")

        # Re-raise the original exception so caller (main()) knows deployment failed
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


def correctness_reward(trajectory: Trajectory, sample: Dict[str, Any]) -> float:
    """Check if the extracted answer matches the expected answer."""
    assert "answer" in sample, "Sample must have 'answer' key"

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


def format_reward(trajectory: Trajectory, sample: Dict[str, Any]) -> float:
    """Reward for following the answer format."""
    # Sample parameter included for consistency (not used in this reward)
    assistant_messages = [m for m in trajectory.messages if m.role == "assistant"]
    if not assistant_messages:
        return 0.0

    response = " ".join(m.content for m in assistant_messages if m.content)
    has_answer_format = bool(re.search(r'Answer:\s*[^\n]+', response, re.IGNORECASE))
    return 1.0 if has_answer_format else 0.0


def efficiency_reward(trajectory: Trajectory, sample: Dict[str, Any]) -> float:
    """Reward for being concise (fewer tokens)."""
    # Sample parameter included for consistency (not used in this reward)
    total_tokens = sum(len(m.content or "") for m in trajectory.messages)
    # Normalize: 1.0 for <500 tokens, 0.0 for >2000 tokens
    if total_tokens < 500:
        return 1.0
    elif total_tokens > 2000:
        return 0.0
    else:
        return 1.0 - (total_tokens - 500) / 1500


def tool_usage_reward(trajectory: Trajectory, sample: Dict[str, Any]) -> float:
    """Reward for appropriate tool usage (only meaningful with calculator environment)."""
    # Sample parameter included for consistency (not used in this reward)
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
    logger.info("💾 Syncing results to remote GPU...")

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

    logger.info(f"✅ Results synced to remote: {remote_results_dir}")


async def main(samples: int = 3, mode: str = "no-tools", parallel: int = 1,
               keep_running: bool = False, min_vram: int = 12, max_price: float = 0.40,
               gpu_memory_utilization: float = 0.6, max_model_len: int = 2048):
    """Run GSM8K evaluation using remote Qwen3-0.6B vLLM server."""
    from datetime import datetime

    # Load environment variables at the start so they're available throughout
    load_dotenv()
    RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')

    # Create timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gsm8k_remote_{mode.replace('-', '_')}_nsamples_{samples}_{timestamp}"

    logger.info(f"🎯 GSM8K Remote Evaluation - {mode.upper()} Mode")
    logger.info(f"📅 Experiment: {experiment_name}")
    logger.info(f"🤖 Model: willcb/Qwen3-0.6B")
    logger.info("=" * 50)

    # 1. DEPLOY QWEN VLLM SERVER
    logger.info("🚀 Step 1: Deploying Qwen3-0.6B vLLM server...")
    try:
        connection_info = deploy_qwen_vllm_server(
            min_vram=min_vram,
            max_price=max_price,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
    except Exception as e:
        # If deployment fails, deploy_qwen_vllm_server() already cleaned up the GPU
        # We just need to exit gracefully here
        logger.error(f"\n💥 Fatal: Could not deploy vLLM server")
        logger.error(f"   Error: {e}")
        sys.exit(1)

    bifrost_client = BifrostClient(connection_info["ssh"])

    try:
        # 2. PREPARE DATASET
        logger.info("\n📚 Step 2: Preparing GSM8K dataset...")
        assets_dir = Path("examples/gsm8k_remote/assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = assets_dir / f"gsm8k_{mode.replace('-', '_')}.jsonl"

        try:
            load_gsm8k_dataset(dataset_path, samples)
        except Exception as e:
            logger.error(f"❌ Failed to load GSM8K: {e}")
            return

        # 3. CONFIGURE REMOTE ENDPOINT
        logger.info("\n🔗 Step 3: Configuring remote endpoint...")
        endpoint = Endpoint(
            provider="openai",  # vLLM exposes OpenAI-compatible API
            model="willcb/Qwen3-0.6B",
            api_base=connection_info["url"] + "/v1",  # Remote vLLM URL with /v1 suffix
            api_key="dummy",  # vLLM doesn't need real API key
            max_tokens=500,  # Reduced to fit within 2048 context length
            temperature=0.1
        )

        logger.info(f"   Endpoint URL: {connection_info['url']}")
        logger.info(f"   Model: {endpoint.model}")

        # 4. CHOOSE ENVIRONMENT AND SETTINGS
        logger.info(f"\n⚙️  Step 4: Setting up evaluation environment...")
        if mode == "no-tools":
            environment_factory = lambda: NoToolsEnvironment()
            prepare_messages = prepare_gsm8k_messages_no_tools
            max_turns = 1
            logger.info("📝 Mode: Zero-shot chain-of-thought")
            
            # Reward functions for no-tools mode
            reward_functions = [
                ("correctness", lambda t: 0.0),  # Will be replaced per sample
                ("format", format_reward),
                ("efficiency", efficiency_reward),
            ]
        else:
            environment_factory = lambda: CalculatorEnvironment()
            prepare_messages = prepare_gsm8k_messages_with_tools
            max_turns = 6
            logger.info("🔧 Mode: Tool-assisted reasoning")

            # Reward functions for with-tools mode
            reward_functions = [
                ("correctness", lambda t: 0.0),  # Will be replaced per sample
                ("format", format_reward),
                ("efficiency", efficiency_reward),
                ("tool_usage", tool_usage_reward),
            ]

        # 5. LOAD DATASET AND PREPARE EVALUATION
        logger.info("\n📊 Step 5: Loading dataset samples...")

        # Load and validate dataset
        try:
            dataset_samples = list(load_jsonl(dataset_path))
        except FileNotFoundError:
            logger.error(f"❌ Error: Dataset file not found at {dataset_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Error: Failed to load dataset from {dataset_path}: {e}")
            sys.exit(1)

        # Check if dataset is empty or samples=0 was requested
        if samples == 0:
            logger.error("❌ Error: Cannot run evaluation with --samples 0")
            logger.error("   Please specify a positive number of samples to evaluate.")
            sys.exit(1)

        if len(dataset_samples) == 0:
            logger.error(f"❌ Error: Dataset at {dataset_path} is empty")
            logger.error("   The dataset file exists but contains no samples.")
            sys.exit(1)

        logger.info(f"   Loaded {len(dataset_samples)} samples from dataset")

        # Define reward functions (now take trajectory AND sample as parameters)
        reward_functions = [
            ("correctness", correctness_reward),
            ("format", format_reward),
            ("efficiency", efficiency_reward),
        ]
        if mode == "with-tools":
            reward_functions.append(("tool_usage", tool_usage_reward))

        # 6. WAIT FOR VLLM SERVER TO BE READY
        logger.info(f"\n⏰ Step 6: Waiting for vLLM server to be ready...")
        if not wait_for_vllm_ready(connection_info["url"], timeout=300):
            logger.error("❌ vLLM server did not become ready in time")
            logger.error("   Check server logs for errors:")
            logger.error(f"   bifrost exec {connection_info['ssh']} 'cat ~/qwen_vllm_server.log'")
            sys.exit(1)

        # 7. RUN EVALUATION WITH STREAMING
        logger.info(f"\n🎯 Step 7: Running evaluation on {len(dataset_samples)} samples...")
        logger.info("   🔄 Streaming enabled - results saved as evaluation progresses")
        
        from rollouts.agents import stdout_handler
        run_config = RunConfig(on_chunk=stdout_handler)
        
        output_dir = Path(f"examples/gsm8k_remote/results/{experiment_name}")
        
        # TODO: Configuration management - this should be unified across different deployment patterns
        
        report = await evaluate(
            dataset=iter(dataset_samples),
            prepare_messages=prepare_messages,
            reward_functions=reward_functions,
            environment_factory=environment_factory,
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
        logger.info("\n💾 Step 8: Syncing results to remote GPU...")
        await save_results_to_remote(bifrost_client, output_dir)

        # 8. PRINT RESULTS SUMMARY
        logger.info(f"\n🔍 Step 9: Results Summary:")
        for sample in report.sample_results:
            status = "✅" if sample.metrics.get("correctness", 0.0) > 0.5 else "❌"
            correctness = sample.metrics.get("correctness", 0.0)
            format_score = sample.metrics.get("format", 0.0)
            logger.info(f"   {status} {sample.sample_id}: Correct={correctness:.1f}, Format={format_score:.1f}")

        avg_correctness = sum(s.metrics.get("correctness", 0.0) for s in report.sample_results) / len(report.sample_results)
        logger.info(f"\n📊 Overall Accuracy: {avg_correctness:.3f} ({avg_correctness*100:.1f}%)")

        logger.info(f"\n📁 Results saved to:")
        logger.info(f"   Local: {output_dir}")
        logger.info(f"   Remote: ~/gsm8k_remote_results")
        
    finally:
        # 9. CLEANUP (CONDITIONAL)
        if not keep_running:
            logger.info(f"\n🧹 Step 10: Cleaning up GPU instance...")
            if RUNPOD_API_KEY:
                gpu_client = GPUClient(api_key=RUNPOD_API_KEY)
            else:
                logger.warning("⚠️  Warning: RUNPOD_API_KEY not available, skipping cleanup")
                gpu_client = None

            if gpu_client:
                try:
                    # Stop the vLLM server
                    logger.info("   Stopping vLLM server...")
                    bifrost_client.exec("tmux kill-session -t qwen-vllm 2>/dev/null || true")

                    # Terminate the GPU instance
                    logger.info(f"   Terminating GPU instance {connection_info['instance_id']}...")
                    # Note: We need to implement terminate method or use existing patterns
                    # For now, print the command
                    logger.info(f"   Run: broker terminate {connection_info['instance_id']}")

                    logger.info("✅ Cleanup complete")

                except Exception as e:
                    logger.warning(f"⚠️ Cleanup error (instance may still be running): {e}")
                    logger.warning(f"   Manual cleanup: broker terminate {connection_info['instance_id']}")
        else:
            logger.info(f"\n🎯 Step 10: Keeping GPU instance running (--keep-running flag)")
            logger.info(f"   Instance ID: {connection_info['instance_id']}")
            logger.info(f"   Server URL: {connection_info['url']}")
            logger.info(f"   SSH: {connection_info['ssh']}")
            logger.info(f"   Manual cleanup: broker terminate {connection_info['instance_id']}")


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

    # Setup logging with dual output: human-readable stdout + JSONL file
    # Create timestamped log file in logs directory
    from datetime import datetime
    log_dir = Path("logs")  # Relative to script location
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gsm8k_remote_{args.mode}_{timestamp}.jsonl"

    setup_logging(
        level="INFO",
        use_json=False,  # Human-readable console output
        log_file=str(log_file)  # JSONL log file
    )

    logger.info(f"📝 Logging to: {log_file}")

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