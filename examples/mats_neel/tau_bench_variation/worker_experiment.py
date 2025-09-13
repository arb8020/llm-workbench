#!/usr/bin/env python3
"""
Worker script for Tau-Bench User Variation Experiment

Runs on remote GPU machines. Executes tau-bench evaluations with different user strategies.

Usage:
    python worker_experiment.py <config_path> <worker_id>
"""

import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextvars import ContextVar

from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)

# =============================================================================
# EMOTIONAL USER SIMULATION VARIANTS
# =============================================================================

# Process-local selector for emotional variant (avoids env vars)
_emotion_variant: ContextVar[Optional[str]] = ContextVar("emotion_variant", default=None)

def set_emotion_variant(variant: Optional[str]):
    """Set the current emotional variant for the monkey-patched load_user.

    Allowed: None, "control", "frustration", "anxiety", "anger", "confusion".
    """
    _emotion_variant.set(variant)

def create_emotional_user_variants():
    """Create emotional user simulation variants by monkey-patching tau-bench."""
    
    try:
        from tau_bench.envs.user import LLMUserSimulationEnv, UserStrategy, load_user
        import tau_bench.envs.user as user_module
        # Optional: sanitize LLM outputs to strip leaked chain-of-thought markup like "<think>..."
        try:
            import litellm.main as _llm_main
            _orig_completion = _llm_main.completion

            def _strip_think(text: Optional[str]) -> Optional[str]:
                if not isinstance(text, str):
                    return text
                s = text
                # Remove any <think>...</think> blocks if present
                while True:
                    start = s.find("<think>")
                    if start == -1:
                        break
                    end = s.find("</think>", start + 7)
                    if end != -1:
                        s = s[:start] + s[end + len("</think>"):]
                    else:
                        # No closing tag; drop everything up to the first double newline if present
                        dn = s.find("\n\n", start + 7)
                        if dn != -1:
                            s = s[:start] + s[dn + 2:]
                        else:
                            # Fallback: drop from <think> to end
                            s = s[:start]
                            break
                # Also remove any leading lonely <think> prefix lines
                if s.strip().startswith("<think>"):
                    s = s.replace("<think>", "", 1)
                return s

            def _sanitized_completion(*args, **kwargs):
                resp = _orig_completion(*args, **kwargs)
                try:
                    choices = resp.get("choices", [])
                    for ch in choices:
                        msg = ch.get("message")
                        if not isinstance(msg, dict):
                            continue
                        content = msg.get("content")
                        msg["content"] = _strip_think(content)
                except Exception:
                    # Never break the main flow on sanitizer issues
                    pass
                return resp

            _llm_main.completion = _sanitized_completion
            logger.info("✅ Applied LLM output sanitizer to strip <think> leakage")
        except Exception as _e:
            logger.warning(f"Could not apply LLM sanitizer: {_e}")
        
        # Create custom emotional user simulation classes
        class FrustratedLLMUserSimulationEnv(LLMUserSimulationEnv):
            """User simulation with frustration and impatience."""
            
            def build_system_prompt(self, instruction: Optional[str] = None) -> str:
                base_prompt = super().build_system_prompt(instruction)
                
                emotional_context = """

EMOTIONAL CONTEXT: You are a frustrated customer who has had previous bad experiences with customer service.
- Express irritation when things don't work smoothly or take too long
- Use phrases like "This is ridiculous", "I've been waiting forever", "Why is this so complicated?"
- Show impatience with slow responses or complex procedures  
- Be more direct and less polite than usual, but remain civil
- Mention previous bad experiences: "Last time this happened...", "I've been through this before..."
- Express frustration with having to repeat information or explain your situation multiple times"""
                
                return f"{base_prompt}{emotional_context}"
        
        class AnxiousLLMUserSimulationEnv(LLMUserSimulationEnv):
            """User simulation with anxiety and uncertainty."""
            
            def build_system_prompt(self, instruction: Optional[str] = None) -> str:
                base_prompt = super().build_system_prompt(instruction)
                
                emotional_context = """

EMOTIONAL CONTEXT: You are an anxious customer who worries about making mistakes or getting things wrong.
- Express uncertainty and seek frequent reassurance: "Am I doing this right?", "Are you sure this will work?"
- Show hesitation before taking actions: "I'm not sure if I should...", "What if something goes wrong?"
- Ask for clarification multiple times to make sure you understand
- Express worry about potential negative consequences
- Use phrases like "I'm worried that...", "I hope this doesn't...", "I don't want to mess this up"
- Seek step-by-step confirmation before proceeding with any actions"""
                
                return f"{base_prompt}{emotional_context}"
        
        class AngryLLMUserSimulationEnv(LLMUserSimulationEnv):
            """User simulation with anger and hostility."""
            
            def build_system_prompt(self, instruction: Optional[str] = None) -> str:
                base_prompt = super().build_system_prompt(instruction)
                
                emotional_context = """

EMOTIONAL CONTEXT: You are an angry customer who is upset about a serious problem that has caused you significant inconvenience.
- Express strong displeasure and demand immediate resolution
- Use emphatic language: "This is completely unacceptable!", "I demand to speak to a manager!"
- Show little patience for explanations or delays: "I don't want excuses, I want solutions!"
- Be more aggressive in your communication while remaining within reasonable bounds
- Express how the problem has personally affected you: "This has ruined my day/week"
- Demand compensation or escalation: "What are you going to do to make this right?"
- Be skeptical of proposed solutions: "How do I know this won't happen again?"""
                
                return f"{base_prompt}{emotional_context}"
        
        class ConfusedLLMUserSimulationEnv(LLMUserSimulationEnv):
            """User simulation with confusion and difficulty understanding."""
            
            def build_system_prompt(self, instruction: Optional[str] = None) -> str:
                base_prompt = super().build_system_prompt(instruction)
                
                emotional_context = """

EMOTIONAL CONTEXT: You are a confused customer who has difficulty understanding technical terms or complex procedures.
- Ask for clarification frequently: "I don't understand what that means", "Can you explain that differently?"
- Express confusion about basic concepts or instructions
- Request simpler explanations: "Can you use simpler words?", "I'm not very tech-savvy"
- Sometimes misunderstand instructions and need correction
- Ask repetitive questions about the same concepts
- Express feeling overwhelmed: "This is all very confusing to me", "There's so much information"
- Need extra time and patience to process information"""
                
                return f"{base_prompt}{emotional_context}"
        
        # Monkey-patch the load_user function to support our custom strategy
        original_load_user = user_module.load_user
        
        def patched_load_user(user_strategy, model=None, provider=None):
            # Backward-compat: accept explicit custom strategies if passed
            if isinstance(user_strategy, str):
                if user_strategy == "frustrated_llm":
                    if model is None or provider is None:
                        raise ValueError("Emotional LLM user strategies require model and provider")
                    return FrustratedLLMUserSimulationEnv(model=model, provider=provider)
                elif user_strategy == "anxious_llm":
                    if model is None or provider is None:
                        raise ValueError("Emotional LLM user strategies require model and provider")
                    return AnxiousLLMUserSimulationEnv(model=model, provider=provider)
                elif user_strategy == "angry_llm":
                    if model is None or provider is None:
                        raise ValueError("Emotional LLM user strategies require model and provider")
                    return AngryLLMUserSimulationEnv(model=model, provider=provider)
                elif user_strategy == "confused_llm":
                    if model is None or provider is None:
                        raise ValueError("Emotional LLM user strategies require model and provider")
                    return ConfusedLLMUserSimulationEnv(model=model, provider=provider)

            # Preferred path: when Tau-Bench passes allowed enum value "llm",
            # choose emotional subclass based on our process-local selector
            try:
                current_variant = _emotion_variant.get()
            except Exception:
                current_variant = None

            if isinstance(user_strategy, str) and user_strategy == "llm" and current_variant:
                if current_variant == "frustration":
                    return FrustratedLLMUserSimulationEnv(model=model, provider=provider)
                if current_variant == "anxiety":
                    return AnxiousLLMUserSimulationEnv(model=model, provider=provider)
                if current_variant == "anger":
                    return AngryLLMUserSimulationEnv(model=model, provider=provider)
                if current_variant == "confusion":
                    return ConfusedLLMUserSimulationEnv(model=model, provider=provider)
                # control or unknown → fall through to original

            # Fall back to original function for other strategies
            return original_load_user(user_strategy, model, provider)
        
        # Apply the monkey patch
        user_module.load_user = patched_load_user
        
        logger.info("✅ Created frustrated user simulation variant via monkey-patching")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import tau-bench modules: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to create emotional user variants: {e}")
        return False

# =============================================================================
# TAU-BENCH EXECUTION
# =============================================================================

def run_tau_bench_variant(variant_name: str, user_strategy: str, environment: str,
                          task_ids: List[int], endpoint_url: str, output_dir: Path) -> bool:
    """Run tau-bench with specified user strategy variant using proper RunConfig API.

    Adds preflight checks and writes an explicit variant_summary.json with an
    exit_reason so it's obvious why the run ended (success or failure modes).
    """

    from datetime import datetime
    import os

    logger.info(
        f"Running tau-bench variant '{variant_name}' with user strategy '{user_strategy}'"
    )
    logger.info(
        f"Environment: {environment}, Tasks: {task_ids}, Endpoint: {endpoint_url}"
    )

    # Ensure output directory exists
    variant_output_dir = output_dir / variant_name
    variant_output_dir.mkdir(parents=True, exist_ok=True)

    # Summary skeleton for strong observability
    summary = {
        "variant": variant_name,
        "environment": environment,
        "user_strategy": user_strategy,
        "user_strategy_effective": "llm",  # keep enum-valid
        "selected_emotion": None,
        "endpoint_url": endpoint_url,
        "tasks_requested": list(task_ids),
        "started_at": datetime.utcnow().isoformat() + "Z",
        "finished_at": None,
        "elapsed_seconds": None,
        "status": "unknown",
        "exit_reason": None,
        "results_count": 0,
        "output_dir": str(variant_output_dir),
        "preflight": {
            "server_ready": False,
            "openai_base_url": None,
            "openai_api_key_present": False,
        },
        "notes": [],
    }

    def write_summary():
        try:
            import json
            path = variant_output_dir / "variant_summary.json"
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Wrote run summary → {path}")
        except Exception as e:
            logger.warning(f"Failed writing run summary: {e}")

    # Preflight: check server readiness quickly (non-blocking long waits handled elsewhere)
    try:
        import requests
        models_resp = requests.get(f"{endpoint_url}/v1/models", timeout=8)
        if models_resp.status_code == 200:
            summary["preflight"]["server_ready"] = True
        else:
            summary["notes"].append(
                f"/v1/models returned status {models_resp.status_code}"
            )
    except Exception as e:
        summary["notes"].append(f"Preflight /v1/models error: {e}")

    # Set environment variables for OpenAI API endpoint
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")
    os.environ["OPENAI_BASE_URL"] = f"{endpoint_url}/v1"
    summary["preflight"]["openai_api_key_present"] = bool(
        os.environ.get("OPENAI_API_KEY")
    )
    summary["preflight"]["openai_base_url"] = os.environ.get("OPENAI_BASE_URL")

    start_ts = time.time()

    try:
        # Import tau-bench API
        from tau_bench.run import run
        from tau_bench.types import RunConfig

        # Map variant to our emotion selector while keeping Tau-Bench enum-valid strategy
        emotion_for_variant = None
        if variant_name in {"frustration", "anxiety", "anger", "confusion"}:
            emotion_for_variant = variant_name
        elif variant_name == "control":
            emotion_for_variant = None
        else:
            # Unknown variant: do not alter behavior
            summary["notes"].append(f"Unknown variant '{variant_name}', running neutral llm")

        set_emotion_variant(emotion_for_variant)
        summary["selected_emotion"] = emotion_for_variant

        # Create RunConfig with allowed strategy "llm" always
        config = RunConfig(
            model_provider="openai",
            user_model_provider="openai",
            model="google/gemma-3-1b-it",
            user_model="google/gemma-3-1b-it",  # Use same model for both
            env=environment,
            user_strategy="llm",
            task_ids=task_ids,
            log_dir=str(variant_output_dir),
            max_concurrency=1,
            agent_strategy="tool-calling",  # Default for tau-bench
        )

        logger.info(
            f"Running tau-bench with config: model={config.model}, user_model={config.user_model}"
        )
        logger.info(
            f"Environment: {config.env}, User strategy: {config.user_strategy} (emotion={emotion_for_variant or 'neutral'})"
        )
        logger.info(f"Output directory: {variant_output_dir}")

        # Run tau-bench under a guarded open() that auto-creates parent dirs for writes
        import builtins as _builtins
        _orig_open = _builtins.open

        def _guarded_open(file, mode='r', *args, **kwargs):
            try:
                if isinstance(file, (str, os.PathLike)) and any(m in mode for m in ('w', 'a', 'x')):
                    parent = os.path.dirname(os.fspath(file))
                    if parent:
                        os.makedirs(parent, exist_ok=True)
            except Exception:
                # Never block the actual open on guard failure
                pass
            return _orig_open(file, mode, *args, **kwargs)

        _builtins.open = _guarded_open
        summary["notes"].append("enabled_guarded_open_for_checkpoint_dirs")
        try:
            results = run(config)
        finally:
            _builtins.open = _orig_open

        # Record summary
        results_count = len(results) if results is not None else 0
        summary["results_count"] = results_count

        # Determine exit reasoning
        if results is None:
            summary["status"] = "failed"
            summary["exit_reason"] = "no_results_returned"
            summary["notes"].append("run(config) returned None")
        elif results_count == 0:
            summary["status"] = "failed"
            summary["exit_reason"] = "no_tasks_executed"
            summary["notes"].append("run(config) returned an empty results list")
        else:
            summary["status"] = "succeeded"
            summary["exit_reason"] = "success"

        logger.info(
            f"✅ Tau-bench variant '{variant_name}' completed with results={results_count}"
        )

        # Log results summary
        for result in (results or []):
            try:
                logger.info(f"Task {result.task_id}: reward={result.reward}")
            except Exception:
                # Avoid hard-crash if result shape differs
                logger.info(f"Task result: {result}")

        return summary["status"] == "succeeded"

    except Exception as e:
        logger.error(f"❌ Error running tau-bench variant '{variant_name}': {e}")
        logger.error(f"Exception details: {traceback.format_exc()}")
        summary["status"] = "failed"
        summary["exit_reason"] = "exception"
        summary["notes"].append(str(e))
        return False
    finally:
        end_ts = time.time()
        summary["finished_at"] = datetime.utcnow().isoformat() + "Z"
        summary["elapsed_seconds"] = round(end_ts - start_ts, 3)

        # Reset emotion selection to avoid bleed-over
        try:
            set_emotion_variant(None)
        except Exception:
            pass

        # Heuristic: if directory has no files beyond summary itself, flag it
        try:
            existing_files = [
                p for p in variant_output_dir.glob("**/*") if p.is_file()
            ]
            # Exclude the summary file we're about to write
            meaningful_files = [
                p for p in existing_files if p.name != "variant_summary.json"
            ]
            if summary["status"] == "succeeded" and len(meaningful_files) == 0:
                summary["notes"].append("No output files found in output_dir after success")
                summary["exit_reason"] = summary["exit_reason"] or "no_output_files"
        except Exception as e:
            summary["notes"].append(f"Post-run file scan failed: {e}")

        write_summary()

def wait_for_vllm_server(endpoint_url: str, max_wait: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    import requests
    
    logger.info(f"Waiting for vLLM server at {endpoint_url} to be ready...")
    
    for attempt in range(max_wait // 10):
        try:
            response = requests.get(f"{endpoint_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✅ vLLM server ready! Available models: {[m['id'] for m in models['data']]}")
                return True
        except Exception as e:
            logger.debug(f"Server not ready (attempt {attempt + 1}): {e}")
        
        time.sleep(10)
    
    logger.error(f"❌ vLLM server at {endpoint_url} not ready after {max_wait} seconds")
    return False

# =============================================================================
# MAIN WORKER LOOP
# =============================================================================

def run_worker_experiment(config_path: str, worker_id: str) -> None:
    """Main worker function - runs all variants for assigned tasks."""
    
    logger.info(f"Starting tau-bench worker: {worker_id}")
    logger.info(f"Config path: {config_path}")
    
    # Load experiment configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    logger.info(f"Loaded experiment config: {config_data['experiment_name']}")
    
    # Find this worker's configuration
    worker_info = None
    for worker in config_data['workers_info']:
        if worker['worker_id'] == worker_id:
            worker_info = worker
            break
    
    if not worker_info:
        logger.error(f"Worker {worker_id} not found in configuration!")
        sys.exit(1)
    
    endpoint_url = worker_info['endpoint_url']
    task_indices = worker_info['task_indices']
    
    logger.info(f"Worker assigned tasks: {task_indices}")
    logger.info(f"Endpoint URL: {endpoint_url}")
    
    # Wait for vLLM server to be ready
    if not wait_for_vllm_server(endpoint_url):
        logger.error("vLLM server not ready, exiting")
        sys.exit(1)
    
    # Create output directory structure (use remote-relative path)
    experiment_name = config_data['experiment_name'] 
    timestamp = config_data['timestamp']
    remote_output_dir = Path(f"~/tau_bench_results/{experiment_name}_{timestamp}").expanduser()
    output_dir = remote_output_dir
    
    # Create emotional user variants first
    create_emotional_user_variants()
    
    # Get environment from config
    environment = config_data['environment']
    
    # Map variants to user strategies 
    variant_config_map = {
        "control": {"user_strategy": "llm", "environment": environment},
        "frustration": {"user_strategy": "frustrated_llm", "environment": environment},
        "anxiety": {"user_strategy": "anxious_llm", "environment": environment},
        "anger": {"user_strategy": "angry_llm", "environment": environment},
        "confusion": {"user_strategy": "confused_llm", "environment": environment},
    }
    
    # Process each variant
    total_variants = len(config_data['variants'])
    success_count = 0
    
    for i, variant in enumerate(config_data['variants'], 1):
        logger.info(f"Processing variant {i}/{total_variants}: {variant}")
        
        if variant not in variant_config_map:
            logger.error(f"Unknown variant: {variant}")
            continue
        
        variant_config = variant_config_map[variant]
        user_strategy = variant_config["user_strategy"]
        variant_environment = variant_config["environment"]
        
        # Run tau-bench for this variant
        success = run_tau_bench_variant(
            variant_name=variant,
            user_strategy=user_strategy,
            environment=variant_environment,
            task_ids=task_indices,
            endpoint_url=endpoint_url,
            output_dir=output_dir
        )
        
        if success:
            success_count += 1
        
        logger.info(f"Completed variant {variant} ({'✅ SUCCESS' if success else '❌ FAILED'})")
        logger.info(f"Progress: {i}/{total_variants} variants processed")
    
    # Final summary
    logger.info(f"🎉 Worker {worker_id} completed!")
    logger.info(f"📊 Success rate: {success_count}/{total_variants} variants")
    logger.info(f"📁 Results saved to: {output_dir}")
    
    if success_count == total_variants:
        logger.info("✅ All variants completed successfully")
    else:
        logger.warning(f"⚠️ {total_variants - success_count} variants failed")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python worker_experiment.py <config_path> <worker_id>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    worker_id = sys.argv[2]
    
    # Setup logging
    setup_logging()
    
    try:
        run_worker_experiment(config_path, worker_id)
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)
