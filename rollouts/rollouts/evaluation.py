"""
Minimal, composable evaluation framework for rollouts.

Design mirrors run_agent/run_agent_step for easy parallelization.
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Iterator, Callable, Awaitable, Tuple
from datetime import datetime

from .dtypes import (
    Trajectory, Message, Endpoint, Actor, AgentState, 
    RunConfig, Environment
)
from .agents import run_agent


@dataclass
class EvalSample:
    """A single evaluation sample with its result."""
    sample_id: str
    input_data: Dict[str, Any]
    trajectory: Trajectory
    agent_states: List[AgentState]  # Full list of agent states from run_agent
    metrics: Dict[str, float]  # All metrics are floats for RL compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        from dataclasses import asdict
        data = asdict(self)
        data['trajectory'] = json.loads(self.trajectory.to_json())
        # Serialize agent states using asdict (they're regular dataclasses)
        data['agent_states'] = [asdict(state) for state in self.agent_states]
        # Sanitize all API keys recursively
        data = sanitize_api_keys(data)
        return json.dumps(data, indent=2, default=str)  # default=str handles datetime objects
    
    @staticmethod
    def from_json(json_str: str) -> 'EvalSample':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        data['trajectory'] = Trajectory.from_json(json.dumps(data['trajectory']))
        # Note: Full AgentState deserialization would require complex reconstruction
        # For now, store as simplified data for analysis
        data['agent_states'] = data.get('agent_states', [])
        return EvalSample(**data)


@dataclass
class EvalReport:
    """Summary report for an evaluation run."""
    eval_name: str
    dataset_path: str
    total_samples: int
    summary_metrics: Dict[str, float]
    sample_results: List[EvalSample]
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def save(self, output_dir: Path) -> None:
        """Save evaluation results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual samples
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        for sample in self.sample_results:
            sample_file = samples_dir / f"{sample.sample_id}.json"
            sample_file.write_text(sample.to_json())
        
        # Save summary report
        summary = {
            "eval_name": self.eval_name,
            "dataset_path": self.dataset_path,
            "total_samples": self.total_samples,
            "summary_metrics": self.summary_metrics,
            "config": self.config,
            "timestamp": self.timestamp,
            "sample_ids": [s.sample_id for s in self.sample_results]
        }
        # Sanitize API keys in the summary before saving
        summary = sanitize_api_keys(summary)
        report_file = output_dir / "report.json"
        report_file.write_text(json.dumps(summary, indent=2))
        
        # Save trajectories separately for easy loading
        trajectories_dir = output_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        for sample in self.sample_results:
            traj_file = trajectories_dir / f"{sample.sample_id}.jsonl"
            Trajectory.save_jsonl([sample.trajectory], str(traj_file))
        
        # Save agent states separately for detailed analysis
        states_dir = output_dir / "agent_states"
        states_dir.mkdir(exist_ok=True)
        for sample in self.sample_results:
            states_file = states_dir / f"{sample.sample_id}.json"
            from dataclasses import asdict
            states_data = [asdict(state) for state in sample.agent_states]
            # Sanitize all API keys recursively
            states_data = sanitize_api_keys(states_data)
            states_file.write_text(json.dumps(states_data, indent=2, default=str))
        
        print(f"✅ Saved evaluation to {output_dir}")
        print(f"   📊 Summary: {report_file}")
        print(f"   📁 Samples: {samples_dir}")
        print(f"   🎯 Trajectories: {trajectories_dir}")
        print(f"   🔧 Agent States: {states_dir}")


# Type for reward/metric functions
RewardFunction = Callable[[Trajectory, Dict[str, Any]], float]


def sanitize_api_keys(data: Any) -> Any:
    """Recursively sanitize API keys from nested data structures."""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key == "api_key" and isinstance(value, str) and value.startswith("sk-"):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = sanitize_api_keys(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_api_keys(item) for item in data]
    else:
        return data


async def evaluate_sample(
    sample_data: Dict[str, Any],
    sample_id: str,
    prepare_messages: Callable[[Dict[str, Any]], List[Message]],
    reward_functions: List[Tuple[str, RewardFunction]],
    environment: Environment,
    endpoint: Endpoint,
    run_config: RunConfig,
    max_turns: int = 10,
    verbose: bool = True
) -> EvalSample:
    """
    Evaluate a single sample - analogous to run_agent_step.

    This is the atomic unit of evaluation that can be easily parallelized.
    Each call should receive a fresh environment instance to ensure state isolation.

    Args:
        sample_data: The raw sample data
        sample_id: Unique identifier for this sample
        prepare_messages: Function to create initial messages
        reward_functions: List of (name, function) pairs for computing metrics
        environment: Fresh Environment instance for this sample (typically created
                    by the environment_factory in evaluate())
        endpoint: LLM endpoint
        run_config: Agent run configuration
        max_turns: Maximum turns allowed
        verbose: Whether to print progress

    Returns:
        EvalSample with trajectory and computed metrics
    """
    if verbose:
        print(f"📝 Evaluating {sample_id}")
    
    # Prepare initial state
    initial_messages = prepare_messages(sample_data)
    initial_trajectory = Trajectory(messages=initial_messages)
    
    actor = Actor(
        trajectory=initial_trajectory,
        endpoint=endpoint,
        tools=environment.get_tools()
    )
    
    initial_state = AgentState(
        actor=actor,
        environment=environment,
        max_turns=max_turns
    )
    
    # Run agent
    states = await run_agent(initial_state, run_config)
    final_trajectory = states[-1].actor.trajectory
    
    # Compute all metrics/rewards
    metrics = {}
    for metric_name, reward_fn in reward_functions:
        try:
            metrics[metric_name] = reward_fn(final_trajectory, sample_data)
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error computing {metric_name}: {e}")
            metrics[metric_name] = 0.0
    
    # Add metadata
    metadata = {
        "turns_used": states[-1].turn_idx,
        "stop_reason": str(states[-1].stop) if states[-1].stop else None,
        "total_tokens": sum(len(m.content or "") for m in final_trajectory.messages)
    }
    
    if verbose and metrics:
        # Print key metrics inline
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in list(metrics.items())[:3])
        print(f"   Metrics: {metric_str}")
    
    return EvalSample(
        sample_id=sample_id,
        input_data=sample_data,
        trajectory=final_trajectory,
        agent_states=states,  # Store full list of agent states
        metrics=metrics,
        metadata=metadata
    )


async def evaluate(
    dataset: Iterator[Dict[str, Any]],
    prepare_messages: Callable[[Dict[str, Any]], List[Message]],
    reward_functions: List[Tuple[str, RewardFunction]],
    environment_factory: Callable[[], Environment],
    endpoint: Endpoint,
    run_config: Optional[RunConfig] = None,
    eval_name: str = "evaluation",
    dataset_path: str = "unknown",
    max_turns: int = 10,
    max_samples: Optional[int] = None,
    max_concurrent: int = 1,  # For parallel evaluation
    output_dir: Optional[Path] = None,
    sample_id_fn: Optional[Callable[[int, Dict[str, Any]], str]] = None,
    verbose: bool = True
) -> EvalReport:
    """
    Run evaluation on a dataset - analogous to run_agent.

    This orchestrates evaluate_sample calls, potentially in parallel.
    Each sample gets a fresh environment instance to ensure state isolation.

    Args:
        dataset: Iterator of sample dictionaries
        prepare_messages: Function to create initial messages from sample
        reward_functions: List of (name, function) pairs for computing metrics
        environment_factory: Factory function that returns a fresh Environment
                           instance for each sample. Example: `lambda: CalculatorEnvironment()`
                           This ensures each sample has isolated environment state.
        endpoint: LLM endpoint configuration
        run_config: Run configuration for agent (default: silent)
        eval_name: Name for this evaluation
        dataset_path: Path to dataset (for reporting)
        max_turns: Maximum turns per sample
        max_samples: Limit number of samples to evaluate
        max_concurrent: Maximum concurrent evaluations (for parallelization)
        output_dir: Directory to save results
        sample_id_fn: Function to generate sample IDs
        verbose: Whether to print progress
    """
    
    # Default run config (silent)
    if run_config is None:
        run_config = RunConfig(
            on_chunk=lambda _: asyncio.sleep(0)
        )
    
    # Default sample ID function
    if sample_id_fn is None:
        sample_id_fn = lambda i, _: f"sample_{i:04d}"
    
    # Collect samples to evaluate
    samples_to_eval = []
    for i, sample_data in enumerate(dataset):
        if max_samples and len(samples_to_eval) >= max_samples:
            break
        sample_id = sample_id_fn(i, sample_data)
        samples_to_eval.append((sample_id, sample_data))
    
    if verbose:
        print(f"\n🎯 Starting evaluation: {eval_name}")
        print(f"📊 Samples to evaluate: {len(samples_to_eval)}")
        print(f"🔧 Max concurrent: {max_concurrent}")
        print(f"{'='*50}")
    
    # Evaluate samples (with concurrency control)
    results = []
    
    if max_concurrent == 1:
        # Sequential evaluation - create fresh environment for each sample
        for sample_id, sample_data in samples_to_eval:
            env = environment_factory()
            result = await evaluate_sample(
                sample_data=sample_data,
                sample_id=sample_id,
                prepare_messages=prepare_messages,
                reward_functions=reward_functions,
                environment=env,
                endpoint=endpoint,
                run_config=run_config,
                max_turns=max_turns,
                verbose=verbose
            )
            results.append(result)
    else:
        # Parallel evaluation with semaphore - create fresh environment for each sample
        semaphore = asyncio.Semaphore(max_concurrent)

        async def eval_with_semaphore(sample_id: str, sample_data: Dict[str, Any]) -> EvalSample:
            async with semaphore:
                env = environment_factory()
                return await evaluate_sample(
                    sample_data=sample_data,
                    sample_id=sample_id,
                    prepare_messages=prepare_messages,
                    reward_functions=reward_functions,
                    environment=env,
                    endpoint=endpoint,
                    run_config=run_config,
                    max_turns=max_turns,
                    verbose=verbose
                )
        
        # Create all tasks
        tasks = [
            eval_with_semaphore(sample_id, sample_data)
            for sample_id, sample_data in samples_to_eval
        ]
        
        # Run in parallel
        results = await asyncio.gather(*tasks)
    
    # Compute summary metrics
    summary_metrics = compute_summary_metrics(results, reward_functions)
    
    # Create report
    # Sanitize endpoint config to exclude sensitive data
    endpoint_config = sanitize_api_keys(asdict(endpoint))
    
    report = EvalReport(
        eval_name=eval_name,
        dataset_path=dataset_path,
        total_samples=len(results),
        summary_metrics=summary_metrics,
        sample_results=results,
        config={
            "endpoint": endpoint_config,
            "max_turns": max_turns,
            "max_samples": max_samples,
            "max_concurrent": max_concurrent,
            "reward_functions": [name for name, _ in reward_functions],
            "evaluation_timestamp": datetime.now().isoformat(),
            "dataset_format": "auto-detected" if hasattr(dataset, '__iter__') else "unknown"
        }
    )
    
    # Save if output directory specified
    if output_dir:
        report.save(Path(output_dir))
    
    # Print summary
    if verbose:
        print(f"\n{'='*50}")
        print(f"📊 Evaluation Summary: {eval_name}")
        print(f"{'='*50}")
        print(f"Samples evaluated: {len(results)}")
        for key, value in summary_metrics.items():
            print(f"{key}: {value:.3f}")
    
    return report


def compute_summary_metrics(
    results: List[EvalSample], 
    reward_functions: List[Tuple[str, RewardFunction]]
) -> Dict[str, float]:
    """Compute summary statistics from results."""
    if not results:
        return {}
    
    summary = {}
    
    # Compute mean, min, max for each metric
    for metric_name, _ in reward_functions:
        values = [r.metrics.get(metric_name, 0.0) for r in results]
        if values:
            summary[f"mean_{metric_name}"] = sum(values) / len(values)
            summary[f"min_{metric_name}"] = min(values)
            summary[f"max_{metric_name}"] = max(values)
            summary[f"std_{metric_name}"] = (
                sum((v - summary[f"mean_{metric_name}"]) ** 2 for v in values) / len(values)
            ) ** 0.5
    
    # Add metadata summaries
    summary["total_samples"] = len(results)
    summary["avg_turns"] = sum(r.metadata.get("turns_used", 0) for r in results) / len(results)
    summary["avg_tokens"] = sum(r.metadata.get("total_tokens", 0) for r in results) / len(results)
    
    # Compute combined reward if multiple metrics (useful for RL)
    if len(reward_functions) > 1:
        total_rewards = [sum(r.metrics.values()) for r in results]
        summary["mean_total_reward"] = sum(total_rewards) / len(total_rewards)
    
    return summary


# Dataset loaders
def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_csv(path: Path) -> Iterator[Dict[str, Any]]:
    """Load CSV dataset."""
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


# Convenience function for simple evaluation
async def simple_evaluate(
    dataset_path: Path,
    prepare_messages: Callable[[Dict[str, Any]], List[Message]],
    reward_functions: List[Tuple[str, RewardFunction]],
    environment_factory: Callable[[], Environment],
    endpoint: Endpoint,
    **kwargs
) -> EvalReport:
    """
    Simple evaluation interface for common cases.

    Auto-detects dataset format and provides sensible defaults.

    Args:
        dataset_path: Path to dataset file (.jsonl or .csv)
        prepare_messages: Function to create initial messages from sample
        reward_functions: List of (name, function) pairs for computing metrics
        environment_factory: Factory function returning fresh Environment instances
        endpoint: LLM endpoint configuration
        **kwargs: Additional arguments passed to evaluate()
    """

    # Auto-detect dataset format
    if dataset_path.suffix == ".jsonl":
        dataset = load_jsonl(dataset_path)
    elif dataset_path.suffix == ".csv":
        dataset = load_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {dataset_path.suffix}")

    # Default eval name from dataset
    eval_name = kwargs.pop("eval_name", dataset_path.stem)

    return await evaluate(
        dataset=dataset,
        prepare_messages=prepare_messages,
        reward_functions=reward_functions,
        environment_factory=environment_factory,
        endpoint=endpoint,
        eval_name=eval_name,
        dataset_path=str(dataset_path),
        **kwargs
    )