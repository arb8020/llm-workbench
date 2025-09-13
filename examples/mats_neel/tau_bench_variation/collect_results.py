#!/usr/bin/env python3
"""
Collect Results from Tau-Bench User Variation Experiment

Downloads results from all workers and aggregates them for analysis.

Usage:
    python collect_results.py --experiment-name "my_experiment" --timestamp "20250912_135102"
    python collect_results.py --results-dir results/my_experiment_20250912_135102/
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Individual task result from tau-bench."""
    task_id: int
    variant: str
    environment: str
    reward: float
    success: bool
    conversation_log: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExperimentResults:
    """Aggregated results from entire experiment."""
    experiment_name: str
    timestamp: str
    total_tasks: int
    variants: List[str]
    environment: str
    task_results: List[TaskResult]
    summary_stats: Dict[str, Any]

def load_experiment_config(results_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration from results directory."""
    config_path = results_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def collect_worker_results(worker_info: Dict[str, Any], results_dir: Path, 
                          experiment_name: str, timestamp: str) -> List[TaskResult]:
    """Collect results from a single worker via SSH."""
    
    worker_id = worker_info['worker_id']
    ssh_connection = worker_info['ssh_connection']
    task_indices = worker_info['task_indices']
    
    logger.info(f"Collecting results from {worker_id} ({ssh_connection})")
    
    # Remote path where results should be stored
    remote_results_dir = f"~/tau_bench_results/{experiment_name}_{timestamp}"
    
    # Local directory for this worker's results  
    local_worker_dir = results_dir / worker_id
    local_worker_dir.mkdir(parents=True, exist_ok=True)
    
    task_results = []
    
    try:
        # Use rsync to download all results from worker
        rsync_cmd = f"rsync -avz -e 'ssh -o StrictHostKeyChecking=no' {ssh_connection}:{remote_results_dir}/ {local_worker_dir}/"
        logger.info(f"Downloading worker results: {rsync_cmd}")
        
        result = os.system(rsync_cmd)
        if result != 0:
            logger.warning(f"rsync failed for {worker_id} (exit code {result})")
            return task_results
        
        logger.info(f"✅ Downloaded results from {worker_id}")
        
        # Process downloaded results for each variant
        for variant_dir in local_worker_dir.iterdir():
            if not variant_dir.is_dir():
                continue
                
            variant_name = variant_dir.name
            logger.info(f"Processing variant: {variant_name}")
            
            # Look for tau-bench result files
            for result_file in variant_dir.rglob("*.json"):
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    
                    # Extract task results (format may vary)
                    if isinstance(result_data, list):
                        # Multiple task results
                        for task_data in result_data:
                            task_result = parse_task_result(task_data, variant_name, worker_info)
                            if task_result:
                                task_results.append(task_result)
                    elif isinstance(result_data, dict):
                        # Single task result
                        task_result = parse_task_result(result_data, variant_name, worker_info)
                        if task_result:
                            task_results.append(task_result)
                            
                except Exception as e:
                    logger.warning(f"Failed to parse result file {result_file}: {e}")
                    continue
            
            # Look for conversation logs
            for log_file in variant_dir.rglob("*.log"):
                # Associate logs with task results
                logger.info(f"Found conversation log: {log_file}")
                
    except Exception as e:
        logger.error(f"Failed to collect results from {worker_id}: {e}")
    
    return task_results

def parse_task_result(result_data: Dict[str, Any], variant: str, worker_info: Dict[str, Any]) -> Optional[TaskResult]:
    """Parse individual task result from tau-bench output."""
    
    try:
        # Extract basic task information
        task_id = result_data.get('task_id')
        reward = result_data.get('reward', 0.0)
        
        if task_id is None:
            logger.warning(f"Task result missing task_id: {result_data}")
            return None
        
        # Determine success based on reward (tau-bench specific logic)
        success = reward > 0.5  # Adjust threshold as needed
        
        # Get environment from worker info
        environment = "unknown"  # Will be filled from experiment config
        
        return TaskResult(
            task_id=task_id,
            variant=variant,
            environment=environment,
            reward=reward,
            success=success,
            metadata=result_data
        )
        
    except Exception as e:
        logger.warning(f"Failed to parse task result: {e}")
        return None

def calculate_summary_stats(task_results: List[TaskResult], variants: List[str]) -> Dict[str, Any]:
    """Calculate summary statistics across all results."""
    
    stats = {
        'total_tasks': len(task_results),
        'by_variant': {},
        'overall': {}
    }
    
    # Overall statistics
    if task_results:
        rewards = [r.reward for r in task_results]
        successes = [r.success for r in task_results]
        
        stats['overall'] = {
            'mean_reward': sum(rewards) / len(rewards),
            'success_rate': sum(successes) / len(successes),
            'max_reward': max(rewards),
            'min_reward': min(rewards)
        }
    
    # Per-variant statistics
    for variant in variants:
        variant_results = [r for r in task_results if r.variant == variant]
        
        if variant_results:
            rewards = [r.reward for r in variant_results]
            successes = [r.success for r in variant_results]
            
            stats['by_variant'][variant] = {
                'count': len(variant_results),
                'mean_reward': sum(rewards) / len(rewards),
                'success_rate': sum(successes) / len(successes),
                'max_reward': max(rewards),
                'min_reward': min(rewards)
            }
        else:
            stats['by_variant'][variant] = {
                'count': 0,
                'mean_reward': 0.0,
                'success_rate': 0.0,
                'max_reward': 0.0,
                'min_reward': 0.0
            }
    
    return stats

def collect_experiment_results(results_dir: Path) -> ExperimentResults:
    """Collect and aggregate results from entire experiment."""
    
    logger.info(f"Collecting results from: {results_dir}")
    
    # Load experiment configuration
    config = load_experiment_config(results_dir)
    experiment_name = config['experiment_name']
    timestamp = config['timestamp']
    variants = config['variants']
    environment = config['environment']
    
    logger.info(f"Experiment: {experiment_name} ({timestamp})")
    logger.info(f"Variants: {variants}")
    logger.info(f"Environment: {environment}")
    
    # Collect results from all workers
    all_task_results = []
    
    for worker_info in config['workers_info']:
        worker_results = collect_worker_results(worker_info, results_dir, 
                                               experiment_name, timestamp)
        
        # Update environment info for each task result
        for result in worker_results:
            result.environment = environment
            
        all_task_results.extend(worker_results)
    
    logger.info(f"Collected {len(all_task_results)} task results total")
    
    # Calculate summary statistics
    summary_stats = calculate_summary_stats(all_task_results, variants)
    
    # Create aggregated results
    results = ExperimentResults(
        experiment_name=experiment_name,
        timestamp=timestamp,
        total_tasks=len(all_task_results),
        variants=variants,
        environment=environment,
        task_results=all_task_results,
        summary_stats=summary_stats
    )
    
    # Save aggregated results
    output_file = results_dir / "aggregated_results.json"
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        results_dict = {
            'experiment_name': results.experiment_name,
            'timestamp': results.timestamp,
            'total_tasks': results.total_tasks,
            'variants': results.variants,
            'environment': results.environment,
            'summary_stats': results.summary_stats,
            'task_results': [
                {
                    'task_id': r.task_id,
                    'variant': r.variant,
                    'environment': r.environment,
                    'reward': r.reward,
                    'success': r.success,
                    'metadata': r.metadata
                }
                for r in results.task_results
            ]
        }
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"✅ Aggregated results saved to: {output_file}")
    
    return results

def print_results_summary(results: ExperimentResults):
    """Print a human-readable summary of results."""
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT RESULTS: {results.experiment_name}")
    print(f"{'='*60}")
    print(f"Timestamp: {results.timestamp}")
    print(f"Environment: {results.environment}")
    print(f"Total Tasks: {results.total_tasks}")
    print(f"Variants: {', '.join(results.variants)}")
    
    print(f"\n📊 OVERALL STATISTICS:")
    overall = results.summary_stats['overall']
    print(f"  Mean Reward: {overall.get('mean_reward', 0):.3f}")
    print(f"  Success Rate: {overall.get('success_rate', 0):.1%}")
    print(f"  Reward Range: {overall.get('min_reward', 0):.3f} - {overall.get('max_reward', 0):.3f}")
    
    print(f"\n📈 BY VARIANT:")
    for variant in results.variants:
        stats = results.summary_stats['by_variant'][variant]
        print(f"  {variant.upper()}:")
        print(f"    Tasks: {stats['count']}")
        print(f"    Mean Reward: {stats['mean_reward']:.3f}")
        print(f"    Success Rate: {stats['success_rate']:.1%}")
        print(f"    Reward Range: {stats['min_reward']:.3f} - {stats['max_reward']:.3f}")
        print()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect tau-bench experiment results")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-name", type=str, 
                      help="Experiment name")
    group.add_argument("--results-dir", type=Path,
                      help="Path to results directory")
    
    parser.add_argument("--timestamp", type=str,
                       help="Experiment timestamp (required with --experiment-name)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        if not args.timestamp:
            logger.error("--timestamp is required when using --experiment-name")
            sys.exit(1)
        results_dir = Path(__file__).parent / "results" / f"{args.experiment_name}_{args.timestamp}"
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    try:
        # Collect and aggregate results
        results = collect_experiment_results(results_dir)
        
        # Print summary
        print_results_summary(results)
        
        logger.info("🎉 Result collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to collect results: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()