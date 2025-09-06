#!/usr/bin/env python3
"""
Wrapper script to run outlier analysis across different model sizes.

This script:
1. Finds appropriate GPU instances for each model size
2. Runs full analysis on each model
3. Extracts key summary statistics (% layers with outliers)
4. Syncs back lightweight results (not massive .pt files)

Usage:
    python run_model_comparison.py --models small medium large
    python run_model_comparison.py --config model_configs.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelConfig:
    """Configuration for a model to analyze."""
    name: str
    model_id: str
    min_vram_gb: int
    max_price_per_hour: float
    sequence_length: int = 2048
    num_sequences: int = 4
    batch_size: int = 1

# Predefined model configurations - verified architectures with same layer access patterns
DEFAULT_MODELS = {
    "olmoe": ModelConfig(
        name="olmoe",
        model_id="allenai/OLMoE-1B-7B-0125-Instruct",  # 16 layers, 1B params
        min_vram_gb=16,
        max_price_per_hour=0.60,
    ),
    "qwen3": ModelConfig(
        name="qwen3", 
        model_id="Qwen/Qwen3-30B-A3B",  # 48 layers, 30B params (3B active)
        min_vram_gb=24,
        max_price_per_hour=1.20,
    ),
    "gpt-oss": ModelConfig(
        name="gpt-oss",
        model_id="openai/gpt-oss-120b",  # 36 layers, 120B params (5B active)
        min_vram_gb=80,
        max_price_per_hour=2.00,
    ),
    "glm": ModelConfig(
        name="glm",
        model_id="zai-org/GLM-4.5-Air",  # 46 layers, 106B params (12B active)
        min_vram_gb=48,
        max_price_per_hour=1.80,
    )
}

def run_command(cmd: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a shell command with timeout."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"Command failed: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s")
        raise

def find_or_create_gpu(config: ModelConfig) -> str:
    """Find existing GPU or create new one with required specs."""
    print(f"\n🔍 Finding GPU for {config.name} (min {config.min_vram_gb}GB VRAM, max ${config.max_price_per_hour}/hr)")
    
    # First, try to find existing instance
    try:
        result = run_command(["broker", "instances", "list", "--simple"])
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 5:
                    instance_id, name, status, gpu, price = parts[:5]
                    if status == "running":
                        print(f"Found existing GPU: {instance_id} ({gpu}, ${price}/hr)")
                        return instance_id
    except Exception as e:
        print(f"Error checking existing instances: {e}")
    
    # Create new instance
    print(f"Creating new GPU instance...")
    create_cmd = [
        "broker", "create",
        "--name", f"outlier-{config.name}",
        "--min-vram", str(config.min_vram_gb),
        "--max-price", str(config.max_price_per_hour)
    ]
    
    try:
        result = run_command(create_cmd, timeout=600)  # 10 min timeout for creation
        # Parse instance ID from output
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if "Instance created:" in line:
                instance_id = line.split(":")[1].strip()
                print(f"✅ Created GPU instance: {instance_id}")
                
                # Wait for SSH to be ready
                print("Waiting for SSH to be ready...")
                time.sleep(60)  # Initial wait
                
                for attempt in range(10):
                    try:
                        run_command(["broker", "instances", "status", instance_id])
                        print("✅ SSH ready")
                        return instance_id
                    except Exception:
                        print(f"SSH not ready, attempt {attempt + 1}/10, waiting...")
                        time.sleep(30)
                
                raise Exception("SSH never became ready")
                
        raise Exception("Could not parse instance ID from broker output")
        
    except Exception as e:
        print(f"Failed to create GPU instance: {e}")
        raise

def run_analysis_on_gpu(instance_id: str, config: ModelConfig) -> Dict:
    """Run outlier analysis on GPU instance and return summary stats."""
    print(f"\n🚀 Running analysis for {config.name} on {instance_id}")
    
    # Create remote command
    remote_cmd = (
        f'cd examples/outlier_features_moe && '
        f'tmux new-session -d -s analysis_{config.name} "'
        f'uv run python run_full_analysis.py '
        f'--model {config.model_id} '
        f'--sequence-length {config.sequence_length} '
        f'--num-sequences {config.num_sequences} '
        f'--batch-size {config.batch_size} '
        f'> analysis_{config.name}.log 2>&1"'
    )
    
    # Deploy and run
    deploy_cmd = ["broker", "instances", "list", "--name", f"outlier-{config.name}", "--ssh-only"]
    result = run_command(deploy_cmd)
    ssh_info = result.stdout.strip()
    
    if not ssh_info:
        raise Exception(f"Could not get SSH info for instance {instance_id}")
    
    # Execute analysis
    exec_cmd = ["bifrost", "exec", ssh_info, remote_cmd]
    run_command(exec_cmd, timeout=60)  # Just start the job
    
    # Wait for completion and monitor progress
    print("⏳ Waiting for analysis to complete...")
    
    for attempt in range(60):  # Wait up to 60 * 30s = 30 minutes
        try:
            # Check if analysis is complete
            check_cmd = ["bifrost", "exec", ssh_info, 
                        f'cd examples/outlier_features_moe && '
                        f'if tmux list-sessions 2>/dev/null | grep -q analysis_{config.name}; then echo "RUNNING"; else echo "COMPLETE"; fi']
            
            result = run_command(check_cmd)
            status = result.stdout.strip()
            
            if "COMPLETE" in status:
                print("✅ Analysis complete!")
                break
            else:
                print(f"Still running... (attempt {attempt + 1}/60)")
                time.sleep(30)
                
        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(30)
    else:
        print("⚠️ Analysis timed out, attempting to extract partial results...")
    
    # Extract results summary
    print("📊 Extracting summary statistics...")
    
    extract_cmd = ["bifrost", "exec", ssh_info,
                  f'cd examples/outlier_features_moe && '
                  f'tail -100 analysis_{config.name}.log | grep -E "(systematic outlier|Feature [0-9]+:|layers affected)"']
    
    try:
        result = run_command(extract_cmd)
        log_output = result.stdout
        
        # Parse summary statistics from log
        summary = parse_analysis_results(log_output, config)
        return summary
        
    except Exception as e:
        print(f"Error extracting results: {e}")
        return {"error": str(e), "model": config.name}

def parse_analysis_results(log_output: str, config: ModelConfig) -> Dict:
    """Parse analysis log output to extract key statistics."""
    lines = log_output.strip().split('\n')
    
    summary = {
        "model_name": config.name,
        "model_id": config.model_id,
        "timestamp": datetime.now().isoformat(),
        "sequence_length": config.sequence_length,
        "num_sequences": config.num_sequences,
        "systematic_features": [],
        "total_layers": 0,
        "layers_with_outliers": 0,
        "percentage_layers_affected": 0.0
    }
    
    layers_affected = set()
    
    for line in lines:
        line = line.strip()
        
        # Parse systematic features
        if "Feature" in line and "layers" in line:
            # Extract feature info from lines like:
            # "Feature 292: 25.0% layers, 12.5% seq_pos, max_mag=8.66"
            try:
                parts = line.split(':')
                feature_id = parts[0].split()[-1]
                
                info_parts = parts[1].split(',')
                layer_pct = float(info_parts[0].strip().replace('% layers', ''))
                
                # Extract affected layers count
                # This is approximate - we'd need the actual layer list for precision
                estimated_affected = int(layer_pct / 100 * 16)  # Assuming 16 layers for now
                
                summary["systematic_features"].append({
                    "feature_id": feature_id,
                    "layer_percentage": layer_pct,
                    "estimated_layers_affected": estimated_affected
                })
                
                layers_affected.add(estimated_affected)
                
            except (ValueError, IndexError) as e:
                print(f"Could not parse feature line: {line} ({e})")
    
    # Calculate summary statistics
    if layers_affected:
        summary["total_layers"] = 16  # This should be dynamic based on model
        summary["layers_with_outliers"] = max(layers_affected) if layers_affected else 0
        summary["percentage_layers_affected"] = (summary["layers_with_outliers"] / summary["total_layers"]) * 100
    
    return summary

def cleanup_gpu(instance_id: str, keep_alive: bool = False):
    """Clean up GPU instance."""
    if keep_alive:
        print(f"Keeping GPU instance {instance_id} alive")
        return
        
    print(f"🧹 Cleaning up GPU instance {instance_id}")
    try:
        run_command(["broker", "instances", "terminate", instance_id])
        print("✅ GPU instance terminated")
    except Exception as e:
        print(f"Error terminating instance: {e}")

def is_result_fresh(result: Dict, max_age_hours: int = None) -> bool:
    """Check if result is fresh enough based on timestamp."""
    if max_age_hours is None:
        return True
    
    if 'timestamp' not in result:
        return False
    
    try:
        from datetime import datetime, timezone
        result_time = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        age_hours = (now - result_time).total_seconds() / 3600
        return age_hours <= max_age_hours
    except Exception:
        return False

def is_valid_result(result: Dict) -> bool:
    """Check if a result is complete and valid."""
    required_fields = [
        'model_name', 'model_id', 'timestamp', 
        'percentage_layers_affected', 'layers_with_outliers'
    ]
    
    # Check all required fields exist
    for field in required_fields:
        if field not in result:
            return False
    
    # Check for error field (indicates failed run)
    if 'error' in result:
        return False
    
    # Check that numeric results are reasonable
    if not isinstance(result['percentage_layers_affected'], (int, float)):
        return False
    
    if not isinstance(result['layers_with_outliers'], int):
        return False
        
    return True

def load_existing_results(output_file: str) -> Dict[str, Dict]:
    """Load existing results to avoid re-running completed analyses."""
    if not Path(output_file).exists():
        return {}
    
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        # Convert list to dict keyed by model_id for precise matching
        valid_results = {}
        if isinstance(results, list):
            for r in results:
                if is_valid_result(r):
                    # Use model_id as key for exact matching
                    model_id = r.get('model_id', '')
                    if model_id:
                        valid_results[model_id] = r
                    else:
                        print(f"⚠️ Result missing model_id: {r.get('model_name', 'unknown')}")
                else:
                    print(f"⚠️ Invalid/incomplete result found for: {r.get('model_name', 'unknown')}")
        
        return valid_results
        
    except Exception as e:
        print(f"⚠️ Could not load existing results: {e}")
        return {}

def save_results(results: List[Dict], output_file: str, incremental: bool = True):
    """Save comparison results to JSON file."""
    print(f"💾 Saving results to {output_file}")
    
    if incremental:
        # Load existing results and merge
        existing = load_existing_results(output_file)
        
        # Update with new results
        for result in results:
            if result and 'model_id' in result:
                existing[result['model_id']] = result
        
        # Convert back to list format
        final_results = list(existing.values())
    else:
        final_results = results
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Also print summary table
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Layers Affected':<15} {'% Affected':<12} {'Systematic Features':<20}")
    print("-" * 80)
    
    for result in results:
        if "error" not in result:
            print(f"{result['model_name']:<20} "
                  f"{result['layers_with_outliers']:<15} "
                  f"{result['percentage_layers_affected']:<12.1f} "
                  f"{len(result['systematic_features']):<20}")
    
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Run outlier analysis across different model sizes")
    parser.add_argument("--models", nargs="+", choices=list(DEFAULT_MODELS.keys()),
                       help="Models to analyze")
    parser.add_argument("--config", help="JSON config file with custom models")
    parser.add_argument("--output", default="outlier_comparison.json",
                       help="Output file for results")
    parser.add_argument("--keep-gpus", action="store_true",
                       help="Keep GPU instances running after analysis")
    parser.add_argument("--force-rerun", action="store_true",
                       help="Force re-run even if results already exist")
    parser.add_argument("--incremental", action="store_true", default=True,
                       help="Skip models that already have results (default: true)")
    parser.add_argument("--max-age-hours", type=int, default=None,
                       help="Maximum age of existing results in hours (default: no limit)")
    
    args = parser.parse_args()
    
    # Load model configurations
    if args.config:
        with open(args.config) as f:
            custom_models = json.load(f)
        models = [ModelConfig(**config) for config in custom_models]
    else:
        if not args.models:
            print("Must specify --models or --config")
            sys.exit(1)
        models = [DEFAULT_MODELS[name] for name in args.models]
    
    # Load existing results for incremental mode
    existing_results = {}
    if args.incremental and not args.force_rerun:
        existing_results = load_existing_results(args.output)
        
        # Filter out models that already have results
        models_to_run = []
        for config in models:
            if config.model_id in existing_results:
                existing_result = existing_results[config.model_id]
                
                # Check if result is fresh enough
                if is_result_fresh(existing_result, args.max_age_hours):
                    age_info = ""
                    if args.max_age_hours and 'timestamp' in existing_result:
                        try:
                            from datetime import datetime, timezone
                            result_time = datetime.fromisoformat(existing_result['timestamp'].replace('Z', '+00:00'))
                            age_hours = (datetime.now(timezone.utc) - result_time).total_seconds() / 3600
                            age_info = f" ({age_hours:.1f}h old)"
                        except:
                            pass
                    
                    print(f"✅ Skipping {config.name} ({config.model_id}){age_info}")
                    print(f"   Previous result: {existing_result['percentage_layers_affected']:.1f}% layers affected")
                else:
                    print(f"⏰ Result for {config.name} is too old, will re-run")
                    models_to_run.append(config)
            else:
                models_to_run.append(config)
        
        models = models_to_run
        
        if not models:
            print("🎉 All requested models already have results!")
            # Just print the summary and exit
            existing_list = list(existing_results.values())
            save_results(existing_list, args.output, incremental=False)
            return
    
    print(f"🎯 Running outlier analysis on {len(models)} models")
    
    results = []
    created_instances = []
    
    try:
        for config in models:
            try:
                # Find/create GPU
                instance_id = find_or_create_gpu(config)
                created_instances.append(instance_id)
                
                # Run analysis
                result = run_analysis_on_gpu(instance_id, config)
                results.append(result)
                
                print(f"✅ Completed analysis for {config.name}")
                
            except Exception as e:
                print(f"❌ Failed analysis for {config.name}: {e}")
                results.append({
                    "error": str(e),
                    "model_name": config.name,
                    "model_id": config.model_id
                })
    
    finally:
        # Cleanup
        if not args.keep_gpus:
            for instance_id in created_instances:
                cleanup_gpu(instance_id)
    
    # Save results
    save_results(results, args.output)
    
    print(f"\n🎉 Analysis complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()