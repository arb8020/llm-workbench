#!/usr/bin/env python3
"""
Analysis script for prompt variation experiment results.

Usage:
    python analyze_results.py results/emotional_pilot_20250911_123456/
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

def load_experiment_results(experiment_dir: Path) -> Dict[str, Any]:
    """Load all results from an experiment directory."""
    
    # Load metadata
    metadata_path = experiment_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {experiment_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load results for each variant
    variant_results = {}
    
    for variant in metadata["variants"]:
        variant_dir = experiment_dir / variant
        if not variant_dir.exists():
            print(f"⚠️  Warning: No results directory for variant '{variant}'")
            continue
            
        variant_results[variant] = []
        
        # Load all sample results for this variant
        for sample_dir in variant_dir.iterdir():
            if sample_dir.is_dir():
                sample_path = sample_dir / "sample.json"
                if sample_path.exists():
                    with open(sample_path, 'r') as f:
                        sample_result = json.load(f)
                    variant_results[variant].append(sample_result)
    
    return {
        "metadata": metadata,
        "results": variant_results
    }

def analyze_performance(experiment_data: Dict[str, Any]) -> None:
    """Analyze performance across variants."""
    
    print("🔍 Performance Analysis")
    print("=" * 50)
    
    results = experiment_data["results"]
    metadata = experiment_data["metadata"]
    
    # Calculate metrics per variant
    variant_metrics = {}
    
    for variant, samples in results.items():
        if not samples:
            continue
            
        correctness_scores = [s["metrics"].get("correctness", 0.0) for s in samples]
        format_scores = [s["metrics"].get("format", 0.0) for s in samples]
        efficiency_scores = [s["metrics"].get("efficiency", 0.0) for s in samples]
        
        variant_metrics[variant] = {
            "samples": len(samples),
            "correctness": {
                "mean": sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0,
                "correct_count": sum(1 for score in correctness_scores if score > 0.5),
                "accuracy": sum(1 for score in correctness_scores if score > 0.5) / len(correctness_scores) if correctness_scores else 0.0
            },
            "format": {
                "mean": sum(format_scores) / len(format_scores) if format_scores else 0.0,
            },
            "efficiency": {
                "mean": sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0,
            }
        }
    
    # Display results
    print(f"Experiment: {metadata['experiment_name']}")
    print(f"Model: {metadata['model']}")
    print(f"Total samples per variant: {metadata['samples']}")
    print()
    
    print("📊 Results by Variant:")
    print()
    
    # Sort variants for consistent display
    sorted_variants = sorted(variant_metrics.keys())
    
    for variant in sorted_variants:
        metrics = variant_metrics[variant]
        correctness = metrics["correctness"]
        
        status_icon = "✅" if correctness["accuracy"] > 0.5 else "❌" if correctness["accuracy"] == 0 else "⚠️"
        
        print(f"{status_icon} {variant.upper()}")
        print(f"   Accuracy: {correctness['accuracy']:.3f} ({correctness['correct_count']}/{metrics['samples']})")
        print(f"   Format:   {metrics['format']['mean']:.3f}")
        print(f"   Efficiency: {metrics['efficiency']['mean']:.3f}")
        print()
    
    # Comparison with control
    if "control" in variant_metrics:
        control_accuracy = variant_metrics["control"]["correctness"]["accuracy"]
        
        print("🔄 Comparison vs Control:")
        print()
        
        for variant in sorted_variants:
            if variant == "control":
                continue
                
            metrics = variant_metrics[variant]
            accuracy = metrics["correctness"]["accuracy"]
            
            diff = accuracy - control_accuracy
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            
            print(f"   {variant}: {accuracy:.3f} ({diff:+.3f}) {arrow}")
        
        print()

def analyze_sample_consistency(experiment_data: Dict[str, Any]) -> None:
    """Analyze how different variants perform on the same samples."""
    
    print("📋 Sample-by-Sample Analysis")
    print("=" * 50)
    
    results = experiment_data["results"]
    
    # Group by sample_id
    sample_performance = defaultdict(dict)
    
    for variant, samples in results.items():
        for sample in samples:
            sample_id = sample["sample_id"]
            correctness = sample["metrics"].get("correctness", 0.0)
            sample_performance[sample_id][variant] = correctness > 0.5  # Convert to boolean
    
    # Display results
    sample_ids = sorted(sample_performance.keys())
    variants = sorted(results.keys())
    
    print("Sample-by-sample correctness (✓ = correct, ✗ = incorrect):")
    print()
    
    # Header
    print(f"{'Sample':<12}", end="")
    for variant in variants:
        print(f"{variant:<12}", end="")
    print()
    print("-" * (12 + 12 * len(variants)))
    
    # Data rows
    for sample_id in sample_ids:
        print(f"{sample_id:<12}", end="")
        for variant in variants:
            if variant in sample_performance[sample_id]:
                symbol = "✓" if sample_performance[sample_id][variant] else "✗"
                print(f"{symbol:<12}", end="")
            else:
                print(f"{'?':<12}", end="")
        print()
    
    print()
    
    # Identify interesting patterns
    consistent_successes = []
    consistent_failures = []
    variant_specific = defaultdict(list)
    
    for sample_id in sample_ids:
        sample_results = sample_performance[sample_id]
        
        if len(sample_results) < len(variants):
            continue  # Skip incomplete samples
        
        correct_variants = [v for v, correct in sample_results.items() if correct]
        
        if len(correct_variants) == len(variants):
            consistent_successes.append(sample_id)
        elif len(correct_variants) == 0:
            consistent_failures.append(sample_id)
        elif len(correct_variants) == 1:
            variant_specific[correct_variants[0]].append(sample_id)
    
    print("🎯 Pattern Analysis:")
    print()
    
    if consistent_successes:
        print(f"   All variants succeed: {len(consistent_successes)} samples")
        print(f"   Samples: {', '.join(consistent_successes)}")
        print()
    
    if consistent_failures:
        print(f"   All variants fail: {len(consistent_failures)} samples")
        print(f"   Samples: {', '.join(consistent_failures)}")
        print()
    
    for variant, samples in variant_specific.items():
        if samples:
            print(f"   Only {variant} succeeds: {len(samples)} samples")
            print(f"   Samples: {', '.join(samples)}")
            print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    
    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    try:
        # Load experiment data
        print(f"📂 Loading results from: {experiment_dir}")
        experiment_data = load_experiment_results(experiment_dir)
        
        print(f"✅ Loaded {len(experiment_data['results'])} variant(s)")
        print()
        
        # Run analysis
        analyze_performance(experiment_data)
        analyze_sample_consistency(experiment_data)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()