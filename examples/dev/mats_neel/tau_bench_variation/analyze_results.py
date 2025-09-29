#!/usr/bin/env python3
"""
Statistical Analysis of Tau-Bench User Variation Results

Performs detailed statistical analysis and comparison of agent performance
across different emotional user states.

Usage:
    python analyze_results.py --results-dir results/my_experiment_20250912_135102/
    python analyze_results.py --compare results/exp1/ results/exp2/ --output comparison.html
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys

# Optional statistical libraries (will gracefully degrade if not available)
try:
    import numpy as np
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("scipy not available, using basic statistics only")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logging.warning("matplotlib/seaborn not available, no plots will be generated")

from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class VariantComparison:
    """Statistical comparison between two variants."""
    variant1: str
    variant2: str
    metric: str
    
    mean1: float
    mean2: float
    std1: float
    std2: float
    
    difference: float  # mean1 - mean2
    effect_size: float  # Cohen's d
    p_value: Optional[float] = None
    significant: Optional[bool] = None
    
    def __str__(self):
        sig_str = ""
        if self.significant is not None:
            sig_str = " (***)" if self.significant else " (n.s.)"
        
        return f"{self.variant1} vs {self.variant2} ({self.metric}): {self.difference:+.3f}{sig_str}"

@dataclass
class ExperimentAnalysis:
    """Complete statistical analysis of experiment results."""
    experiment_name: str
    timestamp: str
    environment: str
    
    # Basic statistics
    variant_stats: Dict[str, Dict[str, float]]
    
    # Comparisons
    pairwise_comparisons: List[VariantComparison]
    
    # Rankings
    variant_rankings: Dict[str, int]  # variant -> rank (1 = best)
    
    # Effect sizes and significance
    significant_differences: List[str]
    
    # Recommendations
    recommendations: List[str]

def load_aggregated_results(results_dir: Path) -> Dict[str, Any]:
    """Load aggregated results from collect_results.py output."""
    
    aggregated_file = results_dir / "aggregated_results.json"
    if not aggregated_file.exists():
        # Try to find any aggregated results file
        possible_files = list(results_dir.glob("*aggregated*.json"))
        if not possible_files:
            raise FileNotFoundError(f"No aggregated results found in {results_dir}. Run collect_results.py first.")
        aggregated_file = possible_files[0]
        logger.info(f"Using aggregated results: {aggregated_file}")
    
    with open(aggregated_file, 'r') as f:
        return json.load(f)

def calculate_detailed_stats(task_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate detailed statistics for each variant."""
    
    # Group results by variant
    by_variant = {}
    for result in task_results:
        variant = result['variant']
        if variant not in by_variant:
            by_variant[variant] = []
        by_variant[variant].append(result)
    
    # Calculate statistics for each variant
    stats = {}
    for variant, results in by_variant.items():
        rewards = [r['reward'] for r in results]
        successes = [r['success'] for r in results]
        
        if rewards:
            reward_stats = {
                'count': len(rewards),
                'mean_reward': statistics.mean(rewards),
                'median_reward': statistics.median(rewards),
                'std_reward': statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
                'min_reward': min(rewards),
                'max_reward': max(rewards),
                'success_rate': sum(successes) / len(successes),
                'q1_reward': np.percentile(rewards, 25) if HAS_SCIPY else min(rewards),
                'q3_reward': np.percentile(rewards, 75) if HAS_SCIPY else max(rewards),
            }
        else:
            reward_stats = {
                'count': 0,
                'mean_reward': 0.0,
                'median_reward': 0.0,
                'std_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'success_rate': 0.0,
                'q1_reward': 0.0,
                'q3_reward': 0.0,
            }
        
        stats[variant] = reward_stats
    
    return stats

def cohen_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    if not group1 or not group2:
        return 0.0
    
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    
    if len(group1) <= 1 and len(group2) <= 1:
        return 0.0
    
    # Pooled standard deviation
    if len(group1) > 1 and len(group2) > 1:
        std1, std2 = statistics.stdev(group1), statistics.stdev(group2)
        n1, n2 = len(group1), len(group2)
        pooled_std = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        pooled_std = pooled_std**0.5
    elif len(group1) > 1:
        pooled_std = statistics.stdev(group1)
    elif len(group2) > 1:
        pooled_std = statistics.stdev(group2)
    else:
        return 0.0
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std

def perform_pairwise_comparisons(task_results: List[Dict[str, Any]], 
                                variants: List[str]) -> List[VariantComparison]:
    """Perform statistical comparisons between all pairs of variants."""
    
    comparisons = []
    
    # Group results by variant
    by_variant = {}
    for result in task_results:
        variant = result['variant']
        if variant not in by_variant:
            by_variant[variant] = []
        by_variant[variant].append(result['reward'])
    
    # Compare each pair of variants
    for i, variant1 in enumerate(variants):
        for variant2 in variants[i+1:]:
            
            group1 = by_variant.get(variant1, [])
            group2 = by_variant.get(variant2, [])
            
            if not group1 or not group2:
                continue
            
            mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
            std1 = statistics.stdev(group1) if len(group1) > 1 else 0.0
            std2 = statistics.stdev(group2) if len(group2) > 1 else 0.0
            
            difference = mean1 - mean2
            effect_size = cohen_d(group1, group2)
            
            # Statistical significance test
            p_value = None
            significant = None
            if HAS_SCIPY and len(group1) > 1 and len(group2) > 1:
                try:
                    t_stat, p_value = scipy.stats.ttest_ind(group1, group2)
                    significant = p_value < 0.05
                except Exception as e:
                    logger.warning(f"Failed to compute t-test for {variant1} vs {variant2}: {e}")
            
            comparison = VariantComparison(
                variant1=variant1,
                variant2=variant2,
                metric="reward",
                mean1=mean1,
                mean2=mean2,
                std1=std1,
                std2=std2,
                difference=difference,
                effect_size=effect_size,
                p_value=p_value,
                significant=significant
            )
            
            comparisons.append(comparison)
    
    # Sort by effect size magnitude
    comparisons.sort(key=lambda x: abs(x.effect_size), reverse=True)
    
    return comparisons

def rank_variants(variant_stats: Dict[str, Dict[str, float]]) -> Dict[str, int]:
    """Rank variants by performance (mean reward)."""
    
    # Sort by mean reward (descending)
    sorted_variants = sorted(variant_stats.items(), 
                           key=lambda x: x[1]['mean_reward'], 
                           reverse=True)
    
    rankings = {}
    for rank, (variant, stats) in enumerate(sorted_variants, 1):
        rankings[variant] = rank
    
    return rankings

def generate_recommendations(analysis: ExperimentAnalysis) -> List[str]:
    """Generate actionable recommendations based on analysis."""
    
    recommendations = []
    
    # Best performing variant
    best_variant = min(analysis.variant_rankings.items(), key=lambda x: x[1])
    recommendations.append(f"🥇 Best performing variant: {best_variant[0].upper()}")
    
    # Significant differences
    sig_differences = [c for c in analysis.pairwise_comparisons if c.significant]
    if sig_differences:
        recommendations.append(f"📊 Found {len(sig_differences)} statistically significant differences")
        
        # Highlight largest effect
        largest_effect = max(sig_differences, key=lambda x: abs(x.effect_size))
        if abs(largest_effect.effect_size) > 0.8:  # Large effect size
            recommendations.append(f"🔍 Large effect size detected: {largest_effect}")
    else:
        recommendations.append("⚠️ No statistically significant differences found")
    
    # Performance insights
    control_stats = analysis.variant_stats.get('control', {})
    if control_stats:
        control_reward = control_stats.get('mean_reward', 0)
        
        for variant, stats in analysis.variant_stats.items():
            if variant == 'control':
                continue
            
            variant_reward = stats.get('mean_reward', 0)
            diff = variant_reward - control_reward
            
            if diff > 0.1:  # Meaningful improvement
                recommendations.append(f"✅ {variant.upper()} outperforms control by {diff:.3f}")
            elif diff < -0.1:  # Meaningful degradation
                recommendations.append(f"❌ {variant.upper()} underperforms control by {abs(diff):.3f}")
    
    # Sample size warnings
    for variant, stats in analysis.variant_stats.items():
        if stats.get('count', 0) < 10:
            recommendations.append(f"⚠️ Small sample size for {variant.upper()} (n={stats['count']})")
    
    return recommendations

def create_analysis_report(analysis: ExperimentAnalysis) -> str:
    """Create a comprehensive text report."""
    
    report = []
    report.append("=" * 80)
    report.append(f"TAU-BENCH EMOTIONAL USER ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Experiment: {analysis.experiment_name}")
    report.append(f"Timestamp: {analysis.timestamp}")
    report.append(f"Environment: {analysis.environment.upper()}")
    report.append("")
    
    # Variant Statistics
    report.append("📊 VARIANT PERFORMANCE STATISTICS")
    report.append("-" * 40)
    for variant, stats in analysis.variant_stats.items():
        rank = analysis.variant_rankings.get(variant, "?")
        report.append(f"{variant.upper()} (Rank #{rank}):")
        report.append(f"  Tasks: {stats['count']}")
        report.append(f"  Mean Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
        report.append(f"  Success Rate: {stats['success_rate']:.1%}")
        report.append(f"  Reward Range: {stats['min_reward']:.3f} - {stats['max_reward']:.3f}")
        report.append(f"  Median: {stats['median_reward']:.3f}")
        report.append("")
    
    # Pairwise Comparisons
    report.append("🔍 PAIRWISE STATISTICAL COMPARISONS")
    report.append("-" * 40)
    for comparison in analysis.pairwise_comparisons:
        effect_desc = ""
        if abs(comparison.effect_size) < 0.2:
            effect_desc = "(negligible effect)"
        elif abs(comparison.effect_size) < 0.5:
            effect_desc = "(small effect)"
        elif abs(comparison.effect_size) < 0.8:
            effect_desc = "(medium effect)"
        else:
            effect_desc = "(large effect)"
        
        sig_desc = ""
        if comparison.significant is True:
            sig_desc = " ***"
        elif comparison.significant is False:
            sig_desc = " (n.s.)"
        
        report.append(f"{comparison.variant1.upper()} vs {comparison.variant2.upper()}:")
        report.append(f"  Mean difference: {comparison.difference:+.3f}{sig_desc}")
        report.append(f"  Effect size (Cohen's d): {comparison.effect_size:.3f} {effect_desc}")
        if comparison.p_value:
            report.append(f"  p-value: {comparison.p_value:.4f}")
        report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 40)
    for rec in analysis.recommendations:
        report.append(f"• {rec}")
    report.append("")
    
    # Footer
    if not HAS_SCIPY:
        report.append("⚠️  Note: Statistical significance tests require scipy installation")
    report.append("=" * 80)
    
    return "\n".join(report)

def analyze_experiment(results_dir: Path) -> ExperimentAnalysis:
    """Perform comprehensive analysis of experiment results."""
    
    logger.info(f"Analyzing experiment results: {results_dir}")
    
    # Load results
    results_data = load_aggregated_results(results_dir)
    
    experiment_name = results_data['experiment_name']
    timestamp = results_data['timestamp']
    environment = results_data['environment']
    variants = results_data['variants']
    task_results = results_data['task_results']
    
    logger.info(f"Loaded {len(task_results)} task results across {len(variants)} variants")
    
    # Calculate detailed statistics
    variant_stats = calculate_detailed_stats(task_results)
    
    # Perform pairwise comparisons
    pairwise_comparisons = perform_pairwise_comparisons(task_results, variants)
    
    # Rank variants
    variant_rankings = rank_variants(variant_stats)
    
    # Find significant differences
    significant_differences = []
    for comp in pairwise_comparisons:
        if comp.significant:
            significant_differences.append(str(comp))
    
    # Create analysis object
    analysis = ExperimentAnalysis(
        experiment_name=experiment_name,
        timestamp=timestamp,
        environment=environment,
        variant_stats=variant_stats,
        pairwise_comparisons=pairwise_comparisons,
        variant_rankings=variant_rankings,
        significant_differences=significant_differences,
        recommendations=[]
    )
    
    # Generate recommendations
    analysis.recommendations = generate_recommendations(analysis)
    
    return analysis

def save_analysis_report(analysis: ExperimentAnalysis, output_path: Path):
    """Save analysis report to file."""
    
    # Text report
    text_report = create_analysis_report(analysis)
    text_path = output_path.with_suffix('.txt')
    
    with open(text_path, 'w') as f:
        f.write(text_report)
    
    logger.info(f"📄 Analysis report saved: {text_path}")
    
    # JSON report with all data
    json_path = output_path.with_suffix('.json')
    
    analysis_dict = {
        'experiment_name': analysis.experiment_name,
        'timestamp': analysis.timestamp,
        'environment': analysis.environment,
        'variant_stats': analysis.variant_stats,
        'variant_rankings': analysis.variant_rankings,
        'significant_differences': analysis.significant_differences,
        'recommendations': analysis.recommendations,
        'pairwise_comparisons': [
            {
                'variant1': c.variant1,
                'variant2': c.variant2,
                'difference': c.difference,
                'effect_size': c.effect_size,
                'p_value': c.p_value,
                'significant': c.significant
            }
            for c in analysis.pairwise_comparisons
        ]
    }
    
    with open(json_path, 'w') as f:
        json.dump(analysis_dict, f, indent=2)
    
    logger.info(f"📊 Analysis data saved: {json_path}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze tau-bench experiment results")
    
    parser.add_argument("results_dir", type=Path,
                       help="Path to experiment results directory")
    parser.add_argument("--output", type=Path,
                       help="Output file for analysis report (default: analysis_report)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Default output path
    if not args.output:
        args.output = args.results_dir / "analysis_report"
    
    try:
        # Perform analysis
        analysis = analyze_experiment(args.results_dir)
        
        # Print summary to console
        print(create_analysis_report(analysis))
        
        # Save detailed reports
        save_analysis_report(analysis, args.output)
        
        logger.info("🎉 Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()