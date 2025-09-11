#!/usr/bin/env python3
import json

def extract_paper_stats():
    """Extract the key statistics for paper replication"""
    
    # Load the comprehensive analysis
    with open('/Users/chiraagbalu/llm-workbench/moe_comprehensive_outlier_analysis.json', 'r') as f:
        data = json.load(f)
    
    results = {
        "paper_replication_data": {
            "description": "Statistics for replicating paper figure: % layers affected vs % sequence positions affected",
            "aggregation_method": "mean (following arXiv:2208.07339 methodology)",
            "models_ordered_by_active_params": {}
        }
    }
    
    # Extract stats for each model
    for model_name, model_data in data["models"].items():
        layer_stats = model_data["layer_percentage_stats"]
        seq_stats = model_data["sequence_percentage_stats"]
        model_info = model_data["model_info"]
        
        results["paper_replication_data"]["models_ordered_by_active_params"][model_name] = {
            "parameters": {
                "total": model_info["total_parameters"],
                "active": model_info["active_parameters"],
                "active_numeric": model_info["active_parameters_numeric"]
            },
            "sample_size": model_data["sample_size"],
            "layer_coverage_percent": {
                "mean": round(layer_stats["mean"], 1),
                "median": round(layer_stats["median"], 1),
                "min": round(layer_stats["min"], 1),
                "max": round(layer_stats["max"], 1),
                "range": f"{round(layer_stats['min'], 1)}-{round(layer_stats['max'], 1)}%",
                "std": round(layer_stats["std"], 1)
            },
            "sequence_coverage_percent": {
                "mean": round(seq_stats["mean"], 1),
                "median": round(seq_stats["median"], 1), 
                "min": round(seq_stats["min"], 1),
                "max": round(seq_stats["max"], 1),
                "range": f"{round(seq_stats['min'], 1)}-{round(seq_stats['max'], 1)}%",
                "std": round(seq_stats["std"], 1)
            },
            "paper_plot_coordinates": {
                "x_axis_active_params": model_info["active_parameters"],
                "y_axis_layer_mean": round(layer_stats["mean"], 1),
                "z_axis_seq_mean": round(seq_stats["mean"], 1)
            }
        }
    
    # Add summary table
    results["summary_table"] = {
        "headers": ["Model", "Active Params", "Layer % (Mean)", "Layer % (Median)", "Layer Range", "Seq % (Mean)", "Seq % (Median)", "Seq Range", "Sample Size"],
        "rows": []
    }
    
    for model_name, stats in results["paper_replication_data"]["models_ordered_by_active_params"].items():
        row = [
            model_name,
            stats["parameters"]["active"],
            f"{stats['layer_coverage_percent']['mean']}%",
            f"{stats['layer_coverage_percent']['median']}%", 
            stats["layer_coverage_percent"]["range"],
            f"{stats['sequence_coverage_percent']['mean']}%",
            f"{stats['sequence_coverage_percent']['median']}%",
            stats["sequence_coverage_percent"]["range"],
            stats["sample_size"]
        ]
        results["summary_table"]["rows"].append(row)
    
    return results

# Generate the stats
stats = extract_paper_stats()

# Save to file
with open('/Users/chiraagbalu/llm-workbench/paper_replication_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("📊 PAPER REPLICATION STATISTICS")
print("=" * 80)
print()

print("🎯 AGGREGATION METHOD: Mean (following arXiv:2208.07339)")
print()

# Print formatted table
headers = stats["summary_table"]["headers"]
rows = stats["summary_table"]["rows"]

print(f"{'Model':<15} {'Active':<8} {'Layer Mean':<12} {'Layer Median':<14} {'Layer Range':<15} {'Seq Mean':<10} {'Seq Median':<12} {'Seq Range':<15} {'Sample':<8}")
print("-" * 120)

for row in rows:
    print(f"{row[0]:<15} {row[1]:<8} {row[2]:<12} {row[3]:<14} {row[4]:<15} {row[5]:<10} {row[6]:<12} {row[7]:<15} {row[8]:<8}")

print()
print("📈 KEY INSIGHTS:")
print("• Mean vs Median differences indicate distribution skew")
print("• Wide ranges show heterogeneity in outlier patterns") 
print("• Sample sizes vary significantly across models")
print()
print("🎨 FOR PLOTTING:")
print("• X-axis: Active Parameters (1.3B → 3.3B → 12B)")
print("• Y-axis: Layer Coverage Mean % (28.0 → 35.5 → 67.4)")  
print("• Color/Size: Sequence Coverage Mean % (14.3 → 45.1 → 46.8)")
print()
print("✅ Complete statistics saved to: paper_replication_stats.json")