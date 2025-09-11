#!/usr/bin/env python3
import json
import os

def calculate_averages(model_path, model_name):
    """Calculate average layer and sequence percentages for a model"""
    with open(model_path, 'r') as f:
        data = json.load(f)
    
    outliers = data['all_systematic_outliers']
    
    layer_percentages = [outlier['layer_percentage'] for outlier in outliers]
    seq_percentages = [outlier['seq_percentage'] for outlier in outliers]
    
    avg_layer = sum(layer_percentages) / len(layer_percentages) * 100
    avg_seq = sum(seq_percentages) / len(seq_percentages) * 100
    
    print(f"{model_name}:")
    print(f"  Average Layer %: {avg_layer:.1f}%")
    print(f"  Average Seq %: {avg_seq:.1f}%")
    print(f"  Total Outliers: {len(outliers)}")
    print()
    
    return avg_layer, avg_seq, len(outliers)

# Calculate for all three models
base_path = "/Users/chiraagbalu/llm-workbench/examples/outlier_features_moe/remote_results"

models = [
    ("olmoe-1b-7b/final_analysis_results.json", "OLMoE-1B-7B (~1B params)"),
    ("glm-4.5-air/batch_001_results.json", "GLM-4.5-Air (~15B params)"),
    ("qwen3-30b-a3b/final_analysis_results.json", "Qwen3-30B-A3B (~30B params)")
]

print("=== MoE Model Outlier Analysis Averages ===")
print()

results = []
for model_file, model_name in models:
    full_path = os.path.join(base_path, model_file)
    if os.path.exists(full_path):
        avg_layer, avg_seq, total_outliers = calculate_averages(full_path, model_name)
        results.append((model_name, avg_layer, avg_seq, total_outliers))
    else:
        print(f"File not found: {full_path}")

print("=== Summary for Paper Figure ===")
for model_name, avg_layer, avg_seq, total_outliers in results:
    param_size = model_name.split("(")[1].split(")")[0]
    print(f"{param_size}: Layer {avg_layer:.1f}%, Seq {avg_seq:.1f}% ({total_outliers} outliers)")