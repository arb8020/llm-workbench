#!/usr/bin/env python3
import json
import numpy as np
import re

def extract_glm_stats():
    """Extract GLM statistics using grep approach due to JSON parsing issues"""
    
    # Use bash commands to extract the data
    import subprocess
    
    # Get layer percentages
    cmd_layer = ["grep", "-o", '"layer_percentage": [0-9.]*', 
                "/Users/chiraagbalu/llm-workbench/examples/outlier_features_moe/remote_results/glm-4.5-air/batch_001_results.json"]
    result_layer = subprocess.run(cmd_layer, capture_output=True, text=True)
    layer_values = [float(line.split(': ')[1]) * 100 for line in result_layer.stdout.strip().split('\n') if line]
    
    # Get sequence percentages  
    cmd_seq = ["grep", "-o", '"seq_percentage": [0-9.]*',
              "/Users/chiraagbalu/llm-workbench/examples/outlier_features_moe/remote_results/glm-4.5-air/batch_001_results.json"]
    result_seq = subprocess.run(cmd_seq, capture_output=True, text=True)
    seq_values = [float(line.split(': ')[1]) * 100 for line in result_seq.stdout.strip().split('\n') if line]
    
    # Get max magnitudes
    cmd_mag = ["grep", "-o", '"max_magnitude": [0-9.]*',
              "/Users/chiraagbalu/llm-workbench/examples/outlier_features_moe/remote_results/glm-4.5-air/batch_001_results.json"]
    result_mag = subprocess.run(cmd_mag, capture_output=True, text=True)
    mag_values = [float(line.split(': ')[1]) for line in result_mag.stdout.strip().split('\n') if line]
    
    return layer_values, seq_values, mag_values

def calculate_stats(values):
    """Calculate comprehensive statistics"""
    if not values:
        return None
    values = np.array(values)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75))
    }

# Extract GLM data
layer_vals, seq_vals, mag_vals = extract_glm_stats()

glm_analysis = {
    "model_info": {
        "name": "GLM-4.5-Air",
        "total_parameters": "106B",
        "active_parameters": "12B", 
        "total_parameters_numeric": 106e9,
        "active_parameters_numeric": 12e9,
        "architecture": "MoE architecture (specific expert config not detailed)"
    },
    "sample_size": len(layer_vals),
    "layer_percentage_stats": calculate_stats(layer_vals),
    "sequence_percentage_stats": calculate_stats(seq_vals), 
    "max_magnitude_stats": calculate_stats(mag_vals)
}

print(f"GLM-4.5-Air analysis:")
print(f"Sample size: {glm_analysis['sample_size']}")
print(f"Layer % - Mean: {glm_analysis['layer_percentage_stats']['mean']:.1f}%, Range: {glm_analysis['layer_percentage_stats']['min']:.1f}%-{glm_analysis['layer_percentage_stats']['max']:.1f}%")
print(f"Seq % - Mean: {glm_analysis['sequence_percentage_stats']['mean']:.1f}%, Range: {glm_analysis['sequence_percentage_stats']['min']:.1f}%-{glm_analysis['sequence_percentage_stats']['max']:.1f}%")

# Save just GLM data
with open('/Users/chiraagbalu/llm-workbench/glm_analysis.json', 'w') as f:
    json.dump(glm_analysis, f, indent=2)