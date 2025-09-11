#!/usr/bin/env python3
import json
import numpy as np
import os
from pathlib import Path

def calculate_full_stats(values):
    """Calculate comprehensive statistics for a list of values"""
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

def analyze_model(file_path, model_info):
    """Analyze a single model's outlier data"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        outliers = data.get('all_systematic_outliers', [])
        
        # Extract percentages
        layer_percentages = [outlier['layer_percentage'] * 100 for outlier in outliers]
        seq_percentages = [outlier['seq_percentage'] * 100 for outlier in outliers]
        max_magnitudes = [outlier['max_magnitude'] for outlier in outliers]
        
        return {
            "model_info": model_info,
            "sample_size": len(outliers),
            "layer_percentage_stats": calculate_full_stats(layer_percentages),
            "sequence_percentage_stats": calculate_full_stats(seq_percentages),
            "max_magnitude_stats": calculate_full_stats(max_magnitudes),
            "total_systematic_outliers": data.get('analysis_summary', {}).get('total_systematic_outliers', len(outliers))
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    base_path = Path("/Users/chiraagbalu/llm-workbench/examples/outlier_features_moe/remote_results")
    
    # Model configurations with parameter info from web search
    models = [
        {
            "name": "OLMoE-1B-7B",
            "file": "olmoe-1b-7b/final_analysis_results.json",
            "total_params": "7B",  # 7 billion total
            "active_params": "1.3B",  # 1.3 billion active per token
            "total_params_numeric": 7e9,
            "active_params_numeric": 1.3e9,
            "architecture": "MoE with 64 experts, 8 activated per token"
        },
        {
            "name": "Qwen3-30B-A3B", 
            "file": "qwen3-30b-a3b/final_analysis_results.json",
            "total_params": "30.5B",  # 30.5 billion total
            "active_params": "3.3B",  # 3.3 billion active per token
            "total_params_numeric": 30.5e9,
            "active_params_numeric": 3.3e9,
            "architecture": "MoE with 128 experts, 8 activated per token"
        },
        {
            "name": "GLM-4.5-Air",
            "file": "glm-4.5-air/batch_001_results.json", 
            "total_params": "106B",  # 106 billion total
            "active_params": "12B",  # 12 billion active per token
            "total_params_numeric": 106e9,
            "active_params_numeric": 12e9,
            "architecture": "MoE architecture (specific expert config not detailed)"
        }
    ]
    
    results = {
        "analysis_metadata": {
            "description": "Comprehensive outlier analysis statistics for MoE models",
            "threshold": "6.0 sigma",
            "sequence_length": 2048,
            "date_generated": "2025-09-09"
        },
        "models": {}
    }
    
    for model_config in models:
        file_path = base_path / model_config["file"]
        
        if file_path.exists():
            print(f"Analyzing {model_config['name']}...")
            
            model_info = {
                "name": model_config["name"],
                "total_parameters": model_config["total_params"],
                "active_parameters": model_config["active_params"],
                "total_parameters_numeric": model_config["total_params_numeric"],
                "active_parameters_numeric": model_config["active_params_numeric"],
                "architecture": model_config["architecture"]
            }
            
            analysis = analyze_model(file_path, model_info)
            if analysis:
                results["models"][model_config["name"]] = analysis
                print(f"✓ {model_config['name']}: {analysis['sample_size']} outliers analyzed")
            else:
                print(f"✗ Failed to analyze {model_config['name']}")
        else:
            print(f"✗ File not found: {file_path}")
    
    # Sort by active parameters for easy comparison
    sorted_models = dict(sorted(results["models"].items(), 
                               key=lambda x: x[1]["model_info"]["active_parameters_numeric"]))
    results["models"] = sorted_models
    
    # Add summary comparison
    results["summary"] = {}
    for model_name, data in results["models"].items():
        active_params = data["model_info"]["active_parameters"]
        layer_mean = data["layer_percentage_stats"]["mean"]
        seq_mean = data["sequence_percentage_stats"]["mean"] 
        sample_size = data["sample_size"]
        
        results["summary"][model_name] = {
            "active_params": active_params,
            "avg_layer_coverage": f"{layer_mean:.1f}%",
            "avg_sequence_coverage": f"{seq_mean:.1f}%",
            "sample_size": sample_size
        }
    
    # Write to JSON
    output_path = "/Users/chiraagbalu/llm-workbench/moe_outlier_comprehensive_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n✓ Complete analysis saved to: {output_path}")
    print(f"\\n=== QUICK SUMMARY ===")
    for model_name, summary in results["summary"].items():
        print(f"{model_name} ({summary['active_params']} active): {summary['avg_layer_coverage']} layers, {summary['avg_sequence_coverage']} sequences ({summary['sample_size']} outliers)")

if __name__ == "__main__":
    main()