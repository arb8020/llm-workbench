import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path
import json

def load_activations(run_dir):
    """Load activations from a run directory."""
    run_path = Path(run_dir)
    
    # Load metadata
    metadata_file = run_path / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    activations = {}
    for layer_idx in metadata["layers_extracted"]:
        # Load each activation file separately (matching how they were saved)
        attn_file = run_path / f"layer_{layer_idx}_ln_attn_activations.pt"
        mlp_file = run_path / f"layer_{layer_idx}_ln_mlp_activations.pt"
        
        attn_tensor = torch.load(attn_file, map_location='cpu')
        mlp_tensor = torch.load(mlp_file, map_location='cpu')
        
        activations[f"layer_{layer_idx}_ln_attn"] = attn_tensor
        activations[f"layer_{layer_idx}_ln_mlp"] = mlp_tensor
        
        print(f"Loaded layer_{layer_idx} activations:")
        print(f"  ln_attn shape={tuple(attn_tensor.shape)}")
        print(f"  ln_mlp shape={tuple(mlp_tensor.shape)}")
        
    return activations, metadata

def find_outliers_in_activations(activations, magnitude_threshold=6.0):
    """
    Find outlier features across all layers and sequence positions.
    
    Paper methodology: "We track dimensions h_i, 0 ≤ i ≤ h, which have at least one value 
    with a magnitude of α ≥ 6"
    
    Args:
        activations: Dict of {layer_name: tensor} from load_activations()
        magnitude_threshold: Minimum magnitude to consider as outlier (paper uses 6.0)
        
    Returns:
        outlier_info: Dict with outlier statistics and locations
    """
    print(f"Searching for outliers with magnitude >= {magnitude_threshold}")
    
    # Track outliers by feature dimension h_i across all layers - this is key for paper's methodology
    # Paper: "we only collect statistics if these outliers occur in the SAME feature dimension h_i"
    feature_outliers = defaultdict(list)  # feature_dim -> [(layer, seq_pos, value), ...]
    layer_stats = {}  # layer_name -> stats
    
    for layer_name, tensor in activations.items():
        print(f"\nAnalyzing {layer_name}: shape={tuple(tensor.shape)}")
        
        # tensor shape: (batch, seq_len, d_model) where d_model contains feature dimensions h_i
        batch_size, seq_len, d_model = tensor.shape
        
        # Find outliers in this layer using paper's magnitude criterion (≥ 6.0)
        outlier_mask = torch.abs(tensor) >= magnitude_threshold
        outlier_positions = torch.where(outlier_mask)  # Returns (batch_idx, seq_idx, feature_idx) tuples
        
        num_outliers = outlier_mask.sum().item()
        max_val = tensor.abs().max().item()
        
        layer_stats[layer_name] = {
            'num_outliers': num_outliers,
            'max_magnitude': max_val,
            'outlier_percentage': (num_outliers / tensor.numel()) * 100,
            'shape': tuple(tensor.shape)
        }
        
        print(f"  Found {num_outliers} outliers ({layer_stats[layer_name]['outlier_percentage']:.3f}%)")
        print(f"  Max magnitude: {max_val:.3f}")
        
        # Record outliers by feature dimension h_i - this implements paper's tracking requirement
        if num_outliers > 0:
            batch_indices, seq_indices, feature_indices = outlier_positions
            values = tensor[outlier_positions]
            
            # Group outliers by their feature dimension h_i (not by layer or sequence position)
            for i in range(len(feature_indices)):
                feature_dim = feature_indices[i].item()  # This is h_i in paper notation
                seq_pos = seq_indices[i].item()          # Position in sequence dimension s
                value = values[i].item()                 # The actual activation value with magnitude ≥ 6.0
                
                # Track each occurrence: same h_i can appear in multiple layers/positions
                feature_outliers[feature_dim].append({
                    'layer': layer_name,  # Which transformer layer (part of "hidden states X_l")
                    'seq_pos': seq_pos,   # Which sequence position (part of "sequence dimensions s")
                    'value': value        # The outlier magnitude
                })
    
    return {
        'feature_outliers': dict(feature_outliers),
        'layer_stats': layer_stats,
        'threshold': magnitude_threshold
    }

def analyze_systematic_outliers(outlier_info, min_layer_percentage=0.25, min_seq_percentage=0.06):
    """
    Find systematic outliers that appear consistently across layers and sequence positions.
    
    Paper's exact criteria: "We define outliers according to the following criteria: 
    the magnitude of the feature is at least 6.0, affects at least 25% of layers, 
    and affects at least 6% of the sequence dimensions."
    
    Args:
        outlier_info: Output from find_outliers_in_activations()
        min_layer_percentage: Paper uses 25% - "at least 25% of transformer layers"
        min_seq_percentage: Paper uses 6% - "at least 6% of all sequence dimensions s"
        
    Returns:
        systematic_outliers: List of feature dimensions h_i that meet all criteria
    """
    feature_outliers = outlier_info['feature_outliers']
    layer_stats = outlier_info['layer_stats']
    
    total_layers = len(layer_stats)
    
    # Get sequence length from any layer (they should all be the same)
    seq_len = next(iter(layer_stats.values()))['shape'][1]
    
    systematic_features = []
    
    print(f"\nAnalyzing systematic outliers:")
    print(f"Criteria: ≥{min_layer_percentage*100}% of {total_layers} layers, ≥{min_seq_percentage*100}% of {seq_len} seq positions")
    
    for feature_dim, outlier_list in feature_outliers.items():
        # Paper criterion 1: "affects at least 25% of layers"
        # Count how many different transformer layers have outliers in this same feature dimension h_i
        layers_with_outlier = set(item['layer'] for item in outlier_list)
        layer_percentage = len(layers_with_outlier) / total_layers
        
        # Paper criterion 2: "affects at least 6% of the sequence dimensions"
        # Count unique sequence positions where this feature h_i has outliers "across all hidden states X_l"
        seq_positions_with_outlier = set(item['seq_pos'] for item in outlier_list)
        seq_percentage = len(seq_positions_with_outlier) / seq_len
        
        # Apply paper's systematic criteria (magnitude ≥6.0 already applied in previous step)
        meets_layer_criteria = layer_percentage >= min_layer_percentage    # ≥25% layers
        meets_seq_criteria = seq_percentage >= min_seq_percentage           # ≥6% sequence positions
        
        if meets_layer_criteria and meets_seq_criteria:
            max_magnitude = max(abs(item['value']) for item in outlier_list)
            
            systematic_features.append({
                'feature_dim': feature_dim,
                'layer_percentage': layer_percentage,
                'seq_percentage': seq_percentage,
                'max_magnitude': max_magnitude,
                'total_occurrences': len(outlier_list),
                'layers_affected': sorted(layers_with_outlier),
                'example_values': [item['value'] for item in outlier_list[:5]]  # First 5 examples
            })
            
            print(f"  Feature {feature_dim}: {layer_percentage:.1%} layers, {seq_percentage:.1%} seq_pos, max_mag={max_magnitude:.2f}")
    
    systematic_features.sort(key=lambda x: x['max_magnitude'], reverse=True)
    
    print(f"\nFound {len(systematic_features)} systematic outlier features")
    
    return systematic_features

def print_outlier_summary(systematic_outliers, outlier_info):
    """Print a nice summary of the outlier analysis."""
    if not systematic_outliers:
        print("\n" + "="*60)
        print("NO SYSTEMATIC OUTLIERS FOUND")
        print("="*60)
        print("This could mean:")
        print("- Text is too simple/short")
        print("- Model doesn't exhibit outlier behavior on this input")
        print("- Need to try deeper layers or different text")
        return
    
    print("\n" + "="*60)
    print("SYSTEMATIC OUTLIER SUMMARY")
    print("="*60)
    
    for i, feature in enumerate(systematic_outliers, 1):
        print(f"\n{i}. Feature Dimension {feature['feature_dim']}:")
        print(f"   Max Magnitude: {feature['max_magnitude']:.2f}")
        print(f"   Appears in {feature['layer_percentage']:.1%} of layers: {feature['layers_affected']}")
        print(f"   Affects {feature['seq_percentage']:.1%} of sequence positions")
        print(f"   Total occurrences: {feature['total_occurrences']}")
        print(f"   Example values: {[f'{v:.2f}' for v in feature['example_values']]}")

# Example usage function
def analyze_run_for_outliers(run_dir, magnitude_threshold=6.0):
    """
    Complete pipeline: load activations and analyze for outliers.
    """
    print(f"Loading activations from: {run_dir}")
    activations, metadata = load_activations(run_dir)
    
    print(f"\nAnalyzing model: {metadata['model_name']}")
    print(f"Input text: '{metadata['sequence_texts']}'")
    
    # Find all outliers
    outlier_info = find_outliers_in_activations(activations, magnitude_threshold)
    
    # Find systematic outliers
    systematic_outliers = analyze_systematic_outliers(outlier_info)
    
    # Print summary
    print_outlier_summary(systematic_outliers, outlier_info)
    
    return systematic_outliers, outlier_info

if __name__ == "__main__":
    # Example: analyze the run you just created
    run_dir = "my_activations/run_20250906_120235"  # Update with your actual run directory
    systematic_outliers, outlier_info = analyze_run_for_outliers(run_dir)
