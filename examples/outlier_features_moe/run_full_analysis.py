#!/usr/bin/env python3
"""
Push-button script to extract activations and analyze for outlier features.
Combines extract_activations.py and analyze_activations.py into a single workflow.
"""

import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer

# Import functions from the other scripts
from extract_activations import extract_activations, extract_activations_optimized
from analyze_activations import analyze_run_for_outliers
from dataset_utils import get_text_sequences


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze activations for outlier features")
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0125-Instruct", 
                       help="HuggingFace model identifier")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                       help="HuggingFace dataset identifier")
    parser.add_argument("--num-sequences", type=int, default=16,
                       help="Number of text sequences to extract")
    parser.add_argument("--sequence-length", type=int, default=2048,
                       help="Target length of each sequence in tokens")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Number of sequences to process in each batch")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                       help="Specific layer indices to extract (e.g., --layers 0 1 2 3). If not specified, extracts from all layers.")
    parser.add_argument("--save-dir", default="./full_analysis_results",
                       help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=6.0,
                       help="Magnitude threshold for outlier detection")
    
    args = parser.parse_args()
    
    print("="*60)
    print("FULL OUTLIER ANALYSIS PIPELINE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Sequences: {args.num_sequences} x {args.sequence_length} tokens")
    print(f"Batch size: {args.batch_size}")
    if args.layers is None:
        print("Layers: All layers (will be determined from model)")
    else:
        print(f"Layers: {args.layers}")
    print(f"Threshold: {args.threshold}")
    print(f"Save dir: {args.save_dir}")
    
    # Validate arguments
    assert args.num_sequences > 0, "num_sequences must be positive"
    assert args.sequence_length > 0, "sequence_length must be positive"  
    assert args.batch_size > 0, "batch_size must be positive"
    assert args.num_sequences % args.batch_size == 0, f"num_sequences ({args.num_sequences}) must be divisible by batch_size ({args.batch_size})"
    
    # Step 0: Load tokenizer and dataset sequences
    print("\n" + "="*40)
    print("STEP 0: LOADING TOKENIZER AND DATASET")
    print("="*40)
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        
        # Load dataset sequences using tokenizer
        print("Loading dataset sequences...")
        text_sequences = get_text_sequences(
            dataset_name=args.dataset,
            num_sequences=args.num_sequences,
            sequence_length=args.sequence_length,
            tokenizer=tokenizer
        )
        print(f"✅ Loaded {len(text_sequences)} sequences")
        
    except Exception as e:
        print(f"❌ Dataset/tokenizer loading failed: {e}")
        sys.exit(1)
    
    # Step 1: Load model once with optimized memory settings
    print("\n" + "="*40)
    print("STEP 1: LOADING MODEL (MEMORY OPTIMIZED)")
    print("="*40)
    
    import torch
    from nnsight import LanguageModel
    
    # Load model once with balanced device mapping and memory limits
    print(f"Loading model: {args.model}")
    print("Using balanced device mapping with 76GB per GPU limit...")
    
    llm = LanguageModel(
        args.model,
        device_map="balanced",
        max_memory={0: "76GiB", 1: "76GiB"},
        torch_dtype=torch.bfloat16
    )
    
    # Disable KV cache to save memory
    llm.model.config.use_cache = False
    
    # Print device mapping for verification
    if hasattr(llm.model, 'hf_device_map'):
        print(f"Device mapping: {llm.model.hf_device_map}")
    
    print("✅ Model loaded successfully")
    
    # Step 2: Extract activations in batches (reusing model)
    print("\n" + "="*40)
    print("STEP 2: EXTRACTING ACTIVATIONS (CHUNKED)")
    print("="*40)
    
    num_batches = args.num_sequences // args.batch_size
    all_run_dirs = []
    
    try:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size
            batch_texts = text_sequences[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch_texts)} sequences)")
            
            run_dir, metadata = extract_activations_optimized(
                llm=llm,
                texts=batch_texts,
                layers=args.layers,
                save_dir=args.save_dir,
                chunk_size=8  # Process 8 layers at a time
            )
            all_run_dirs.append(run_dir)
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
            print(f"✅ Batch {batch_idx + 1} completed: {run_dir}")
        
        print(f"\n✅ All {num_batches} batches completed!")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)
    
    # Step 2: Analyze for outliers across all batches
    print("\n" + "="*40)
    print("STEP 2: ANALYZING FOR OUTLIERS")
    print("="*40)
    
    all_systematic_outliers = []
    all_outlier_info = []
    
    try:
        for i, run_dir in enumerate(all_run_dirs, 1):
            print(f"\nAnalyzing batch {i}/{len(all_run_dirs)}: {run_dir}")
            
            systematic_outliers, outlier_info = analyze_run_for_outliers(
                run_dir=run_dir,
                magnitude_threshold=args.threshold
            )
            
            all_systematic_outliers.extend(systematic_outliers)
            all_outlier_info.append(outlier_info)
            
            print(f"Batch {i}: Found {len(systematic_outliers)} systematic outlier features")
        
        print(f"\n✅ Analysis completed across all {len(all_run_dirs)} batches!")
        print(f"Total systematic outlier features found: {len(all_systematic_outliers)}")
        
        # Aggregate summary across all batches
        if all_systematic_outliers:
            # Group by feature dimension and aggregate
            feature_aggregates = {}
            for feature in all_systematic_outliers:
                dim = feature['feature_dim']
                if dim not in feature_aggregates:
                    feature_aggregates[dim] = {
                        'feature_dim': dim,
                        'max_magnitude': feature['max_magnitude'],
                        'occurrences': 1
                    }
                else:
                    feature_aggregates[dim]['max_magnitude'] = max(
                        feature_aggregates[dim]['max_magnitude'],
                        feature['max_magnitude']
                    )
                    feature_aggregates[dim]['occurrences'] += 1
            
            # Sort by max magnitude
            top_features = sorted(feature_aggregates.values(), 
                                key=lambda x: x['max_magnitude'], reverse=True)
            
            print(f"\nTop outlier features across all batches:")
            for i, feature in enumerate(top_features[:5], 1):
                print(f"  {i}. Feature {feature['feature_dim']}: max_mag={feature['max_magnitude']:.2f}, "
                      f"appeared in {feature['occurrences']}/{len(all_run_dirs)} batches")
        else:
            print("\nNo systematic outlier features found across any batch.")
            print("Consider:")
            print("- Using longer sequences or more complex text")
            print("- Trying different layers")
            print("- Lowering the magnitude threshold")
        
        return all_run_dirs, all_systematic_outliers, all_outlier_info
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
