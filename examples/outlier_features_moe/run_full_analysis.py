#!/usr/bin/env python3
"""
Push-button script to extract activations and analyze for outlier features.
Combines extract_activations.py and analyze_activations.py into a single workflow.
"""

import sys
import argparse
from pathlib import Path

# Import functions from the other scripts
from extract_activations import extract_activations
from analyze_activations import analyze_run_for_outliers


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze activations for outlier features")
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0125-Instruct", 
                       help="HuggingFace model identifier")
    parser.add_argument("--text", default="The capital of France is Paris.", 
                       help="Input text to analyze")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                       help="Layer indices to extract (e.g., --layers 0 1 2 3)")
    parser.add_argument("--save-dir", default="./full_analysis_results",
                       help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=6.0,
                       help="Magnitude threshold for outlier detection")
    
    args = parser.parse_args()
    
    print("="*60)
    print("FULL OUTLIER ANALYSIS PIPELINE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Text: '{args.text}'")
    print(f"Layers: {args.layers}")
    print(f"Threshold: {args.threshold}")
    print(f"Save dir: {args.save_dir}")
    
    # Step 1: Extract activations
    print("\n" + "="*40)
    print("STEP 1: EXTRACTING ACTIVATIONS")
    print("="*40)
    
    try:
        run_dir, metadata = extract_activations(
            model_name=args.model,
            text=args.text,
            layers=args.layers,
            save_dir=args.save_dir
        )
        print(f"✅ Extraction completed. Results saved to: {run_dir}")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)
    
    # Step 2: Analyze for outliers
    print("\n" + "="*40)
    print("STEP 2: ANALYZING FOR OUTLIERS")
    print("="*40)
    
    try:
        systematic_outliers, outlier_info = analyze_run_for_outliers(
            run_dir=run_dir,
            magnitude_threshold=args.threshold
        )
        
        print(f"\n✅ Analysis completed!")
        print(f"Found {len(systematic_outliers)} systematic outlier features")
        
        # Summary
        if systematic_outliers:
            print(f"\nTop outlier features:")
            for i, feature in enumerate(systematic_outliers[:3], 1):
                print(f"  {i}. Feature {feature['feature_dim']}: max_mag={feature['max_magnitude']:.2f}")
        
        return run_dir, systematic_outliers, outlier_info
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
