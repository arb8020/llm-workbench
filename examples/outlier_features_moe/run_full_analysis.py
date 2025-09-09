#!/usr/bin/env python3
"""
Push-button script to extract activations and analyze for outlier features.
Combines extract_activations.py and analyze_activations.py into a single workflow.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

# Import shared logging setup
from shared.logging_config import setup_logging

# Import functions from the other scripts
from extract_activations import extract_activations, extract_activations_optimized
from analyze_activations import analyze_run_for_outliers
from dataset_utils import get_text_sequences

# Setup logger for this module
logger = logging.getLogger(__name__)


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
    parser.add_argument("--chunk-layers", type=int, default=None,
                       help="Number of layers to process at once (default: process all layers together). Use smaller values if running out of GPU memory.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Setup logging first
    setup_logging(level=args.log_level)
    
    logger.info("="*60)
    logger.info("FULL OUTLIER ANALYSIS PIPELINE")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Sequences: {args.num_sequences} x {args.sequence_length} tokens")
    logger.info(f"Batch size: {args.batch_size}")
    if args.layers is None:
        logger.info("Layers: All layers (will be determined from model)")
    else:
        logger.info(f"Layers: {args.layers}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Save dir: {args.save_dir}")
    logger.debug(f"Chunk layers: {args.chunk_layers}")
    
    # Validate arguments
    assert args.num_sequences > 0, "num_sequences must be positive"
    assert args.sequence_length > 0, "sequence_length must be positive"  
    assert args.batch_size > 0, "batch_size must be positive"
    assert args.num_sequences % args.batch_size == 0, f"num_sequences ({args.num_sequences}) must be divisible by batch_size ({args.batch_size})"
    
    # Step 0: Load tokenizer and dataset sequences
    logger.info("\n" + "="*40)
    logger.info("STEP 0: LOADING TOKENIZER AND DATASET")
    logger.info("="*40)
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        logger.debug(f"Tokenizer model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        
        # Load dataset sequences using tokenizer
        logger.info("Loading dataset sequences...")
        logger.debug(f"Dataset: {args.dataset}, sequences: {args.num_sequences}, length: {args.sequence_length}")
        text_sequences = get_text_sequences(
            dataset_name=args.dataset,
            num_sequences=args.num_sequences,
            sequence_length=args.sequence_length,
            tokenizer=tokenizer
        )
        logger.info(f"✅ Loaded {len(text_sequences)} sequences")
        
    except Exception as e:
        logger.error(f"❌ Dataset/tokenizer loading failed: {e}")
        sys.exit(1)
    
    # Step 1: Load model once with optimized memory settings
    logger.info("\n" + "="*40)
    logger.info("STEP 1: LOADING MODEL (MEMORY OPTIMIZED)")
    logger.info("="*40)
    
    import torch
    from nnsight import LanguageModel
    
    # Load model with dynamic GPU detection
    logger.info(f"Loading model: {args.model}")
    
    # Auto-detect available GPUs
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} GPU(s)")
    
    if gpu_count == 1:
        logger.info("Using single-GPU configuration with device_map='auto'")
        llm = LanguageModel(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        logger.info(f"Using multi-GPU balanced configuration with 76GB per GPU limit...")
        max_memory = {i: "76GiB" for i in range(gpu_count)}
        llm = LanguageModel(
            args.model,
            device_map="balanced",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16
        )
    
    # Disable KV cache to save memory
    llm.model.config.use_cache = False
    logger.debug("KV cache disabled to save memory")
    
    # Print device mapping for verification
    if hasattr(llm.model, 'hf_device_map'):
        logger.debug(f"Device mapping: {llm.model.hf_device_map}")
    
    logger.info("✅ Model loaded successfully")
    
    # Clean up HuggingFace cache to free disk space
    print("\n🧹 Cleaning up HuggingFace cache to free disk space...")
    import shutil
    import os
    
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(cache_dir)
                        for filename in filenames) / (1024**3)  # GB
        print(f"   Cache directory size: {cache_size:.2f} GB")
        shutil.rmtree(cache_dir)
        print(f"✅ Deleted HuggingFace cache ({cache_size:.2f} GB freed)")
    else:
        print("   No HuggingFace cache found")
    
    # Step 2: Extract activations in batches (reusing model)
    print("\n" + "="*40)
    print("STEP 2: EXTRACTING ACTIVATIONS (CHUNKED)")
    print("="*40)
    
    num_batches = args.num_sequences // args.batch_size
    all_run_dirs = []
    
    # Process each batch: extract → analyze → sync → cleanup
    all_batch_results = []
    
    try:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size
            batch_texts = text_sequences[start_idx:end_idx]
            
            logger.info(f"\n{'='*20} BATCH {batch_idx + 1}/{num_batches} {'='*20}")
            logger.info(f"Processing {len(batch_texts)} sequences")
            logger.debug(f"Batch texts preview: {[t[:50] + '...' for t in batch_texts]}")
            
            # Step 1: Extract activations
            if args.chunk_layers:
                logger.info(f"Using layer chunking: {args.chunk_layers} layers at a time")
                logger.debug(f"Extraction config: layers={args.layers}, save_dir={args.save_dir}")
                run_dir, metadata = extract_activations_optimized(
                    llm=llm,
                    texts=batch_texts,
                    layers=args.layers,
                    save_dir=args.save_dir,
                    chunk_size=args.chunk_layers
                )
            else:
                logger.info("Processing all layers together (no chunking)")
                logger.debug(f"Extraction config: layers={args.layers}, save_dir={args.save_dir}")
                run_dir, metadata = extract_activations_optimized(
                    llm=llm,
                    texts=batch_texts,
                    layers=args.layers,
                    save_dir=args.save_dir,
                    chunk_size=None  # Process all layers at once
                )
            
            logger.info(f"✅ Activation extraction completed: {run_dir}")
            logger.debug(f"Extraction metadata: {metadata}")
            
            # Step 2: Immediately analyze this batch for outliers
            logger.info(f"🔍 Analyzing batch {batch_idx + 1} for outliers...")
            logger.debug(f"Using threshold: {args.threshold}")
            systematic_outliers, outlier_info = analyze_run_for_outliers(
                run_dir=run_dir,
                magnitude_threshold=args.threshold
            )
            
            # Step 3: Create batch result summary
            batch_result = {
                "batch_id": batch_idx + 1,
                "run_dir": str(run_dir),
                "sequences_processed": len(batch_texts),
                "systematic_outliers": systematic_outliers,
                "outlier_info": outlier_info,
                "timestamp": datetime.now().isoformat(),
                "analysis_params": {
                    "threshold": args.threshold,
                    "model": args.model,
                    "sequence_length": args.sequence_length,
                    "layers": args.layers
                }
            }
            all_batch_results.append(batch_result)
            
            # Step 4: Save batch result to disk immediately
            batch_result_file = Path(args.save_dir) / f"batch_{batch_idx + 1:03d}_results.json"
            with open(batch_result_file, 'w') as f:
                json.dump(batch_result, f, indent=2)
            
            logger.info(f"✅ Batch {batch_idx + 1}: Found {len(systematic_outliers)} systematic outlier features")
            logger.info(f"📁 Results saved: {batch_result_file}")
            logger.debug(f"Batch result summary: {len(batch_result['systematic_outliers'])} outliers, {batch_result['sequences_processed']} sequences")
            
            # Step 5: Clean up activation files to save disk space
            import shutil
            from pathlib import Path
            run_dir_path = Path(run_dir) if isinstance(run_dir, str) else run_dir
            if run_dir_path.exists():
                disk_freed_mb = sum(f.stat().st_size for f in run_dir_path.rglob('*.pt')) / (1024*1024)
                shutil.rmtree(run_dir_path)
                logger.info(f"🗑️  Cleaned up activation files: {run_dir_path} ({disk_freed_mb:.1f}MB freed)")
            
            # Step 6: Clear GPU cache
            torch.cuda.empty_cache()
            logger.debug(f"🧹 GPU cache cleared")
        
        print(f"\n✅ All {num_batches} batches completed!")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        print(f"💾 Partial results saved: {len(all_batch_results)} batches completed")
        sys.exit(1)
    
    # Step 3: Aggregate results across all batches
    print("\n" + "="*40)
    print("STEP 3: AGGREGATING RESULTS")
    print("="*40)
    
    all_systematic_outliers = []
    all_outlier_info = []
    
    print(f"Aggregating results from {len(all_batch_results)} completed batches...")
    
    for batch_result in all_batch_results:
        all_systematic_outliers.extend(batch_result['systematic_outliers'])
        all_outlier_info.append(batch_result['outlier_info'])
        print(f"Batch {batch_result['batch_id']}: {len(batch_result['systematic_outliers'])} systematic outlier features")
        
    
    print(f"✅ Analysis completed across all {len(all_batch_results)} batches!")
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
                  f"appeared in {feature['occurrences']}/{len(all_batch_results)} batches")
    else:
        print("\nNo systematic outlier features found across any batch.")
        print("Consider:")
        print("- Using longer sequences or more complex text")
        print("- Trying different layers")
        print("- Lowering the magnitude threshold")
    
    # Save final aggregated results
    final_results = {
        "analysis_summary": {
            "total_batches": len(all_batch_results),
            "total_sequences": sum(br['sequences_processed'] for br in all_batch_results),
            "total_systematic_outliers": len(all_systematic_outliers),
            "analysis_params": all_batch_results[0]['analysis_params'] if all_batch_results else {},
            "completion_time": datetime.now().isoformat()
        },
        "top_features": top_features if all_systematic_outliers else [],
        "all_systematic_outliers": all_systematic_outliers,
        "batch_results": all_batch_results
    }
    
    final_results_file = Path(args.save_dir) / "final_analysis_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n📁 Final results saved: {final_results_file}")
    print("🎉 ANALYSIS COMPLETE")
    
    return all_batch_results, all_systematic_outliers, all_outlier_info


if __name__ == "__main__":
    main()
