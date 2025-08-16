# repo_utils.py - Utilities for exploring HuggingFace repositories
from __future__ import annotations
import pathlib
from typing import List

def list_repo_files(repo_id: str, pattern: str = None) -> List[str]:
    """
    List files in a HuggingFace repository.
    
    Args:
        repo_id: HuggingFace model identifier (e.g., "openai-community/gpt2")
        pattern: Optional pattern to filter files (e.g., "*.json")
    
    Returns:
        List of file paths in the repository
    """
    try:
        from huggingface_hub import list_repo_files
        
        files = list_repo_files(repo_id)
        
        if pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]
            
        return sorted(files)
        
    except ImportError:
        raise ImportError("huggingface_hub not found. Install with: pip install transformers")

def find_tokenizer_files(repo_id: str) -> dict[str, str]:
    """
    Find tokenizer files in a repository by detecting common patterns.
    
    Returns:
        Dictionary mapping file types to actual filenames
    """
    files = list_repo_files(repo_id)
    
    tokenizer_files = {}
    
    # Look for vocab/encoder files
    for f in files:
        fname = f.lower()
        if 'vocab' in fname and ('.json' in fname or '.bpe' in fname):
            if '.json' in fname:
                tokenizer_files['encoder'] = f
            elif '.bpe' in fname:
                tokenizer_files['merges'] = f
        elif 'encoder' in fname and '.json' in fname:
            tokenizer_files['encoder'] = f
        elif 'merges' in fname and ('.txt' in fname or '.bpe' in fname):
            tokenizer_files['merges'] = f
        elif fname == 'tokenizer.json':
            tokenizer_files['tokenizer_json'] = f
            
    return tokenizer_files

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="HuggingFace repository ID")
    ap.add_argument("--pattern", help="File pattern to filter (e.g., '*.json')")
    args = ap.parse_args()
    
    print(f"Files in {args.repo_id}:")
    files = list_repo_files(args.repo_id, args.pattern)
    for f in files:
        print(f"  {f}")
    
    print(f"\nTokenizer files detected:")
    tokenizer_files = find_tokenizer_files(args.repo_id)
    for file_type, filename in tokenizer_files.items():
        print(f"  {file_type}: {filename}")