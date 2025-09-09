"""
Pure functions for loading and processing datasets for outlier analysis.
"""

from typing import List, Iterator, Optional
from datasets import load_dataset
from transformers import AutoTokenizer


def load_streaming_dataset(dataset_name: str, split: str = "train") -> Iterator[str]:
    """
    Load streaming dataset and yield text content.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "HuggingFaceFW/fineweb-edu")
        split: Dataset split to use
        
    Yields:
        str: Text content from dataset
    """
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        for item in dataset:
            yield item["text"]
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")


def chunk_text_by_chars(text: str, chunk_size: int) -> List[str]:
    """
    Split text into character-based chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        
    Returns:
        List of text chunks
    """
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
    assert isinstance(text, str), f"Expected str, got {type(text)}"
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)
    
    return chunks


def chunk_text_by_tokens(text: str, tokenizer: AutoTokenizer, chunk_size: int) -> List[str]:
    """
    Split text into token-based chunks.
    
    Args:
        text: Input text to chunk
        tokenizer: HuggingFace tokenizer to use for tokenization
        chunk_size: Target size of each chunk in tokens
        
    Returns:
        List of text chunks, each approximately chunk_size tokens
    """
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
    assert isinstance(text, str), f"Expected str, got {type(text)}"
    
    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():  # Skip empty chunks
            chunks.append(chunk_text)
    
    return chunks


def get_text_sequences(
    dataset_name: str, 
    num_sequences: int, 
    sequence_length: int,
    tokenizer: Optional[AutoTokenizer] = None,
    split: str = "train"
) -> List[str]:
    """
    Get N sequences of specified length from streaming dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        num_sequences: Number of sequences to extract
        sequence_length: Target length of each sequence (in tokens if tokenizer provided, chars otherwise)
        tokenizer: Optional tokenizer for token-based chunking. If None, uses character-based chunking
        split: Dataset split to use
        
    Returns:
        List of text sequences, each approximately sequence_length tokens/characters
    """
    assert num_sequences > 0, f"num_sequences must be positive, got {num_sequences}"
    assert sequence_length > 0, f"sequence_length must be positive, got {sequence_length}"
    
    sequences = []
    current_text = ""
    
    dataset_stream = load_streaming_dataset(dataset_name, split)
    
    try:
        for text in dataset_stream:
            current_text += " " + text  # Add space between documents
            
            # Determine if we have enough content for sequence extraction
            if tokenizer is not None:
                # Token-based: check if we have enough tokens
                current_tokens = tokenizer.encode(current_text, add_special_tokens=False)
                content_length = len(current_tokens)
                min_needed = sequence_length
            else:
                # Character-based: check if we have enough characters
                content_length = len(current_text)
                min_needed = sequence_length
            
            # Extract sequences while we have enough content
            while content_length >= min_needed and len(sequences) < num_sequences:
                if tokenizer is not None:
                    # Token-based extraction
                    tokens = tokenizer.encode(current_text, add_special_tokens=False)
                    sequence_tokens = tokens[:sequence_length]
                    sequence = tokenizer.decode(sequence_tokens, skip_special_tokens=True)
                    
                    # Remove extracted tokens plus some overlap for fresh content
                    overlap_size = sequence_length // 2
                    remaining_tokens = tokens[overlap_size:]
                    current_text = tokenizer.decode(remaining_tokens, skip_special_tokens=True)
                    current_tokens = remaining_tokens
                    content_length = len(current_tokens)
                else:
                    # Character-based extraction (original logic)
                    sequence = current_text[:sequence_length]
                    current_text = current_text[sequence_length // 2:]  # 50% overlap
                    content_length = len(current_text)
                
                sequences.append(sequence)
                
            if len(sequences) >= num_sequences:
                break
                
    except Exception as e:
        raise RuntimeError(f"Failed to extract sequences from {dataset_name}: {e}")
    
    assert len(sequences) == num_sequences, f"Could only extract {len(sequences)} sequences, needed {num_sequences}"
    
    return sequences