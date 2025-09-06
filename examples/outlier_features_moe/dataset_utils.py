"""
Pure functions for loading and processing datasets for outlier analysis.
"""

from typing import List, Iterator
from datasets import load_dataset


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


def get_text_sequences(
    dataset_name: str, 
    num_sequences: int, 
    sequence_length: int,
    split: str = "train"
) -> List[str]:
    """
    Get N sequences of specified character length from streaming dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        num_sequences: Number of sequences to extract
        sequence_length: Target length of each sequence in characters
        split: Dataset split to use
        
    Returns:
        List of text sequences, each approximately sequence_length characters
    """
    assert num_sequences > 0, f"num_sequences must be positive, got {num_sequences}"
    assert sequence_length > 0, f"sequence_length must be positive, got {sequence_length}"
    
    sequences = []
    current_text = ""
    
    dataset_stream = load_streaming_dataset(dataset_name, split)
    
    try:
        for text in dataset_stream:
            current_text += " " + text  # Add space between documents
            
            # Check if we have enough text to extract sequences
            while len(current_text) >= sequence_length and len(sequences) < num_sequences:
                # Extract one sequence
                sequence = current_text[:sequence_length]
                sequences.append(sequence)
                
                # Remove extracted text plus some overlap to get fresh content
                current_text = current_text[sequence_length // 2:]  # 50% overlap
                
            if len(sequences) >= num_sequences:
                break
                
    except Exception as e:
        raise RuntimeError(f"Failed to extract sequences from {dataset_name}: {e}")
    
    assert len(sequences) == num_sequences, f"Could only extract {len(sequences)} sequences, needed {num_sequences}"
    
    return sequences