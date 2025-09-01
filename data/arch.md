# Data Architecture

## Overview

The data module provides unified data loading and preprocessing for all training types (pretraining, SFT, RL). It abstracts away format differences, tokenization strategies, and distributed data handling, exposing a clean interface to the trainer.

## Core Principles

1. **Format Agnostic**: Handle JSONL, Parquet, raw text, rollout data uniformly
2. **Streaming First**: Memory-efficient processing for large datasets  
3. **Distribution Aware**: Built-in sharding and load balancing
4. **Extensible**: Plugin system for custom preprocessing pipelines

## Directory Structure

```
data/
├── loaders/
│   ├── base.py              # DataLoader abstract interface
│   ├── conversation.py      # SFT conversation data (chat format)
│   ├── pretrain.py          # Raw text chunks for pretraining
│   ├── rollout.py           # RL rollout data from inference engines
│   └── mixed.py             # Multi-source dataset mixing
├── preprocessing/
│   ├── tokenization.py      # Tokenization strategies and caching
│   ├── formatting.py        # Chat templates and prompt formatting
│   ├── filtering.py         # Quality filtering and deduplication
│   └── transforms.py        # Custom data transformations
├── distributed/
│   ├── sharding.py          # Data parallel sharding strategies
│   ├── balancing.py         # Sequence length balancing
│   └── synchronization.py   # Multi-worker coordination
├── streaming/
│   ├── memory.py            # Memory-efficient streaming
│   ├── caching.py           # Disk caching and prefetching
│   └── compression.py       # On-the-fly compression/decompression
└── types.py                 # Common data structures and interfaces
```

## Core Interfaces

### DataLoader Interface
```python
@abstractmethod
class DataLoader:
    def get_batch(self, batch_size: int) -> TrainingBatch:
        """Returns a single batch ready for training"""
        
    def get_iterator(self) -> Iterator[TrainingBatch]:
        """Returns iterator for full epoch/dataset"""
        
    def get_num_batches(self) -> int:
        """Returns total number of batches"""
        
    def reset(self) -> None:
        """Reset iterator to beginning"""
```

### TrainingBatch Structure
```python
@dataclass
class TrainingBatch:
    input_ids: torch.Tensor          # Tokenized input sequences
    attention_mask: torch.Tensor     # Attention masks
    labels: torch.Tensor             # Training targets
    loss_mask: Optional[torch.Tensor] # Which tokens to compute loss on
    metadata: Dict[str, Any]         # Additional context (rewards, etc.)
```

## Usage Examples

### SFT Training
```python
loader = ConversationDataLoader(
    path="sft_data.jsonl",
    tokenizer=tokenizer,
    max_length=2048,
    chat_template="chatml"
)
batch = loader.get_batch(batch_size=32)
```

### Pretraining
```python
loader = PretrainDataLoader(
    path="corpus/",
    tokenizer=tokenizer,
    chunk_size=2048,
    streaming=True
)
for batch in loader.get_iterator():
    # Train on batch
```

### RL Training
```python
loader = RolloutDataLoader(
    rollout_data=rollout_samples,
    tokenizer=tokenizer,
    compute_advantages=True
)
batch = loader.get_batch(batch_size=16)  # Includes rewards, advantages
```

## Integration Points

- **Trainer Module**: Consumes `TrainingBatch` objects
- **Engine Module**: Provides tokenizers and model configs
- **Rollouts Module**: Provides rollout data for RL training
- **Shared Module**: Logging and utilities

## Extension Points

1. **Custom Preprocessors**: Implement `DataProcessor` interface
2. **Format Plugins**: Add new file format support
3. **Distributed Strategies**: Custom sharding/balancing logic
4. **Caching Backends**: Different storage backends (Redis, S3, etc.)