# THUDM Slime Architecture Analysis

## Overview

THUDM Slime is an LLM post-training framework for RL scaling that provides high-performance training by connecting Megatron with SGLang. This document analyzes how Slime establishes interfaces between its modules and identifies its pure function patterns.

## Core Architecture & Module Interfaces

### Three Main Modules

1. **Training (Megatron)** - `/slime/backends/megatron_utils/`
   - Responsible for the main training process
   - Reads data from the Data Buffer
   - Synchronizes parameters to the rollout module after training

2. **Rollout (SGLang + router)** - `/slime/rollout/`  
   - Generates new data (including rewards/verifier outputs)
   - Stores generated data in the Data Buffer

3. **Data Buffer** - `/slime/ray/buffer.py`
   - Bridge module that manages prompt initialization
   - Handles custom data and rollout generation methods

## Key Interface Patterns

### Primary Data Contract

The `Sample` class (`/slime/utils/types.py:9-45`) is the core data structure flowing between modules:

```python
@dataclass
class Sample:
    index: Optional[int] = None
    # prompt
    prompt: Union[str, list[dict[str, str]]] = ""
    tokens: list[int] = field(default_factory=list)
    # response
    response: str = ""
    response_length: int = 0
    label: Optional[str] = None
    reward: Optional[Union[float, dict[str, Any]]] = None
    loss_mask: Optional[list[int]] = None
    rollout_log_probs: Optional[list[float]] = None  # Log probabilities from rollout engine
    
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
    
    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)
```

### Module Responsibilities

#### Training Module (`MegatronTrainRayActor`)
- **Input:** `rollout_data_ref` (list of Samples with responses & rewards)
- **Function:** `async_train(rollout_id, rollout_data_ref)` 
- **Output:** Updated model weights
- **Pure Transform:** `samples_with_rewards → updated_model_weights`

#### Rollout Module (`SGLangRolloutManager`)
- **Input:** Model weights + prompts from Dataset
- **Function:** `async_generate(rollout_id)`
- **Output:** `rollout_data_ref` (Samples with generated responses)
- **Pure Transform:** `prompts + model_weights → samples_with_responses`

#### Reward Module (`rm_hub`)
- **Input:** Sample with prompt + response
- **Function:** `async_rm(args, sample)` / `batched_async_rm(args, samples)`
- **Output:** Reward score (float/int)
- **Pure Transform:** `(prompt, response, label) → reward_score`

## Pure Function Examples

### Data Processing
Location: `/slime/utils/data.py:11-21`
```python
def read_file(path) -> Iterator[dict]:
    """Pure function: file_path → data_stream"""
    if path.endswith(".jsonl") or path.endswith(".json"):
        ds = hf_ds.from_json(path)
    elif path.endswith(".parquet"):
        ds = hf_ds.from_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    for data in ds:
        yield data
```

### Reward Computation
Location: `/slime/rollout/rm_hub/__init__.py:29-55`
```python
async def async_rm(args, sample: Sample) -> Union[int, float]:
    """Pure function: (sample.response, sample.label) → reward_score"""
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    rm_type = args.rm_type
    response = sample.response
    label = sample.label
    
    if rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    # ... other reward types
```

### Dataset Transformation
Location: `/slime/utils/data.py:23-79`
```python
class Dataset:
    """Transforms raw data into Sample objects"""
    def __init__(self, path, tokenizer, max_length, **kwargs):
        self.origin_samples = []
        for data in read_file(path):
            prompt = data[prompt_key]
            if apply_chat_template:
                prompt = tokenizer.apply_chat_template(
                    prompt, tools, tokenize=False, add_generation_prompt=True
                )
            
            self.origin_samples.append(
                Sample(
                    prompt=prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=data.get(metadata_key) or {},
                )
            )
```

## Data Flow Pattern

```
Dataset → Rollout → RM_Hub → Training → Model_Update
   ↓         ↓        ↓         ↓          ↓
prompts → responses → rewards → loss → new_weights
```

### Detailed Flow

1. **Dataset** (`/slime/utils/data.py`)
   - Transforms: `raw_data_files → Sample_objects_with_prompts`

2. **Rollout** (`/slime/rollout/sglang_rollout.py`)
   - Transforms: `Sample_objects_with_prompts + model_weights → Sample_objects_with_responses`

3. **Reward Model Hub** (`/slime/rollout/rm_hub/`)
   - Transforms: `Sample_objects_with_responses → Sample_objects_with_rewards`

4. **Training** (`/slime/backends/megatron_utils/actor.py`)
   - Transforms: `Sample_objects_with_rewards → model_weight_updates`

5. **Weight Update** (`/slime/backends/megatron_utils/update_weight_utils.py`)
   - Transforms: `model_weight_updates → updated_model_weights`

## Interface Contracts

### Core Abstractions

- **Sample**: Universal data container that flows between all modules
- **Dataset**: Provides standardized data loading and preprocessing
- **RayTrainGroup**: Manages distributed training actors
- **RolloutManager**: Coordinates inference engines and data generation
- **PlacementGroup**: Handles GPU allocation and resource management

### Communication Patterns

- **Asynchronous Ray calls**: All inter-module communication uses `ray.get()` and `async_*` methods
- **Data references**: Large data structures passed as Ray object references to avoid serialization overhead
- **Status tracking**: Sample objects carry status information for pipeline coordination

## Key Design Principles

1. **Functional Purity**: Core transformation functions are pure - same inputs always produce same outputs
2. **Clear Boundaries**: Each module has well-defined responsibilities and interfaces  
3. **Type Safety**: Strong typing with dataclasses for all data contracts
4. **Distributed Coordination**: Ray framework handles distribution while maintaining functional core
5. **Extensibility**: Plugin system for custom reward models and data processing functions

Each module consumes well-defined inputs (primarily `Sample` objects) and produces well-defined outputs, with clear transformation responsibilities. The system uses Ray for distributed coordination but maintains functional purity within individual transformation steps.