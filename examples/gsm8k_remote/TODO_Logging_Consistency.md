# TODO: Make Logging Consistent with Broker CLI

## Problem
The gsm8k_remote example script uses different logging format/style than the broker CLI, making the output inconsistent.

## Examples of Inconsistency

**Broker CLI style** (from `broker instances list`):
```
📋 Listing all instances...
✅ Found instance: 053a3u1fefyf9s
   Status: InstanceStatus.RUNNING
   GPU: RTX A5000 x1
   Price: $0.270/hr
🗑️ Terminating instance: 053a3u1fefyf9s
✅ Instance 053a3u1fefyf9s terminated successfully
```

**Script style** (from deploy_and_evaluate.py):
```
2025-09-29 13:13:17,675 - __main__ - INFO - ✅ GPU provisioned: 053a3u1fefyf9s
2025-09-29 13:13:17,675 - __main__ - INFO - ⏳ Waiting for SSH connection to be ready...
```

## Issues
1. Broker CLI doesn't use timestamps or logger prefixes (cleaner)
2. Script uses full logger format (more verbose but harder to read)
3. Different formatting for similar operations
4. User sees different styles when using broker CLI vs script

## Potential Solutions

**Option A**: Match broker CLI style
- Remove timestamps from console output
- Keep JSONL file with full structured logs
- Console shows clean, emoji-heavy output like CLI

**Option B**: Configure different formatters
- Console: Minimal format (just message)
- File: Full structured JSONL
- Already partially done with `use_json=False` for console

**Option C**: Use broker/bifrost logging config
- Import and use their logging setup
- Ensures consistency across all tools

## Current Implementation
`shared/logging_config.py` already supports custom formatters:
```python
formatters = {
    "standard": {
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

Could change to:
```python
formatters = {
    "standard": {
        "format": "%(message)s"  # Just the message, no prefix
    }
}
```

## Files to Update
- `shared/logging_config.py` - Simplify console formatter
- `examples/gsm8k_remote/deploy_and_evaluate.py` - Verify it matches broker style

## Testing
Compare side-by-side:
```bash
broker instances list
python examples/gsm8k_remote/deploy_and_evaluate.py --samples 1
```

Both should have similar look and feel.
