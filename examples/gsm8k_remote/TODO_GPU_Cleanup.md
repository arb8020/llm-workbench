# TODO: Fix GPU Instance Cleanup/Termination

## Problem
The cleanup code in the script **does not actually terminate GPU instances**. It just logs a message saying what command to run, but doesn't execute the termination.

## Evidence
From the logs of failed runs, we saw:
```
logger.info(f"   Terminating GPU instance {connection_info['instance_id']}...")
# Note: We need to implement terminate method or use existing patterns
# For now, print the command
logger.info(f"   Run: broker terminate {connection_info['instance_id']}")
logger.info("✅ Cleanup complete")
```

This resulted in **4 GPUs running simultaneously**, wasting money ($0.27/hr each = ~$1/hr total).

## Current Code Location
`examples/gsm8k_remote/deploy_and_evaluate.py:707-714` in the `finally` block of `main()`

## Root Cause
The code has a TODO comment and just prints the termination command instead of actually calling the API.

## Solution
The `GPUInstance` class already has a `terminate()` method that works correctly (we used it to clean up). We need to:

1. **Store the GPU client** so it's available in the finally block
2. **Call the terminate API** instead of just logging

### Implementation
```python
# In finally block around line 707:
try:
    logger.info(f"   Terminating GPU instance {connection_info['instance_id']}...")
    
    # Get instance and terminate it
    from broker.api import terminate_instance
    result = terminate_instance(
        connection_info['instance_id'],
        provider='runpod',
        api_key=RUNPOD_API_KEY
    )
    
    if result:
        logger.info("✅ GPU instance terminated successfully")
    else:
        logger.error("❌ Failed to terminate GPU instance")
        logger.error(f"   Manual cleanup: broker instances terminate {connection_info['instance_id']}")

except Exception as cleanup_error:
    logger.error(f"⚠️  Failed to terminate GPU instance: {cleanup_error}")
    logger.error(f"   IMPORTANT: Manually terminate to stop billing:")
    logger.error(f"   broker instances terminate {connection_info['instance_id']}")
```

## Why This Matters
- Each forgotten GPU costs $0.27/hr
- Failed runs leave GPUs running indefinitely
- In our testing session, 4 GPUs accumulated quickly
- Manual cleanup via `broker instances terminate` required

## Testing
After fixing, verify:
1. Run script and let it fail → GPU should be terminated
2. Check `broker instances list` → Should show 0 instances
3. Run with `--keep-running` → GPU should stay running
