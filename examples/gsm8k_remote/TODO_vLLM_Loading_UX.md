# TODO: Improve vLLM Loading UX

## Problem
When the script waits for vLLM to be ready, it's polling `/v1/models` endpoint but getting 502 errors because the server is still loading the model. From the user's perspective, it just says "Server returned status 502, waiting..." which doesn't explain what's happening.

## What's Actually Happening
Looking at the vLLM logs, the server goes through several stages:
1. Model download (~2s)
2. Model loading (~3s) 
3. torch.compile (~54s) ← **This is the long part**
4. KV cache allocation (~1s)
5. CUDA graph capture (~3s)
6. Total: ~64s before server is ready

During this time, the HTTP server is running but returns 502 because the model isn't ready yet.

## Proposed Solution
Add more informative logging during the wait:
- Parse the vLLM logs to detect which stage it's in
- Show progress like "⏳ vLLM compiling model (this takes ~1 minute)..."
- Or: "⏳ Model loading in progress - torch.compile stage (50s remaining)..."

## Implementation Ideas
1. **Option A**: Tail the remote log file and parse it for progress
   ```python
   if "torch.compile" in log_line:
       logger.info("   Model is compiling (takes ~50s, please wait)...")
   ```

2. **Option B**: Show elapsed time with context
   ```python
   logger.info(f"   [{elapsed}s] Server loading model (expect ~60s total)...")
   ```

3. **Option C**: Check tmux session for progress indicators
   ```python
   tmux_output = bifrost_client.exec("tmux capture-pane -t qwen-vllm -p")
   if "Capturing CUDA graphs" in tmux_output:
       logger.info("   Almost ready - finalizing CUDA graphs...")
   ```

## Current Code Location
`examples/gsm8k_remote/deploy_and_evaluate.py:39-92` in `wait_for_vllm_ready()`

Currently just logs generic "Server returned status 502, waiting..." messages.
