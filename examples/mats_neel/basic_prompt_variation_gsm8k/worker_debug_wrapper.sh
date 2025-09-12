#!/bin/bash
# Debug wrapper for worker_experiment.py that captures all output
# 
# CRITICAL: This wrapper exists because remote machines need 'uv run python' not 'python'
# - Direct 'python' calls fail with "No module named rollouts.evaluation"  
# - Only 'uv run python' properly sets up the dependencies environment
# - This wrapper also tests imports before running to catch errors early
#
# Usage: worker_debug_wrapper.sh <config_path> <worker_id> <log_file_path>

CONFIG_PATH="$1"
WORKER_ID="$2"
LOG_FILE="$3"

# Expand log file path and ensure directory exists
LOG_FILE_EXPANDED="${LOG_FILE/#~/$HOME}"
mkdir -p "$(dirname "$LOG_FILE_EXPANDED")"

# Create debug log function
debug_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEBUG] $1" | tee -a "$LOG_FILE_EXPANDED"
}

error_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" | tee -a "$LOG_FILE_EXPANDED"
}

# Start logging immediately
debug_log "=== Worker Debug Wrapper Starting ==="
debug_log "Worker ID: $WORKER_ID"
debug_log "Config Path: $CONFIG_PATH"
debug_log "Log File: $LOG_FILE_EXPANDED"
debug_log "Current Directory: $(pwd)"
debug_log "Python Version: $(python --version 2>&1)"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    error_log "Config file not found: $CONFIG_PATH"
    exit 1
fi

debug_log "Config file exists and is readable"

# Check if worker script exists
WORKER_SCRIPT="examples/mats_neel/basic_prompt_variation_gsm8k/worker_experiment.py"
if [ ! -f "$WORKER_SCRIPT" ]; then
    error_log "Worker script not found: $WORKER_SCRIPT"
    exit 1
fi

debug_log "Worker script exists: $WORKER_SCRIPT"

# Test critical imports before running the real script
debug_log "Testing critical imports..."

uv run python -c "
import sys
import logging
import json
import os
import time
print('Basic imports: OK')
" 2>&1 | tee -a "$LOG_FILE_EXPANDED"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error_log "Basic imports failed"
    exit 1
fi

uv run python -c "
from rollouts.evaluation import evaluate_sample, load_jsonl
from rollouts.dtypes import Message, Endpoint, AgentState, RunConfig
print('Rollouts imports: OK')
" 2>&1 | tee -a "$LOG_FILE_EXPANDED"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error_log "Rollouts imports failed - this is likely the issue"
    exit 1
fi

uv run python -c "
import requests
import asyncio
print('Additional imports: OK')
" 2>&1 | tee -a "$LOG_FILE_EXPANDED"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error_log "Additional imports (requests/asyncio) failed"
    exit 1
fi

debug_log "All imports successful - starting worker script"

# Run the actual worker script and capture all output
debug_log "=== Starting Python Worker Script ==="

uv run python "$WORKER_SCRIPT" "$CONFIG_PATH" "$WORKER_ID" "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE_EXPANDED"
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    debug_log "=== Worker script completed successfully ==="
else
    error_log "=== Worker script failed with exit code $EXIT_CODE ==="
fi

exit $EXIT_CODE