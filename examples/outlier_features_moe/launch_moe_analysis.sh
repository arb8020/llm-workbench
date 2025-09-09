#!/bin/bash
# MoE Outlier Analysis Launcher
# Usage: ./launch_moe_analysis.sh [model_name] [options]

set -e

# Default configurations
DEFAULT_SEQUENCES=16
DEFAULT_LENGTH=2048
DEFAULT_BATCH_SIZE=1
DEFAULT_THRESHOLD=6.0
DEFAULT_CHUNK_LAYERS=""

# Model configurations function
get_model_config() {
    case "$1" in
        "glm_4_5_air")
            echo "zai-org/GLM-4.5-Air,1,H100,80,500,2"
            ;;
        "olmoe_1b_7b")
            echo "allenai/OLMoE-1B-7B-0924-Instruct,1,,16,150,"
            ;;
        "qwen3_30b_a3b")
            echo "Qwen/Qwen3-30B-A3B,1,A100,80,,"
            ;;
        # "llama_scout")
        #     echo "meta-llama/Llama-3.2-1B-Scout,1,,16,,"
        #     ;;
        # "llama_maverick")
        #     echo "meta-llama/Llama-3.2-3B-Maverick,1,,24,,"
        #     ;;
        "gpt_oss_120b")
            echo "openai/gpt-oss-120b,1,H100,80,,"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Get list of available models
get_available_models() {
    echo "glm_4_5_air olmoe_1b_7b gpt_oss_120b qwen3_30b_a3b"
}

# Parse command line arguments
SELECTED_MODEL=""
NUM_SEQUENCES=$DEFAULT_SEQUENCES
SEQUENCE_LENGTH=$DEFAULT_LENGTH
BATCH_SIZE=$DEFAULT_BATCH_SIZE
THRESHOLD=$DEFAULT_THRESHOLD
CHUNK_LAYERS=$DEFAULT_CHUNK_LAYERS
KEEP_RUNNING="--keep-running" # Keep GPU running by default for debugging
DRY_RUN=false

usage() {
    echo "Usage: $0 [model_name] [options]"
    echo ""
    echo "Available models:"
    for model in $(get_available_models); do
        echo "  $model"
    done
    echo ""
    echo "Options:"
    echo "  --sequences N          Number of sequences (default: $DEFAULT_SEQUENCES)"
    echo "  --length N             Sequence length (default: $DEFAULT_LENGTH)"
    echo "  --batch-size N         Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --threshold N          Outlier threshold (default: $DEFAULT_THRESHOLD)"
    echo "  --chunk-layers N       Number of layers to process at once (default: model-specific)"
    echo "  --keep-running         Keep GPU running after analysis (default)"
    echo "  --no-keep-running      Clean up GPU after analysis"
    echo "  --dry-run              Show commands without executing"
    echo "  --all                  Launch all models"
    echo ""
    echo "Examples:"
    echo "  $0 olmoe_1b_7b"
    echo "  $0 glm_4_5_air --sequences 32 --length 1024"
    echo "  $0 --all --sequences 8"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sequences)
            NUM_SEQUENCES="$2"
            shift 2
            ;;
        --length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --chunk-layers)
            CHUNK_LAYERS="$2"
            shift 2
            ;;
        --keep-running)
            KEEP_RUNNING="--keep-running"
            shift
            ;;
        --no-keep-running)
            KEEP_RUNNING=""
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --all)
            SELECTED_MODEL="all"
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [[ -z "$SELECTED_MODEL" ]]; then
                SELECTED_MODEL="$1"
            else
                echo "Multiple model names specified"
                usage
            fi
            shift
            ;;
    esac
done

# Validate model selection
if [[ -z "$SELECTED_MODEL" ]]; then
    echo "Error: No model specified"
    usage
fi

if [[ "$SELECTED_MODEL" != "all" && -z "$(get_model_config "$SELECTED_MODEL")" ]]; then
    echo "Error: Unknown model '$SELECTED_MODEL'"
    usage
fi

# Function to launch a single model
launch_model() {
    local model_key="$1"
    local config="$(get_model_config "$model_key")"
    
    # Parse config
    IFS=',' read -r model_name gpu_count gpu_filter min_vram container_disk model_chunk_layers <<< "$config"
    
    # Build command
    local cmd="python examples/outlier_features_moe/deploy_and_analyze.py"
    cmd+=" --model \"$model_name\""
    cmd+=" --gpu-count $gpu_count"
    cmd+=" --num-sequences $NUM_SEQUENCES"
    cmd+=" --sequence-length $SEQUENCE_LENGTH"
    cmd+=" --batch-size $BATCH_SIZE"
    cmd+=" --threshold $THRESHOLD"
    
    # Use command line chunk layers if provided, otherwise use model default
    local effective_chunk_layers="${CHUNK_LAYERS:-$model_chunk_layers}"
    [[ -n "$effective_chunk_layers" ]] && cmd+=" --chunk-layers $effective_chunk_layers"
    
    # Add optional parameters
    [[ -n "$gpu_filter" ]] && cmd+=" --gpu-filter \"$gpu_filter\""
    [[ -n "$min_vram" ]] && cmd+=" --min-vram $min_vram"
    [[ -n "$container_disk" ]] && cmd+=" --container-disk $container_disk"
    [[ -n "$KEEP_RUNNING" ]] && cmd+=" $KEEP_RUNNING"
    
    # Add logging
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="examples/outlier_features_moe/logs/${model_key}_${timestamp}.log"
    cmd+=" 2>&1 | tee $log_file"
    
    # Session name
    local session_name="${model_key}_analysis"
    
    # Full tmux command
    local tmux_cmd="tmux new-session -d -s '$session_name' '$cmd'"
    
    echo "🚀 Launching $model_key ($model_name)"
    echo "   Session: $session_name"
    echo "   Log: $log_file"
    echo "   GPU: $gpu_count x ${gpu_filter:-any} (${min_vram}GB VRAM)"
    [[ -n "$container_disk" ]] && echo "   Disk: ${container_disk}GB"
    echo ""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "   Command: $tmux_cmd"
        echo ""
    else
        eval "$tmux_cmd"
        sleep 2  # Brief pause between launches
    fi
    
    # Return log file path for monitoring
    echo "$log_file"
}

# Create logs directory if it doesn't exist
mkdir -p examples/outlier_features_moe/logs

# Launch models
LAUNCHED_LOG_FILES=()

if [[ "$SELECTED_MODEL" == "all" ]]; then
    echo "🎯 Launching all MoE models with:"
    echo "   Sequences: $NUM_SEQUENCES x $SEQUENCE_LENGTH tokens"
    echo "   Batch size: $BATCH_SIZE"
    echo "   Threshold: $THRESHOLD"
    echo ""
    
    for model in $(get_available_models); do
        log_file=$(launch_model "$model")
        LAUNCHED_LOG_FILES+=("$log_file")
    done
else
    echo "🎯 Launching $SELECTED_MODEL with:"
    echo "   Sequences: $NUM_SEQUENCES x $SEQUENCE_LENGTH tokens" 
    echo "   Batch size: $BATCH_SIZE"
    echo "   Threshold: $THRESHOLD"
    echo ""
    
    log_file=$(launch_model "$SELECTED_MODEL")
    LAUNCHED_LOG_FILES+=("$log_file")
fi

if [[ "$DRY_RUN" != "true" ]]; then
    echo "✅ All deployments launched!"
    echo ""
    echo "Monitor progress:"
    echo "  tmux list-sessions"
    echo "  tmux capture-pane -t <session_name> -p"
    echo ""
    echo "Tail specific log files:"
    for log_file in "${LAUNCHED_LOG_FILES[@]}"; do
        echo "  tail -f $log_file"
    done
    echo ""
    echo "Active sessions:"
    tmux list-sessions 2>/dev/null | grep "_analysis" || echo "  (none yet)"
fi
