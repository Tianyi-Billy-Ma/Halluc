#!/bin/bash

# Parse mode argument and set flags for which phases to run
# Usage: 
#   Direct execution: source ./bash/sys/parse_mode.sh <mode>
#   qsub with env var: qsub -v MODE=full e2e.sh
#   mode can be: "full", "train", "merge", "eval", or comma-separated combinations

# Get mode from environment variable (for qsub) or command-line argument (for direct execution)
# Priority: environment variable > command-line argument > default to 'full'
if [ -n "$MODE" ]; then
    # MODE is set as environment variable (e.g., from qsub -v MODE=full)
    MODE_INPUT="$MODE"
elif [ -n "$1" ]; then
    # MODE is passed as command-line argument (e.g., ./e2e.sh full)
    MODE_INPUT="$1"
else
    # Default to 'full' if nothing is passed
    MODE_INPUT="full"
fi

# Store original mode for display purposes
MODE="$MODE_INPUT"

# Normalize mode: if 'full', expand to all phases
if [ "$MODE_INPUT" = "full" ]; then
    MODE_INPUT="train,merge,eval"
fi

# Convert comma-separated mode to flags
DO_TRAIN=false
DO_MERGE=false
DO_EVAL=false

IFS=',' read -ra MODES <<< "$MODE_INPUT"
for m in "${MODES[@]}"; do
    case "${m,,}" in  # Convert to lowercase
        train)
            DO_TRAIN=true
            ;;
        merge)
            DO_MERGE=true
            ;;
        eval|evaluate)
            DO_EVAL=true
            ;;
        *)
            echo "Warning: Unknown mode '$m'. Ignoring."
            ;;
    esac
done

