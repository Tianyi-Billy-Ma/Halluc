#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=4     # Run on 1 GPU card
#$ -N llmhalluc      # Specify job name



# Parse mode argument and set phase flags
source ./bash/sys/parse_mode.sh "$1"

source ./bash/sys/init_env.sh llmhalluc

# Resolve experiment configuration via helper script
E2E_CONFIG_PATH=${E2E_CONFIG_PATH:-"./configs/llmhalluc/e2e.yaml"}
E2E_SETUP_CMD=(python -m llmhalluc.scripts.e2e_setup --format shell --config "$E2E_CONFIG_PATH" \
    --do-train "$DO_TRAIN" \
    --do-merge "$DO_MERGE" \
    --do-eval "$DO_EVAL")

if [ -n "${E2E_OVERRIDES:-}" ]; then
    for kv in ${E2E_OVERRIDES}; do
        E2E_SETUP_CMD+=(--override "$kv")
    done
fi

echo "================================================"
echo "Experiment Config: $E2E_CONFIG_PATH"
echo "================================================"

if ! E2E_VARS="$("${E2E_SETUP_CMD[@]}")"; then
    echo "Failed to generate stage configs. Exiting."
    exit 1
fi

eval "$E2E_VARS"

echo "Generated configs:"
if [ "$DO_TRAIN" = true ]; then
    echo "  Train -> $TRAIN_CONFIG_PATH"
fi
if [ "$DO_MERGE" = true ]; then
    echo "  Merge -> $MERGE_CONFIG_PATH"
fi
if [ "$DO_EVAL" = true ]; then
    echo "  Eval  -> $EVAL_CONFIG_PATH"
fi
if [ "$DO_TRAIN" = true ] || [ "$DO_MERGE" = true ]; then
    echo "  Special Tokens -> $SPECIAL_TOKEN_CONFIG_PATH"
fi
echo "================================================"

if [ "$DO_TRAIN" = true ]; then
    echo "================================================"
    echo "Training Phase"
    echo "================================================"

    source ./bash/sys/init_env.sh llamafactory

    ./bash/sys/log_yaml.sh $TRAIN_CONFIG_PATH
    llamafactory-cli train $TRAIN_CONFIG_PATH
else
    echo "Skipping Training Phase (not in mode: $MODE)"
fi

if [ "$DO_MERGE" = true ]; then
    echo "================================================"
    echo "Merge Models"
    echo "================================================"

    source ./bash/sys/init_env.sh llamafactory

    ./bash/sys/log_yaml.sh $MERGE_CONFIG_PATH
    llamafactory-cli export $MERGE_CONFIG_PATH
else
    echo "Skipping Merge Phase (not in mode: $MODE)"
fi 

if [ "$DO_EVAL" = true ]; then
    echo "================================================"
    echo "Evaluation Phase"
    echo "================================================"

    source ./bash/sys/init_env.sh lm_eval

    echo "================================================"
    echo "Available GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "Eval Config: ${EVAL_CONFIG_PATH}"
    echo "Accelerate: accelerate launch"
    echo "================================================"

    accelerate launch -m lm_eval --config "${EVAL_CONFIG_PATH}"
else
    echo "Skipping Evaluation Phase (not in mode: $MODE)"
fi
    
    
./bash/sys/notify.sh
