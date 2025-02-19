#!/bin/bash

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate pool

# Default values
CONFIG_PATH="config/evaluation_config.yml"
TRAINING_MODE="standard"
SKIP_TRAINING=false
EVAL_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python run_pipeline.py --config ${CONFIG_PATH} --training-mode ${TRAINING_MODE}"

if [ "$SKIP_TRAINING" = true ]; then
    CMD="${CMD} --skip-training"
fi

if [ "$EVAL_ONLY" = true ]; then
    CMD="${CMD} --eval-only"
fi

# Run the pipeline
echo "Running command: ${CMD}"
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline failed!"
    exit 1
fi 