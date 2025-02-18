#!/bin/bash
set -e

# Generate a unique run identifier
RUN_ID=$(date +%Y%m%d_%H%M%S)_${MODEL_NAME##*/}

# Create output directory for this run
mkdir -p output/$RUN_ID

# Update config with environment variables
cat > pipeline/runtime_config.py << EOL
import os
import torch

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training Configuration
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))

# Data Configuration
DATA_PATH = os.getenv("DATA_PATH", "prototype/RLdata_unix.txt")
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", "0.8"))

# Output Configuration
OUTPUT_DIR = f"output/{os.getenv('RUN_ID')}"
MODEL_SAVE_DIR = f"saved_models/{os.getenv('RUN_ID')}"

# Logging configuration
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", "100"))
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", "500"))
EOL

# Export the run ID for use in Python
export RUN_ID

echo "Starting training with configuration:"
echo "Model: $MODEL_NAME"
echo "Run ID: $RUN_ID"
echo "Training mode: $TRAINING_MODE"

# Run training
python -m pipeline.main --mode $TRAINING_MODE

# Run inference on test cases
echo "Running inference on test cases..."
python inference.py --model-dir "saved_models/$RUN_ID" --output-dir "output/$RUN_ID/test_results"

# Generate comparison metrics
echo "Generating comparison metrics..."
python -m pipeline.metrics --results-dir "output/$RUN_ID/test_results" --output-file "output/$RUN_ID/metrics.json"

echo "Training and evaluation complete. Results available in output/$RUN_ID/" 