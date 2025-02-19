from typing import List, Dict
import os

# Model Configurations for Evaluation
MODELS_TO_EVALUATE = [
    {
        "name": "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
        "type": "base",
        "description": "Base Llama 3 8B model"
    },
    {
        "name": "saved_models/grpo_final",
        "type": "finetuned",
        "description": "GRPO finetuned model"
    }
]

# Evaluation Settings
EVAL_BATCH_SIZE = 4
NUM_EVAL_SAMPLES = 100  # Number of samples to evaluate in dry run
FULL_EVAL_SAMPLES = -1  # -1 means use all available samples

# Metrics Configuration
METRICS_TO_TRACK = [
    "script_similarity",
    "execution_success",
    "response_time",
    "memory_usage",
    "format_compliance",
    "reasoning_quality"
]

# Monitoring Configuration
TENSORBOARD_LOG_DIR = "runs"
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "sharknet_evaluation"

# Resource Monitoring
MONITOR_GPU = True
MONITOR_MEMORY = True
MONITOR_RESPONSE_TIME = True
MONITORING_INTERVAL = 1.0  # seconds

# Dry Run Configuration
DRY_RUN_SAMPLES = 5
DRY_RUN_BATCH_SIZE = 2

# Output Configuration
EVAL_OUTPUT_DIR = "evaluation_results"
VISUALIZATION_OUTPUT_DIR = "visualizations"

# Create necessary directories
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True) 