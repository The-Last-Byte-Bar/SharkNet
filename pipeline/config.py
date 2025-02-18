import torch

# Model Configuration
MODEL_NAME = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
MAX_SEQ_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training Configuration
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 4

# Data Configuration
DATA_PATH = "prototype/RLdata_unix.txt"
TRAIN_TEST_SPLIT = 0.8

# Output Configuration
OUTPUT_DIR = "output"
MODEL_SAVE_DIR = "saved_models"

# Logging configuration
LOG_INTERVAL = 100  # Log every N steps
EVAL_INTERVAL = 500  # Evaluate every N steps 