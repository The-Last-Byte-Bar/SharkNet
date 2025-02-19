import torch
from typing import Optional

# Model Configuration
MODEL_NAME = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
MAX_SEQ_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# Mixed Precision Configuration
USE_MIXED_PRECISION = True
MIXED_PRECISION_DTYPE = torch.float16  # or torch.bfloat16 depending on hardware support

# Memory Optimization
USE_GRADIENT_CHECKPOINTING = True
EMPTY_CACHE_FREQUENCY = 100  # Empty CUDA cache every N steps

# Early Stopping Configuration
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.01

# Checkpoint Configuration
SAVE_TOTAL_LIMIT = 3  # Maximum number of checkpoints to keep
SAVE_STRATEGY = "steps"  # Can be "steps" or "epoch"
SAVE_STEPS = 500  # Save checkpoint every N steps if SAVE_STRATEGY="steps"

# Data Configuration
DATA_PATH = "prototype/RLdata_unix.txt"
TRAIN_TEST_SPLIT = 0.8
DATALOADER_NUM_WORKERS = 4
PIN_MEMORY = True

# Output Configuration
OUTPUT_DIR = "output"
MODEL_SAVE_DIR = "checkpoints/test_run"

# Logging Configuration
LOG_INTERVAL = 10
EVAL_INTERVAL = 50

# Validation Configuration
EVAL_STRATEGY = "steps"  # Can be "steps" or "epoch"
EVAL_STEPS = 100  # Evaluate every N steps if EVAL_STRATEGY="steps"

# Monitoring Configuration
ENABLE_TENSORBOARD = True
TENSORBOARD_UPDATE_FREQ = 10  # Update tensorboard every N steps
PLOT_METRICS_FREQ = 100  # Generate metric plots every N steps
RESOURCE_MONITORING_FREQ = 10  # Monitor system resources every N steps
MAX_KEPT_METRIC_PLOTS = 5  # Maximum number of metric plot versions to keep
MONITOR_MEMORY_USAGE = True
MONITOR_GPU_STATS = True
LOG_LEVEL = "INFO"

def get_cosine_schedule_with_warmup(optimizer, num_training_steps: int, num_warmup_steps: Optional[int] = None):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    if num_warmup_steps is None:
        num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())) 