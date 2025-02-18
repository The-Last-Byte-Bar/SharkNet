FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make output directories
RUN mkdir -p output saved_models

# Set environment variables with defaults
ENV MODEL_NAME="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit" \
    MAX_SEQ_LENGTH=512 \
    LEARNING_RATE=2e-5 \
    BATCH_SIZE=4 \
    NUM_EPOCHS=3 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    TRAIN_TEST_SPLIT=0.8 \
    TRAINING_MODE="standard"

# Create entrypoint script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"] 