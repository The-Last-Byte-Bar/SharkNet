# SharkNet - ErgoScript AI Assistant

An AI-powered assistant for learning and working with ErgoScript, built using the Llama model and optimized with Unsloth.

## Features

- Interactive Q&A about ErgoScript
- Smart contract code generation
- Explanation of ErgoScript concepts
- Training on custom ErgoScript examples
- Optimized inference using 4-bit quantization
- Containerized training environment with experiment tracking

## Requirements

- Docker and Docker Compose
- CUDA-capable GPU (tested on NVIDIA RTX 3090)
- NVIDIA Container Toolkit installed
- 16GB+ RAM recommended

## Quick Start with Docker Compose

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SharkNet.git
cd SharkNet
```

2. Start the training environment:
```bash
docker compose up -d
```

This will start:
- The main training container
- MLflow server for experiment tracking (optional)

3. View training progress:
```bash
docker compose logs -f trainer
```

4. Access MLflow UI:
- Open http://localhost:5000 in your browser
- View experiment metrics, compare runs, and track model performance

## Configuration

You can configure the training process by editing the environment variables in `docker-compose.yml`:

### Model Configuration
- `MODEL_NAME`: Base model to use (default: "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 512)

### Training Configuration
- `TRAINING_MODE`: Training method to use ("standard" or "grpo", default: "standard")
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `BATCH_SIZE`: Batch size (default: 4)
- `NUM_EPOCHS`: Number of epochs (default: 3)
- `GRADIENT_ACCUMULATION_STEPS`: Steps for gradient accumulation (default: 4)

### Example with custom configuration:
```yaml
# In docker-compose.yml
services:
  trainer:
    environment:
      - MODEL_NAME=your-model-name
      - TRAINING_MODE=grpo
      - BATCH_SIZE=8
      - NUM_EPOCHS=5
```

## Output Structure

Each training run creates a unique output directory with the following structure:
```
output/
└── YYYYMMDD_HHMMSS_model-name/
    ├── test_results/
    │   └── test_results.json
    └── metrics.json
```

The `metrics.json` file contains:
- Script similarity scores
- Execution success rates
- Generation times
- Other relevant metrics

## Project Structure

```
SharkNet/
├── pipeline/           # Training pipeline components
├── prototype/         # Training data and prototypes
├── docker-compose.yml # Docker Compose configuration
├── Dockerfile        # Container definition
├── inference.py      # Inference script
└── README.md        # This file
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Add your ErgoScript examples to `prototype/RLdata_unix.txt`
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using the Llama model from Meta
- Optimized with Unsloth for faster training and inference
- Training data contributed by the Ergo community

# SharkNet Model Training and Evaluation

This repository contains a containerized solution for training and evaluating language models for script generation.

## Quick Start

1. Build the Docker image:
```bash
docker build -t sharknet .
```

2. Run training with default settings:
```bash
docker run --gpus all -v $(pwd)/output:/app/output -v $(pwd)/saved_models:/app/saved_models sharknet
```

## Configuration

You can configure the training process using environment variables:

### Model Configuration
- `MODEL_NAME`: Base model to use (default: "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 512)

### Training Configuration
- `TRAINING_MODE`: Training method to use ("standard" or "grpo", default: "standard")
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `BATCH_SIZE`: Batch size (default: 4)
- `NUM_EPOCHS`: Number of epochs (default: 3)
- `GRADIENT_ACCUMULATION_STEPS`: Steps for gradient accumulation (default: 4)

### Example with custom configuration:
```bash
docker run --gpus all \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/saved_models:/app/saved_models \
  -e MODEL_NAME="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit" \
  -e TRAINING_MODE="grpo" \
  -e BATCH_SIZE=8 \
  -e NUM_EPOCHS=5 \
  sharknet
```

## Output Structure

Each training run creates a unique output directory with the following structure:
```
output/
└── YYYYMMDD_HHMMSS_model-name/
    ├── test_results/
    │   └── test_results.json
    └── metrics.json
```

The `metrics.json` file contains:
- Script similarity scores
- Execution success rates
- Generation times
- Other relevant metrics

## Comparing Models

To compare different models, check the metrics.json files in their respective output directories. Each run creates a unique directory named with timestamp and model name for easy tracking.
