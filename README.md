# SharkNet - ErgoScript AI Assistant

An AI-powered assistant for learning and working with ErgoScript, built using the Llama model and optimized with Unsloth.

## Features

- Interactive Q&A about ErgoScript
- Smart contract code generation
- Explanation of ErgoScript concepts
- Training on custom ErgoScript examples
- Optimized inference using 4-bit quantization
- GRPO (Guided Reward Policy Optimization) training by default
- User-friendly web interface for model interaction
- MLflow integration for experiment tracking
- Containerized environment for training and inference

## Requirements

- Python 3.8+
- CUDA-capable GPU (tested on NVIDIA RTX 3090)
- 16GB+ RAM recommended

For Docker deployment:
- Docker and Docker Compose
- NVIDIA Container Toolkit

## Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SharkNet.git
cd SharkNet
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Using Docker

1. Start the complete environment:
```bash
docker compose up -d
```

This will start:
- Training service (using GRPO mode)
- Inference service with web UI
- MLflow for experiment tracking

2. Access the interfaces:
- Web UI: http://localhost:7860
- MLflow: http://localhost:5000

### Using Standalone Inference

For quick testing or development, you can use the standalone inference script:

```bash
python inference_standalone.py
```

This will start either:
- A web interface (if gradio is installed)
- A CLI interface (if gradio is not installed)

## Configuration

### Model Configuration
- `MODEL_NAME`: Base model to use (default: "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 512)

### Training Configuration
- `TRAINING_MODE`: Training method ("grpo" or "standard", default: "grpo")
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `BATCH_SIZE`: Batch size (default: 4)
- `NUM_EPOCHS`: Number of epochs (default: 3)
- `GRADIENT_ACCUMULATION_STEPS`: Steps for gradient accumulation (default: 4)

### Docker Environment Variables
You can configure the services by editing `docker-compose.yml` or using environment variables:

```yaml
services:
  trainer:
    environment:
      - MODEL_NAME=your-model-name
      - TRAINING_MODE=grpo
      - BATCH_SIZE=8
      - NUM_EPOCHS=5
```

## Project Structure

```
SharkNet/
├── pipeline/           # Training pipeline components
├── prototype/         # Training data and prototypes
├── webui/            # Web interface components
├── docker-compose.yml # Docker configuration
├── Dockerfile        # Training container definition
├── Dockerfile.inference # Inference container definition
├── inference_standalone.py # Standalone inference script
└── README.md        # This file
```

## Output Structure

Each training run and inference session creates a unique output directory:
```
output/
└── YYYYMMDD_HHMMSS_model-name/
    ├── test_results/
    │   └── test_results.json
    ├── metrics.json
    └── interactions/        # Web UI interaction history
        └── interaction_YYYYMMDD_HHMMSS.json
```

## Using the Web Interface

1. Select a trained model from the dropdown
2. Adjust generation parameters:
   - Temperature: Controls creativity vs. consistency
   - Top P: Nucleus sampling threshold
   - Max Length: Maximum response length
3. Enter your prompt or select an example
4. Click "Generate" to get a response

Features:
- Real-time model switching
- Parameter adjustment
- Example prompts
- Response history
- Copy-to-clipboard functionality

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
