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

## Requirements

- Python 3.8+
- CUDA-capable GPU (tested on NVIDIA RTX 3090)
- 16GB+ RAM recommended

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SharkNet.git
cd SharkNet
```

2. Create a conda environment:
```bash
conda create -n pool python=3.8
conda activate pool
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

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

## Project Structure

```
SharkNet/
├── pipeline/           # Training pipeline components
├── prototype/         # Training data and prototypes
├── webui/            # Web interface components
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

## Training

To train the model, you can use the training pipeline provided in the `pipeline/` directory. The training pipeline supports two modes: **standard** and **GRPO**.

### Standard Training

Run standard training with:

```bash
python pipeline/main.py --mode standard
```

This will run the default training procedure without any configuration overrides.

### GRPO Training

The GRPO training mode supports different training scales by using the `--scale` flag. Available scales are: `small`, `medium`, `large`, or `full` (default).

For example:

- To run a **small GRPO training run** (1 epoch, batch size 2):

  ```bash
  python pipeline/main.py --mode grpo --scale small
  ```

- To run a **medium GRPO training run** (2 epochs, batch size 4):

  ```bash
  python pipeline/main.py --mode grpo --scale medium
  ```

- To run a **large GRPO training run** (3 epochs, batch size 8):

  ```bash
  python pipeline/main.py --mode grpo --scale large
  ```

- To run a **full GRPO training run** with the default settings:

  ```bash
  python pipeline/main.py --mode grpo --scale full
  ```

These options help you quickly test different training scales during development.

### Requirements

- Python 3.8+
- CUDA-capable GPU (if using GPU acceleration)
- Conda environment named `pool` (as shown below)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/SharkNet.git
   cd SharkNet
   ```

2. Create and activate the conda environment:

   ```bash
   conda create -n pool python=3.8
   conda activate pool
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Now you're ready to train your models using either standard or GRPO training modes.
