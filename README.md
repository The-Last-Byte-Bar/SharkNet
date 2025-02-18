# SharkNet - ErgoScript AI Assistant

An AI-powered assistant for learning and working with ErgoScript, built using the Llama model and optimized with Unsloth.

## Features

- Interactive Q&A about ErgoScript
- Smart contract code generation
- Explanation of ErgoScript concepts
- Training on custom ErgoScript examples
- Optimized inference using 4-bit quantization

## Requirements

- Python 3.8+
- CUDA-capable GPU (tested on NVIDIA RTX 3090)
- 16GB+ RAM recommended

## Installation

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

## Project Structure

```
SharkNet/
├── pipeline/           # Training pipeline components
│   ├── __init__.py
│   ├── config.py      # Configuration settings
│   ├── data_loader.py # Data loading and processing
│   ├── model.py       # Model creation and setup
│   ├── trainer.py     # Training loop implementation
│   └── main.py        # Main training script
├── prototype/         # Training data and prototypes
│   └── RLdata_unix.txt # Training data file
├── inference.py      # Inference script for using the model
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## Training the Model

To train the model on the ErgoScript examples:

```bash
python -m pipeline.main
```

The training process will:
1. Load the base Llama model
2. Process the training data from `prototype/RLdata_unix.txt`
3. Fine-tune the model using the instruction format
4. Save checkpoints in the `saved_models` directory

## Using the Model

To use the trained model for generating ErgoScript code and explanations:

```bash
python inference.py
```

This will start an interactive session where you can:
- Ask questions about ErgoScript
- Get code examples for different smart contracts
- Receive explanations about ErgoScript concepts
- Type 'quit' to exit the session

Example usage:
```
What would you like to know about ErgoScript? How do I create a simple payment contract?

Generating response...

Response:
--------------------------------------------------
Here's how to create a simple payment contract in ErgoScript:

```scala
{
    // Define the recipient's public key
    val recipientPubKey = PK("recipient_pub_key")
    
    // Core logic: Only the recipient can spend the funds
    recipientPubKey
}
```

This contract:
1. Defines the recipient's public key
2. Ensures that only the recipient can spend the funds
3. Is a basic example of a payment contract in ErgoScript

You can deploy this contract by wrapping it in a proper Ergo transaction output.
--------------------------------------------------
```

## Configuration

Key configuration settings in `pipeline/config.py`:
- `MODEL_NAME`: Base model to use
- `MAX_SEQ_LENGTH`: Maximum sequence length for inputs
- `LEARNING_RATE`: Training learning rate
- `BATCH_SIZE`: Training batch size
- `NUM_EPOCHS`: Number of training epochs

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
