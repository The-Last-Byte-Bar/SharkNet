import os
import logging
import torch
from pipeline.trainer import train_model
from pipeline.data_loader import create_dataloaders
from pipeline.model import create_model
from pipeline import config

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

def main():
    """Main entry point for the training pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join('output', 'training.log'))
        ]
    )
    
    # Create necessary directories
    setup_directories()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders()
    
    # Create model and tokenizer
    model, tokenizer = create_model()
    
    # Train model
    train_model(model, tokenizer, train_loader, val_loader)

if __name__ == "__main__":
    main() 