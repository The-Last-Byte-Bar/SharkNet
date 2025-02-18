import os
import logging
import torch
from pipeline.trainer import train_model as standard_train_model
from pipeline.grpo_trainer import train as grpo_train
from pipeline.data_loader import create_dataloaders
from pipeline.model import create_model
from pipeline import config

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

def main(training_mode="standard"):
    """Main entry point for the training pipeline.
    
    Args:
        training_mode (str): Either "standard" or "grpo" to select training method
    """
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
    
    # Train model using selected method
    if training_mode == "grpo":
        logging.info("Starting GRPO training...")
        grpo_train()
    else:
        logging.info("Starting standard training...")
        # Create dataloaders and model for standard training
        train_loader, val_loader = create_dataloaders()
        model, tokenizer = create_model()
        standard_train_model(model, tokenizer, train_loader, val_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train the model using specified method')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'grpo'],
                      help='Training mode: standard or grpo (default: standard)')
    args = parser.parse_args()
    main(args.mode) 