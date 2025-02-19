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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SharkNet Training Interface")
    parser.add_argument('--mode', choices=['standard', 'grpo'], default='standard', help='Training mode to run')
    parser.add_argument('--scale', choices=['small', 'medium', 'large', 'full'], default='full', help='Training scale for GRPO mode')
    args = parser.parse_args()

    # Set up directories
    setup_directories()

    if args.mode == 'standard':
        print("Running standard training mode")
        standard_train_model()
    elif args.mode == 'grpo':
        from pipeline import config
        if args.scale == 'small':
            config.NUM_EPOCHS = 1
            config.BATCH_SIZE = 2
            print("Running a small GRPO training run: 1 epoch, batch size 2")
        elif args.scale == 'medium':
            config.NUM_EPOCHS = 2
            config.BATCH_SIZE = 4
            print("Running a medium GRPO training run: 2 epochs, batch size 4")
        elif args.scale == 'large':
            config.NUM_EPOCHS = 3
            config.BATCH_SIZE = 8
            print("Running a large GRPO training run: 3 epochs, batch size 8")
        else:
            print("Running full GRPO training run with default settings")
        grpo_train()

if __name__ == "__main__":
    main() 