import os
import torch
import argparse
from pipeline import config
from pipeline.trainer import train
from pipeline.distributed_utils import DistributedTrainer
import warnings
warnings.filterwarnings("ignore")

def setup_environment():
    """Setup training environment and directories."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Set PyTorch settings for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--run_name', type=str, default='default_run',
                       help='Name for this training run')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override default batch size')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Override default number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override default learning rate')
    
    args = parser.parse_args()
    
    # Override config values if provided
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.num_epochs is not None:
        config.NUM_EPOCHS = args.num_epochs
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    
    # Setup environment
    setup_environment()
    
    # Create run directory
    run_dir = os.path.join(config.OUTPUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize distributed training
    with DistributedTrainer() as trainer:
        if trainer.is_main_process:
            print(f"Starting training run: {args.run_name}")
            print(f"Distributed training: {trainer.is_distributed}")
            if trainer.is_distributed:
                print(f"World size: {trainer.world_size}")
            print(f"Training configuration:")
            print(f"  Batch size: {config.BATCH_SIZE}")
            print(f"  Number of epochs: {config.NUM_EPOCHS}")
            print(f"  Learning rate: {config.LEARNING_RATE}")
        
        try:
            # Run training
            train(run_dir, trainer)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        finally:
            if trainer.is_main_process:
                print("Training completed or interrupted. Cleaning up...")

if __name__ == "__main__":
    main() 