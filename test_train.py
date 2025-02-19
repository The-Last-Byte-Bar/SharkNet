from pipeline.grpo_trainer import train
import os
import torch

def main():
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    
    # Create output directories if they don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints/test_run", exist_ok=True)
    
    print("Starting test training run...")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        train()
        print("Test training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 