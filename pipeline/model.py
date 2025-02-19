from typing import Tuple, Any
import torch
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from unsloth import FastLanguageModel
from pipeline import config

def create_model() -> Tuple[torch.nn.Module, Any]:
    """Create and configure the model."""
    # Initialize the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    # Move model to device
    model = model.to(config.DEVICE)
    
    return model, tokenizer

def prepare_optimizer_and_scheduler(model, num_training_steps: int):
    """Prepare optimizer and learning rate scheduler."""
    # Prepare optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Prepare scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler

def save_model(model, tokenizer, epoch: int):
    """Save model and tokenizer checkpoints."""
    import os
    
    # Create save directory if it doesn't exist
    save_path = os.path.join(config.MODEL_SAVE_DIR, f"checkpoint-{epoch}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path) 