import os
import torch
import shutil
from typing import Dict, Optional
from dataclasses import dataclass
from pipeline import config

@dataclass
class EarlyStoppingState:
    """Track early stopping state."""
    best_loss: float = float('inf')
    counter: int = 0
    should_stop: bool = False
    best_model_path: Optional[str] = None

class EarlyStopping:
    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, threshold: float = config.EARLY_STOPPING_THRESHOLD):
        self.patience = patience
        self.threshold = threshold
        self.state = EarlyStoppingState()
    
    def __call__(self, val_loss: float, model, epoch: int, step: int) -> bool:
        if val_loss < self.state.best_loss * (1 - self.threshold):
            self._save_best_model(model, val_loss, epoch, step)
            self.state.counter = 0
        else:
            self.state.counter += 1
            if self.state.counter >= self.patience:
                self.state.should_stop = True
        
        return self.state.should_stop
    
    def _save_best_model(self, model, val_loss: float, epoch: int, step: int):
        self.state.best_loss = val_loss
        checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f"best_model_epoch{epoch}_step{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        
        if self.state.best_model_path and os.path.exists(self.state.best_model_path):
            shutil.rmtree(self.state.best_model_path)
        
        self.state.best_model_path = checkpoint_path

class CheckpointManager:
    def __init__(self, save_total_limit: int = config.SAVE_TOTAL_LIMIT):
        self.save_total_limit = save_total_limit
        self.checkpoints = []
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, step: int, 
                       loss: float, val_loss: Optional[float] = None):
        checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f"checkpoint_epoch{epoch}_step{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        model.save_pretrained(checkpoint_path)
        
        # Save optimizer and scheduler states
        torch.save({
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'val_loss': val_loss
        }, os.path.join(checkpoint_path, "training_state.pt"))
        
        self.checkpoints.append(checkpoint_path)
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        if len(self.checkpoints) > self.save_total_limit:
            checkpoint_to_remove = self.checkpoints.pop(0)
            if os.path.exists(checkpoint_to_remove):
                shutil.rmtree(checkpoint_to_remove)

class MetricsTracker:
    def __init__(self):
        self.current_loss = 0.0
        self.steps = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def update(self, loss: float):
        self.current_loss += loss
        self.steps += 1
    
    def get_average_loss(self) -> float:
        if self.steps == 0:
            return 0.0
        avg_loss = self.current_loss / self.steps
        self.history['train_loss'].append(avg_loss)
        return avg_loss
    
    def log_validation(self, val_loss: float):
        self.history['val_loss'].append(val_loss)
    
    def log_lr(self, lr: float):
        self.history['learning_rates'].append(lr)
    
    def reset(self):
        self.current_loss = 0.0
        self.steps = 0

def setup_mixed_precision():
    """Setup mixed precision training."""
    if not config.USE_MIXED_PRECISION:
        return None
    
    if torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None

def should_evaluate(step: int, epoch: int) -> bool:
    """Determine if evaluation should be performed based on current step/epoch."""
    if config.EVAL_STRATEGY == "steps":
        return step > 0 and step % config.EVAL_STEPS == 0
    return epoch > 0  # Evaluate at the end of each epoch

def should_save_checkpoint(step: int, epoch: int) -> bool:
    """Determine if checkpoint should be saved based on current step/epoch."""
    if config.SAVE_STRATEGY == "steps":
        return step > 0 and step % config.SAVE_STEPS == 0
    return epoch > 0  # Save at the end of each epoch

def clear_cuda_cache(step: int):
    """Clear CUDA cache if needed."""
    if torch.cuda.is_available() and step % config.EMPTY_CACHE_FREQUENCY == 0:
        torch.cuda.empty_cache() 