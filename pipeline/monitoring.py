import os
import time
import json
import logging
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import GPUtil

@dataclass
class ResourceMetrics:
    """Track system resource usage."""
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float]
    gpu_memory_used: Optional[float]
    gpu_memory_total: Optional[float]
    
    @classmethod
    def collect(cls) -> 'ResourceMetrics':
        """Collect current resource metrics."""
        gpu_stats = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        
        return cls(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            gpu_utilization=gpu_stats.load * 100 if gpu_stats else None,
            gpu_memory_used=gpu_stats.memoryUsed if gpu_stats else None,
            gpu_memory_total=gpu_stats.memoryTotal if gpu_stats else None
        )

@dataclass
class TrainingMetrics:
    """Track training metrics."""
    step: int
    epoch: int
    train_loss: float
    learning_rate: float
    val_loss: Optional[float] = None
    train_perplexity: Optional[float] = None
    val_perplexity: Optional[float] = None
    grad_norm: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricsLogger:
    """Comprehensive training metrics logger."""
    
    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = os.path.join(log_dir, run_name)
        self.tensorboard = SummaryWriter(self.log_dir)
        self.metrics_history: List[Dict[str, Any]] = []
        self.resource_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.log_dir, 'training.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        metrics_dict = metrics.to_dict()
        self.metrics_history.append(metrics_dict)
        
        # Log to tensorboard
        for name, value in metrics_dict.items():
            if value is not None:
                self.tensorboard.add_scalar(f'training/{name}', value, metrics.step)
        
        # Log resource usage
        resource_metrics = ResourceMetrics.collect()
        resource_dict = asdict(resource_metrics)
        self.resource_history.append({
            'step': metrics.step,
            'epoch': metrics.epoch,
            'time': time.time() - self.start_time,
            **resource_dict
        })
        
        # Log to tensorboard
        for name, value in resource_dict.items():
            if value is not None:
                self.tensorboard.add_scalar(f'resources/{name}', value, metrics.step)
        
        # Log to console
        logging.info(
            f"Step {metrics.step} | Epoch {metrics.epoch} | "
            f"Train Loss: {metrics.train_loss:.4f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"GPU Util: {resource_metrics.gpu_utilization:.1f}% | "
            f"GPU Mem: {resource_metrics.gpu_memory_used:.0f}MB"
        )
    
    def plot_metrics(self):
        """Generate and save training metric plots."""
        metrics_df = self._create_metrics_dataframe()
        
        # Plot training curves
        self._plot_training_curves(metrics_df)
        
        # Plot resource usage
        self._plot_resource_usage()
        
        # Save metrics history
        self.save_metrics()
    
    def _create_metrics_dataframe(self):
        """Convert metrics history to structured format."""
        import pandas as pd
        return pd.DataFrame(self.metrics_history)
    
    def _plot_training_curves(self, df):
        """Plot training and validation curves."""
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(df['step'], df['train_loss'], label='Train Loss')
        if 'val_loss' in df.columns:
            plt.plot(df['step'], df['val_loss'], label='Val Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot learning rate
        plt.subplot(2, 2, 2)
        plt.plot(df['step'], df['learning_rate'])
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        # Plot perplexity if available
        if 'train_perplexity' in df.columns:
            plt.subplot(2, 2, 3)
            plt.plot(df['step'], df['train_perplexity'], label='Train')
            if 'val_perplexity' in df.columns:
                plt.plot(df['step'], df['val_perplexity'], label='Val')
            plt.xlabel('Step')
            plt.ylabel('Perplexity')
            plt.legend()
            plt.title('Perplexity')
        
        # Plot gradient norm if available
        if 'grad_norm' in df.columns:
            plt.subplot(2, 2, 4)
            plt.plot(df['step'], df['grad_norm'])
            plt.xlabel('Step')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norm')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()
    
    def _plot_resource_usage(self):
        """Plot system resource usage."""
        import pandas as pd
        df = pd.DataFrame(self.resource_history)
        
        plt.figure(figsize=(12, 8))
        
        # Plot CPU usage
        plt.subplot(2, 2, 1)
        plt.plot(df['time'] / 3600, df['cpu_percent'])  # Convert time to hours
        plt.xlabel('Time (hours)')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Utilization')
        
        # Plot memory usage
        plt.subplot(2, 2, 2)
        plt.plot(df['time'] / 3600, df['memory_percent'])
        plt.xlabel('Time (hours)')
        plt.ylabel('Memory Usage (%)')
        plt.title('Memory Utilization')
        
        # Plot GPU utilization if available
        if 'gpu_utilization' in df.columns:
            plt.subplot(2, 2, 3)
            plt.plot(df['time'] / 3600, df['gpu_utilization'])
            plt.xlabel('Time (hours)')
            plt.ylabel('GPU Usage (%)')
            plt.title('GPU Utilization')
        
        # Plot GPU memory if available
        if 'gpu_memory_used' in df.columns:
            plt.subplot(2, 2, 4)
            plt.plot(df['time'] / 3600, df['gpu_memory_used'])
            plt.xlabel('Time (hours)')
            plt.ylabel('GPU Memory (MB)')
            plt.title('GPU Memory Usage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'resource_usage.png'))
        plt.close()
    
    def save_metrics(self):
        """Save metrics history to file."""
        metrics_file = os.path.join(self.log_dir, 'metrics_history.json')
        with open(metrics_file, 'w') as f:
            json.dump({
                'training_metrics': self.metrics_history,
                'resource_metrics': self.resource_history
            }, f, indent=2)
    
    def close(self):
        """Cleanup and close logger."""
        self.plot_metrics()
        self.tensorboard.close()

class MetricsCallback:
    """Callback for collecting training metrics."""
    
    def __init__(self, logger: MetricsLogger):
        self.logger = logger
    
    def on_step_end(self, metrics: TrainingMetrics):
        """Called at the end of each training step."""
        self.logger.log_metrics(metrics)
    
    def on_training_end(self):
        """Called at the end of training."""
        self.logger.plot_metrics()
        self.logger.close() 