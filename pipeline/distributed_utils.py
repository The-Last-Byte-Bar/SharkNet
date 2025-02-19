import os
import torch
import torch.distributed as dist
from typing import Tuple, Optional

def setup_distributed() -> Tuple[bool, int, int]:
    """
    Setup distributed training.
    Returns:
        Tuple of (is_distributed, world_size, rank)
    """
    if not torch.cuda.is_available():
        return False, 1, 0
    
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        return True, world_size, rank
    
    return False, 1, 0

def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce tensor across all devices during distributed training.
    """
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

class DistributedTrainer:
    """Helper class for managing distributed training state."""
    
    def __init__(self):
        self.is_distributed, self.world_size, self.rank = setup_distributed()
        self.is_main_process = self.rank == 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare model for distributed training."""
        if not self.is_distributed:
            return model
        
        # Move model to correct device
        model = model.to(self.rank)
        
        # Wrap with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=True
        )
        
        return model
    
    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce loss across all processes."""
        return reduce_tensor(loss, self.world_size) if self.is_distributed else loss
    
    def should_save_model(self) -> bool:
        """Whether this process should save model checkpoints."""
        return not self.is_distributed or self.is_main_process
    
    def synchronize(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()

def all_gather_object(obj: object) -> list:
    """
    Gather objects from all processes.
    Useful for collecting metrics and results.
    """
    if not dist.is_initialized():
        return [obj]
    
    world_size = dist.get_world_size()
    gathered_objects = [None] * world_size
    dist.all_gather_object(gathered_objects, obj)
    return gathered_objects 