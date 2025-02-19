import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import os
import json
import numpy as np
from functools import partial
from pipeline import config
from torch.utils.data.distributed import DistributedSampler

class CachedDataset(Dataset):
    """Dataset with memory-efficient caching capabilities."""
    
    def __init__(self, texts: List[str], cache_dir: str = None):
        self.texts = texts
        self.cache_dir = cache_dir
        self.cache = {}
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"item_{idx}.pt")
            
            # Try to load from cache
            if idx in self.cache:
                return self.cache[idx]
            elif os.path.exists(cache_file):
                item = torch.load(cache_file)
                # Only cache if memory usage is below threshold
                if len(self.cache) < 1000:  # Adjust based on your memory constraints
                    self.cache[idx] = item
                return item
        
        # If not cached, process the item
        item = {"text": self.texts[idx]}
        
        # Cache the processed item if caching is enabled
        if self.cache_dir and len(self.cache) < 1000:
            self.cache[idx] = item
            torch.save(item, os.path.join(self.cache_dir, f"item_{idx}.pt"))
        
        return item

def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Custom collate function with dynamic padding."""
    texts = [item["text"] for item in batch]
    return {"text": texts}

def create_dataloaders(
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create optimized data loaders with proper memory management."""
    
    # Load and preprocess data
    with open(config.DATA_PATH, 'r') as f:
        data = f.readlines()
    
    # Clean data
    texts = [line.strip() for line in data if line.strip()]
    
    # Split data
    train_size = int(len(texts) * config.TRAIN_TEST_SPLIT)
    val_size = int((len(texts) - train_size) * 0.5)
    
    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size + val_size]
    test_texts = texts[train_size + val_size:]
    
    # Create datasets with caching
    cache_dir = os.path.join(config.OUTPUT_DIR, "dataset_cache")
    train_dataset = CachedDataset(train_texts, os.path.join(cache_dir, "train"))
    val_dataset = CachedDataset(val_texts, os.path.join(cache_dir, "val"))
    test_dataset = CachedDataset(test_texts, os.path.join(cache_dir, "test"))
    
    # Setup samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if distributed else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if distributed else None
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if distributed else None
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=config.DATALOADER_NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
        sampler=train_sampler,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
        sampler=val_sampler,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
        sampler=test_sampler,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader 