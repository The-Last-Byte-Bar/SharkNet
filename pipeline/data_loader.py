import json
from typing import Dict, List, Tuple
import random
from torch.utils.data import Dataset, DataLoader
import re
from pipeline import config

class ErgoDataset(Dataset):
    def __init__(self, conversations: List[Dict]):
        self.conversations = conversations
        
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict:
        conv = self.conversations[idx]
        
        # Format the conversation as instruction and response
        instruction = conv['question']
        response = conv['answer']
        
        # Format for model input
        formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        return {
            "text": formatted_text,
            "instruction": instruction,
            "response": response,
            "metadata": conv.get('metadata', {})
        }

def clean_json_string(content: str) -> str:
    """Clean and prepare JSON string for parsing."""
    # Remove any BOM characters
    content = content.strip('\ufeff')
    
    # Handle escaped characters in code blocks
    content = re.sub(r'```scala\n(.*?)```', lambda m: m.group(0).replace('\n', '\\n'), content, flags=re.DOTALL)
    
    # Handle newlines outside of code blocks
    content = re.sub(r'\n(?!```)', '\\n', content)
    
    return content

def load_data(data_path: str) -> List[Dict]:
    """Load conversations from the RLdata file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean the JSON content
    content = clean_json_string(content)
    
    try:
        data = json.loads(content)
        conversations = data['conversations']
        print(f"Loaded {len(conversations)} conversations from {data_path}")
        return conversations
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")
        raise

def split_data(conversations: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split conversations into training and validation sets."""
    if len(conversations) <= 1:
        # If we have 1 or fewer conversations, use it for both training and validation
        return conversations, conversations
    
    random.shuffle(conversations)
    split_idx = max(1, int(len(conversations) * train_ratio))  # Ensure at least 1 conversation in training
    train_convs = conversations[:split_idx]
    val_convs = conversations[split_idx:]
    
    # If validation set is empty, use the last training conversation
    if not val_convs:
        val_convs = [train_convs[-1]]
    
    print(f"Split data into {len(train_convs)} training and {len(val_convs)} validation conversations")
    return train_convs, val_convs

def create_dataloaders(batch_size: int = config.BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    # Load and split data
    conversations = load_data(config.DATA_PATH)
    train_convs, val_convs = split_data(conversations, config.TRAIN_TEST_SPLIT)
    
    # Create datasets
    train_dataset = ErgoDataset(train_convs)
    val_dataset = ErgoDataset(val_convs)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader 