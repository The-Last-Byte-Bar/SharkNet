import json
from typing import Dict, List, Tuple
import random
from torch.utils.data import Dataset, DataLoader, random_split
import re
from pipeline import config
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Log sequence length
        logger.info(f"Sequence length: {len(formatted_text.split())}")
        
        return {
            "text": formatted_text,
            "instruction": instruction,
            "response": response,
            "metadata": conv.get('metadata', {})
        }

class ConversationDataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations
        self._validate_and_truncate()
    
    def _validate_and_truncate(self):
        """Validate and truncate conversations to fit within max length."""
        max_words = config.MAX_SEQ_LENGTH // 2  # Rough estimate of tokens to words
        truncated = 0
        
        for i, conv in enumerate(self.conversations):
            words = len(conv.split())
            if words > max_words:
                # Truncate to max length while preserving complete sentences
                sentences = conv.split('.')
                truncated_conv = []
                word_count = 0
                
                for sent in sentences:
                    sent_words = len(sent.split())
                    if word_count + sent_words <= max_words:
                        truncated_conv.append(sent)
                        word_count += sent_words
                    else:
                        break
                
                self.conversations[i] = '. '.join(truncated_conv) + '.'
                truncated += 1
        
        if truncated > 0:
            logger.info(f"Truncated {truncated} conversations to fit within max length")
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        return {"text": self.conversations[idx]}

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

def load_conversations(file_path="prototype/RLdata_unix.txt"):
    """Load conversations from file."""
    conversations = []
    with open(file_path, 'r') as f:
        conversations = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(conversations)} conversations from {file_path}")
    return conversations

def create_dataloaders():
    """Create train, validation, and test dataloaders."""
    # Load data
    conversations = load_conversations()
    
    # Create dataset
    dataset = ConversationDataset(conversations)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Split data into {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test conversations")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process test data one at a time
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader 