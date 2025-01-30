# tools/validate_ergo.py
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

def validate_qa_format(file_path: Path) -> bool:
    """
    Validates the format of a Q&A JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict) or 'conversations' not in data:
            print(f"Error: Invalid format in {file_path}")
            return False
            
        for qa in data['conversations']:
            required_fields = ['id', 'question', 'answer', 'metadata']
            if not all(field in qa for field in required_fields):
                print(f"Error: Missing required fields in {file_path}")
                return False
                
            metadata_fields = ['domain', 'difficulty', 'tags', 'source', 'language', 'contributor']
            if not all(field in qa['metadata'] for field in metadata_fields):
                print(f"Error: Missing metadata fields in {file_path}")
                return False
                
        return True
        
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False