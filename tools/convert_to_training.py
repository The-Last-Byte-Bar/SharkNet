# tools/convert_to_training.py
import json
from pathlib import Path
from typing import Dict, List

def convert_to_training_format(input_dir: Path, output_file: Path) -> None:
    """
    Converts Q&A pairs to training format.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Path to save the converted dataset
    """
    training_data = []
    
    for json_file in input_dir.glob('**/*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for qa in data['conversations']:
                training_example = {
                    'instruction': qa['question'],
                    'input': '',
                    'output': qa['answer'],
                    'metadata': qa['metadata']
                }
                training_data.append(training_example)
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'data': training_data}, f, indent=2, ensure_ascii=False)