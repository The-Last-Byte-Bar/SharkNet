import torch
from unsloth import FastLanguageModel
from pipeline import config
import os
import json
from datetime import datetime
from pathlib import Path

def find_latest_model():
    """Find the latest trained model in the model save directory."""
    model_dir = Path(config.MODEL_SAVE_DIR)
    
    # Look for different types of checkpoints
    checkpoints = []
    
    # Standard checkpoints
    std_checkpoints = list(model_dir.glob("checkpoint-*"))
    if std_checkpoints:
        latest_std = max(std_checkpoints, key=lambda x: int(x.name.split("-")[1]))
        checkpoints.append(("standard", latest_std))
    
    # GRPO final model
    grpo_final = model_dir / "grpo_final"
    if grpo_final.exists():
        checkpoints.append(("grpo", grpo_final))
    
    if not checkpoints:
        raise ValueError(f"No trained models found in {model_dir}")
    
    # Return the most recently modified checkpoint
    latest = max(checkpoints, key=lambda x: x[1].stat().st_mtime)
    return latest[0], str(latest[1])

def load_model(checkpoint_path, model_type="standard"):
    """Load the trained model and tokenizer."""
    print(f"Loading {model_type} model from {checkpoint_path}")
    
    # Initialize base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,  # Use original model name
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    # Load LoRA weights if it's a GRPO model
    if model_type == "grpo":
        model.load_adapter(checkpoint_path)
    
    # Prepare model for inference
    model = FastLanguageModel.for_inference(model)
    model = model.to(config.DEVICE)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    """Generate a response for the given question."""
    # Format input with system prompt from GRPO training
    system_prompt = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
    prompt = f"{system_prompt}\n\nUser: {question}"
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(config.DEVICE)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and clean up response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def extract_ergoscript(response):
    """Extract ErgoScript code from the response."""
    try:
        answer_section = response.split("<answer>")[1].split("</answer>")[0].strip()
        # Look for code blocks
        if "```ergoscript" in answer_section:
            code = answer_section.split("```ergoscript")[1].split("```")[0].strip()
        elif "```" in answer_section:
            code = answer_section.split("```")[1].split("```")[0].strip()
        else:
            code = answer_section
        return code
    except:
        return response

def save_response(question, response, output_dir="test_outputs"):
    """Save the response to a .es file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename based on the first few words of the question
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    question_slug = "_".join(question.split()[:5]).lower()
    question_slug = "".join(c if c.isalnum() or c == "_" else "" for c in question_slug)
    filename = f"{timestamp}_{question_slug}.es"
    
    # Extract ErgoScript code
    code = extract_ergoscript(response)
    
    # Save to file
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(f"// Question: {question}\n")
        f.write(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write(code)
    
    print(f"\nSaved ErgoScript to: {filepath}")
    return filepath

def load_test_cases(test_file="test_cases.json"):
    """Load test cases from a JSON file."""
    if not os.path.exists(test_file):
        default_cases = {
            "test_cases": [
                {
                    "question": "Write a simple ErgoScript contract that implements a basic token with minting and burning capabilities.",
                    "description": "Basic token with mint/burn",
                    "category": "tokens",
                    "complexity": "basic"
                },
                {
                    "question": "Create an ErgoScript smart contract for a time-locked vault.",
                    "description": "Time-locked vault",
                    "category": "vaults",
                    "complexity": "intermediate"
                },
                {
                    "question": "Write an ErgoScript contract for a 2-of-3 multi-signature wallet.",
                    "description": "Multi-signature wallet",
                    "category": "wallets",
                    "complexity": "advanced"
                }
            ]
        }
        with open(test_file, "w") as f:
            json.dump(default_cases, f, indent=2)
    
    with open(test_file, "r") as f:
        data = json.load(f)
    return data["test_cases"]

def main():
    # Find and load the latest model
    model_type, checkpoint_path = find_latest_model()
    model, tokenizer = load_model(checkpoint_path, model_type)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("test_outputs", f"run_{timestamp}")
    
    print("\nErgoScript Assistant Ready!")
    print("-" * 50)
    
    # Load test cases
    test_cases = load_test_cases()
    
    # Interactive mode or test mode
    mode = input("Choose mode (1: Interactive, 2: Run test cases): ").strip()
    
    if mode == "2":
        print("\nAvailable test cases:")
        for i, case in enumerate(test_cases, 1):
            print(f"{i}. {case['description']} ({case['category']}, {case['complexity']})")
        
        # Allow selecting specific test cases
        selection = input("\nEnter test case numbers to run (comma-separated) or 'all': ").strip()
        
        if selection.lower() == 'all':
            cases_to_run = test_cases
        else:
            try:
                indices = [int(i.strip()) - 1 for i in selection.split(',')]
                cases_to_run = [test_cases[i] for i in indices if 0 <= i < len(test_cases)]
            except:
                print("Invalid selection. Running all test cases.")
                cases_to_run = test_cases
        
        print("\nRunning selected test cases...")
        for i, case in enumerate(cases_to_run, 1):
            print(f"\nTest Case {i}: {case['description']}")
            print(f"Category: {case['category']}")
            print(f"Complexity: {case['complexity']}")
            print("-" * 50)
            response = generate_response(model, tokenizer, case['question'])
            save_response(case['question'], response, output_dir)
    else:
        while True:
            question = input("\nWhat would you like to know about ErgoScript? (Type 'quit' to exit) ").strip()
            
            if question.lower() in ['quit', 'exit']:
                break
            
            if not question:
                continue
            
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, question)
            print("\nResponse:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Save the response
            save_response(question, response, output_dir)

if __name__ == "__main__":
    main() 