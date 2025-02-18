import torch
from unsloth import FastLanguageModel
from pipeline import config
import os

def load_model(checkpoint_path):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load the model and tokenizer directly from the checkpoint path
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,  # Use checkpoint path instead of original model name
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    # Prepare model for inference
    model = FastLanguageModel.for_inference(model)
    model = model.to(config.DEVICE)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    """Generate a response for the given question."""
    # Format input like during training
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    
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

def main():
    # Load the latest checkpoint
    checkpoints = [d for d in os.listdir(config.MODEL_SAVE_DIR) if d.startswith("checkpoint-")]
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, latest_checkpoint)
    
    # Load model and tokenizer
    model, tokenizer = load_model(checkpoint_path)
    
    print("\nErgoScript Assistant Ready! (Type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        question = input("\nWhat would you like to know about ErgoScript? ").strip()
        
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

if __name__ == "__main__":
    main() 