import torch
from torch.optim import AdamW
from tqdm import tqdm
import os
from pipeline import config, model, data_loader

def train_model(model, tokenizer, train_loader, val_loader):
    """Train the model using the provided data loaders."""
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # Prepare inputs
            inputs = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to(config.DEVICE)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Prepare inputs
                inputs = tokenizer(
                    batch["text"],
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_SEQ_LENGTH,
                    return_tensors="pt"
                ).to(config.DEVICE)
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        save_path = os.path.join(config.MODEL_SAVE_DIR, f"checkpoint-{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # Save optimizer state and metrics
        torch.save({
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, os.path.join(save_path, "training_state.pt"))
        
        print(f"Saved checkpoint to {save_path}")

def train():
    """Main training loop."""
    # Set up model and tokenizer
    llama_model, tokenizer = model.setup_model()
    
    # Create dataloaders
    train_loader, val_loader = data_loader.create_dataloaders()
    
    # Train the model
    train_model(llama_model, tokenizer, train_loader, val_loader)
    
    print("Training completed!") 