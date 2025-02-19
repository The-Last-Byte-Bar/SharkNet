import torch
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from pipeline import config, model, data_loader_optimized
from pipeline.training_utils import (
    EarlyStopping, CheckpointManager, MetricsTracker,
    setup_mixed_precision, should_evaluate, should_save_checkpoint, clear_cuda_cache
)
from pipeline.monitoring import MetricsLogger, TrainingMetrics, MetricsCallback

def evaluate_model(model, tokenizer, test_loader, output_dir):
    """Evaluate model and save test results."""
    model.eval()
    results = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Record start time for performance metrics
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            # Prepare inputs
            inputs = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Generate response for samples
            if i < 5:  # Only generate for first 5 samples to save time
                generated = model.generate(
                    **inputs,
                    max_length=config.MAX_SEQ_LENGTH,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                end_time.record()
                torch.cuda.synchronize()
                
                # Decode output
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                # Save result
                results.append({
                    "id": i,
                    "input": batch["text"][0],
                    "generated_output": generated_text,
                    "performance": {
                        "execution_time": start_time.elapsed_time(end_time) / 1000,  # Convert to seconds
                        "memory_usage_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
                    }
                })
    
    # Calculate average loss
    avg_loss = total_loss / num_batches
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "metrics": {
                "avg_loss": avg_loss
            }
        }, f, indent=2)
    
    return avg_loss, results

def train_model(model, tokenizer, train_loader, val_loader, test_loader, run_dir, trainer):
    """Train the model using the provided data loaders."""
    # Setup training components
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    num_training_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = config.get_cosine_schedule_with_warmup(optimizer, num_training_steps)
    
    # Setup training utilities
    early_stopping = EarlyStopping()
    checkpoint_manager = CheckpointManager()
    metrics_tracker = MetricsTracker()
    scaler = setup_mixed_precision()
    
    # Setup monitoring
    metrics_logger = MetricsLogger(config.OUTPUT_DIR, os.path.basename(run_dir))
    metrics_callback = MetricsCallback(metrics_logger)
    
    # Enable gradient checkpointing if configured
    if config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    global_step = 0
    best_val_loss = float('inf')
    
    try:
        for epoch in range(config.NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
            
            # Training phase
            model.train()
            metrics_tracker.reset()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                # Clear CUDA cache periodically
                clear_cuda_cache(global_step)
                
                # Prepare inputs
                inputs = tokenizer(
                    batch["text"],
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_SEQ_LENGTH,
                    return_tensors="pt"
                ).to(config.DEVICE)
                
                # Mixed precision training
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                        # Calculate gradient norm for monitoring
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                        
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
                    loss.backward()
                    
                    if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                
                # Update metrics
                metrics_tracker.update(loss.item() * config.GRADIENT_ACCUMULATION_STEPS)
                
                # Log metrics
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    metrics = TrainingMetrics(
                        step=global_step,
                        epoch=epoch,
                        train_loss=metrics_tracker.get_average_loss(),
                        learning_rate=scheduler.get_last_lr()[0],
                        grad_norm=grad_norm.item() if grad_norm is not None else None
                    )
                    metrics_callback.on_step_end(metrics)
                
                global_step += 1
                
                # Evaluation phase
                if should_evaluate(global_step, epoch):
                    model.eval()
                    val_loss, _ = evaluate_model(model, tokenizer, val_loader, os.path.join(run_dir, f"eval_step_{global_step}"))
                    metrics_tracker.log_validation(val_loss)
                    
                    # Update metrics with validation results
                    metrics = TrainingMetrics(
                        step=global_step,
                        epoch=epoch,
                        train_loss=metrics_tracker.get_average_loss(),
                        learning_rate=scheduler.get_last_lr()[0],
                        val_loss=val_loss,
                        grad_norm=grad_norm.item() if grad_norm is not None else None
                    )
                    metrics_callback.on_step_end(metrics)
                    
                    # Early stopping check
                    if early_stopping(val_loss, model, epoch, global_step):
                        print("Early stopping triggered!")
                        break
                    
                    # Save checkpoint if it's the best model so far
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint_manager.save_checkpoint(
                            model, optimizer, scheduler, epoch, global_step,
                            metrics_tracker.get_average_loss(), val_loss
                        )
                    
                    model.train()
                
                # Regular checkpoint saving
                if should_save_checkpoint(global_step, epoch):
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, scheduler, epoch, global_step,
                        metrics_tracker.get_average_loss()
                    )
            
            # End of epoch logging
            avg_loss = metrics_tracker.get_average_loss()
            print(f"Epoch {epoch + 1} - Average training loss: {avg_loss:.4f}")
            
            if early_stopping.state.should_stop:
                print("Training stopped early!")
                break
        
        # Final evaluation
        print("Running final evaluation...")
        test_loss, test_results = evaluate_model(model, tokenizer, test_loader, os.path.join(run_dir, "final_eval"))
        print(f"Final test loss: {test_loss:.4f}")
        
        # Final metrics update
        metrics = TrainingMetrics(
            step=global_step,
            epoch=config.NUM_EPOCHS,
            train_loss=metrics_tracker.get_average_loss(),
            learning_rate=scheduler.get_last_lr()[0],
            val_loss=test_loss
        )
        metrics_callback.on_step_end(metrics)
        
    finally:
        # Ensure proper cleanup
        metrics_callback.on_training_end()
    
    return test_results

def train(run_dir, trainer):
    """Main training loop."""
    # Set up model and tokenizer
    llama_model, tokenizer = model.create_model()
    
    # Prepare model for distributed training if needed
    llama_model = trainer.prepare_model(llama_model)
    
    # Create dataloaders with distributed support
    train_loader, val_loader, test_loader = data_loader_optimized.create_dataloaders(
        distributed=trainer.is_distributed,
        world_size=trainer.world_size,
        rank=trainer.rank
    )
    
    # Train the model
    test_results = train_model(llama_model, tokenizer, train_loader, val_loader, test_loader, run_dir, trainer)
    
    if trainer.is_main_process:
        print("Training completed!") 