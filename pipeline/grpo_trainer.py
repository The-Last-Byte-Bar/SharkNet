import torch
from torch.optim import AdamW
from tqdm import tqdm
import os
from pipeline import config, model, data_loader
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL
from peft import LoraConfig
import re
from datasets import Dataset

# Patch FastRL for GRPO
PatchFastRL("GRPO", FastLanguageModel)

# System prompt for formatting responses
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def count_xml(text) -> float:
    """Count XML tags and calculate format score."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

# Reward functions
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for XML tag formatting."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Soft reward function for XML format checking."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Strict reward function for XML format checking."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function for answer correctness."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def train():
    """Main training loop."""
    # Initialize base model with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,  # Enable vLLM for fast inference
        max_lora_rank=32,  # Set max LoRA rank
        gpu_memory_utilization=0.6,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Create dataloaders
    train_loader, val_loader = data_loader.create_dataloaders()
    
    # Convert DataLoader to Dataset format required by GRPO
    train_data = []
    for batch in train_loader:
        train_data.extend([{
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            "answer": text
        } for text in batch["text"]])
    
    train_dataset = Dataset.from_dict({
        "prompt": [d["prompt"] for d in train_data],
        "answer": [d["answer"] for d in train_data]
    })

    # Setup GRPO training arguments
    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference
        learning_rate=5e-6,  # Match notebook learning rate
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=6,
        max_prompt_length=256,
        max_completion_length=200,
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=config.MODEL_SAVE_DIR
    )

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()
    
    # Save the final model
    save_path = os.path.join(config.MODEL_SAVE_DIR, "grpo_final")
    os.makedirs(save_path, exist_ok=True)
    model.save_lora(save_path)
    print(f"Saved GRPO trained model to {save_path}")
    
    print("GRPO Training completed!") 