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
import logging
import time
from huggingface_hub import HfApi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patch FastRL for GRPO
PatchFastRL("GRPO", FastLanguageModel)

# System prompt for formatting responses (shortened for efficiency)
SYSTEM_PROMPT = """Use XML tags for responses:
<reasoning>...</reasoning>
<answer>...</answer>"""

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

def setup_http_session():
    """Setup HTTP session with retries and increased timeout."""
    session = requests.Session()
    retries = Retry(total=5,
                   backoff_factor=1,
                   status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def load_model_with_retry(model_name, **kwargs):
    """Load model with retry logic."""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading model attempt {attempt + 1}/{max_retries}")
            
            # Create custom session with increased timeout
            with requests.Session() as session:
                session.request = lambda method, url, *args, **kwargs: super(requests.Session, session).request(
                    method=method,
                    url=url,
                    *args,
                    **{**kwargs, 'timeout': 30}
                )
                
                # Set the session for HuggingFace API
                api = HfApi()
                api._session = session
                
                # Load the model
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    **kwargs
                )
                logger.info("Model loaded successfully!")
                return model, tokenizer
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All attempts to load model failed")
                raise

def train():
    """Main training loop."""
    # Initialize base model with unsloth using retry logic
    model, tokenizer = load_model_with_retry(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.5,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Create dataloaders
    train_loader, val_loader, _ = data_loader.create_dataloaders()
    
    # Convert DataLoader to Dataset format required by GRPO
    train_data = []
    max_length = 0
    for batch in train_loader:
        for text in batch["text"]:
            # Truncate text if needed
            tokens = tokenizer.encode(text)
            if len(tokens) > config.MAX_SEQ_LENGTH // 2:
                # Decode truncated tokens to ensure we don't cut in the middle of a word
                text = tokenizer.decode(tokens[:(config.MAX_SEQ_LENGTH // 2)])
                logger.info(f"Truncated text to {len(tokenizer.encode(text))} tokens")
            
            # Create prompt with system message
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
            
            # Verify total length
            total_tokens = len(tokenizer.encode(SYSTEM_PROMPT)) + len(tokenizer.encode(text))
            if total_tokens > config.MAX_SEQ_LENGTH:
                logger.warning(f"Skipping example with total length {total_tokens} > {config.MAX_SEQ_LENGTH}")
                continue
                
            max_length = max(max_length, total_tokens)
            logger.info(f"Total sequence length: {total_tokens}")
            
            train_data.append({
                "prompt": prompt,
                "answer": text
            })
    
    logger.info(f"Maximum sequence length in dataset: {max_length}")
    logger.info(f"Number of training examples: {len(train_data)}")
    
    if len(train_data) == 0:
        raise ValueError("No valid training examples found after filtering")
    
    train_dataset = Dataset.from_dict({
        "prompt": [d["prompt"] for d in train_data],
        "answer": [d["answer"] for d in train_data]
    })

    # Setup GRPO training arguments with adjusted lengths
    max_prompt_len = config.MAX_SEQ_LENGTH - 50  # Leave room for generation
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=max_prompt_len,
        max_completion_length=50,  # Shorter completions for safety
        max_steps=50,
        save_steps=50,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=config.MODEL_SAVE_DIR
    )

    # Initialize GRPO trainer with error handling
    try:
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
        
        # Save the final model with configuration
        save_path = os.path.join(config.MODEL_SAVE_DIR, "grpo_final")
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA weights and configuration
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # Save base model reference and configuration
        config_dict = {
            "base_model_name_or_path": config.MODEL_NAME,
            "model_type": "llama",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "max_seq_length": config.MAX_SEQ_LENGTH
        }
        
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
            
        logger.info(f"Saved GRPO trained model and configuration to {save_path}")
        
        logger.info("GRPO Training completed!")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise 