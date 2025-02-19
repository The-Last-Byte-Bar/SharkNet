import os
import json
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import shutil
from peft import PeftModel, PeftConfig
from unsloth import FastLanguageModel

class ModelManager:
    def __init__(self):
        self.models_dir = "checkpoints"
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.current_model_name = None
        
    def list_models(self):
        """List all available models recursively."""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for root, dirs, files in os.walk(self.models_dir):
            # Check if this directory contains model files
            if any(f in files for f in ["config.json", "adapter_config.json", "pytorch_model.bin", "adapter_model.safetensors"]):
                # Get relative path from models_dir
                rel_path = os.path.relpath(root, self.models_dir)
                if rel_path != ".":  # Skip the root directory itself
                    models.append(rel_path)
        
        return sorted(models, reverse=True)
    
    def load_model(self, model_name):
        """Load a model and its tokenizer."""
        if model_name == self.current_model_name and self.loaded_model is not None:
            return self.loaded_model, self.loaded_tokenizer
            
        model_path = os.path.join(self.models_dir, model_name)
        
        # Free up memory if another model was loaded
        if self.loaded_model is not None:
            del self.loaded_model
            del self.loaded_tokenizer
            torch.cuda.empty_cache()
        
        try:
            # First try to load as a LoRA model
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if config.get("peft_type") == "LORA":
                    # Load base model
                    base_model_name = config.get("base_model_name_or_path")
                    base_model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=base_model_name,
                        max_seq_length=config.get("max_seq_length", 2048),
                        load_in_4bit=True,
                        device_map="auto"
                    )
                    
                    # Load LoRA adapter
                    model = PeftModel.from_pretrained(
                        base_model,
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    
                    # Prepare model for inference
                    model = FastLanguageModel.for_inference(model)
                else:
                    # Load as regular model
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        max_memory={0: os.getenv("MAX_MEMORY", "16GB")}
                    )
            else:
                # Load as regular model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    max_memory={0: os.getenv("MAX_MEMORY", "16GB")}
                )
            
            self.loaded_model = model
            self.loaded_tokenizer = tokenizer
            self.current_model_name = model_name
            
            return model, tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

def generate_script(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate ErgoScript based on the prompt."""
    # Add system prompt for LoRA models
    if isinstance(model, PeftModel):
        formatted_prompt = [
            {"role": "system", "content": "Use XML tags for responses:\n<reasoning>...</reasoning>\n<answer>...</answer>"},
            {"role": "user", "content": prompt}
        ]
        # Convert to string format
        prompt = "\n\n".join([msg["content"] for msg in formatted_prompt])
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set generation parameters based on model type
    generation_kwargs = {
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1
    }
    
    # Add specific parameters for Unsloth models
    if hasattr(model, "unsloth_model"):
        generation_kwargs.update({
            "use_cache": True,
            "max_new_tokens": max_length - inputs["input_ids"].shape[1],  # Adjust for input length
            "early_stopping": True
        })
    
    outputs = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def save_interaction(prompt, response, model_name, save_dir="output/interactions"):
    """Save the interaction for future reference."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    interaction = {
        "timestamp": timestamp,
        "model": model_name,
        "prompt": prompt,
        "response": response,
        "metadata": {
            "model_name": model_name,
            "timestamp": timestamp
        }
    }
    
    filename = os.path.join(save_dir, f"interaction_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(interaction, f, indent=2)
    
    return filename

def create_ui():
    """Create the Gradio interface."""
    model_manager = ModelManager()
    available_models = model_manager.list_models()
    default_model = available_models[0] if available_models else None
    
    with gr.Blocks(title="SharkNet - ErgoScript Assistant") as interface:
        gr.Markdown("""
        # SharkNet - ErgoScript Assistant
        
        Generate and explain ErgoScript code using our trained AI models.
        Select a model and start generating smart contracts!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="Select Model",
                    info="Choose a trained model to use"
                )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Higher values make output more creative but less focused"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="Top P",
                        info="Nucleus sampling threshold"
                    )
                
                max_length = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Max Length",
                    info="Maximum length of generated response"
                )
            
            with gr.Column(scale=1):
                model_info = gr.JSON(
                    label="Model Information",
                    value={"Status": "No model loaded"}
                )
                
                refresh_btn = gr.Button("🔄 Refresh Models")
        
        with gr.Row():
            with gr.Column():
                input_prompt = gr.Textbox(
                    lines=4,
                    label="Your Question/Prompt",
                    placeholder="Ask about ErgoScript or describe the smart contract you want to create..."
                )
                
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                output_text = gr.Textbox(
                    lines=15,
                    label="Generated Response",
                    show_copy_button=True
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    save_path = gr.Textbox(
                        label="Last Interaction Save Path",
                        interactive=False
                    )
        
        # Example prompts
        with gr.Row():
            gr.Markdown("""
            ### Example Prompts:
            Click on any example to try it out.
            """)
            
        examples = gr.Examples(
            examples=[
                ["Create a simple token lock contract that requires a specific deadline and recipient public key"],
                ["Explain how to implement a time-locked vault contract with multiple stages of unlocking"],
                ["Generate a smart contract for an NFT marketplace with royalty support"],
                ["How do I implement atomic swap functionality in ErgoScript?"],
                ["Create a DAO voting contract with minimum stake requirements"],
            ],
            inputs=input_prompt
        )
        
        def refresh_models():
            models = model_manager.list_models()
            return gr.Dropdown(choices=models), {"Status": "Models refreshed", "Available Models": len(models)}
        
        def update_model_info(model_name):
            if not model_name:
                return {"Status": "No model selected"}
            
            model_path = os.path.join("checkpoints", model_name)
            metrics_path = os.path.join("artifacts/latest", f"{model_name}_metrics.json")
            config_path = os.path.join(model_path, "config.json")
            
            info = {
                "Status": "Model selected",
                "Name": model_name,
                "Path": model_path,
                "Last Modified": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add model configuration if available
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if config.get("peft_type") == "LORA":
                        info["Model Type"] = "LoRA-adapted LLM"
                        info["Base Model"] = config.get("base_model_name_or_path", "Unknown")
                        info["Max Sequence Length"] = config.get("max_seq_length", "Unknown")
                    else:
                        info["Model Type"] = config.get("model_type", "Unknown")
            
            # Add metrics if available
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    info["Metrics"] = {
                        "Format Score": f"{metrics['format_metrics']['format_score']:.2%}",
                        "Average Generation Time": f"{metrics['performance_metrics']['average_generation_time']:.2f}s",
                        "Memory Usage": f"{metrics['performance_metrics']['average_memory_usage_mb']:.0f}MB"
                    }
                    if metrics['performance_metrics'].get('average_gpu_memory_mb'):
                        info["Metrics"]["GPU Memory"] = f"{metrics['performance_metrics']['average_gpu_memory_mb']:.0f}MB"
            
            return info
        
        def on_clear():
            return "", None
        
        def on_submit(model_name, prompt, temp, p, max_len):
            try:
                if not model_name:
                    return "Error: Please select a model first", None
                
                model, tokenizer = model_manager.load_model(model_name)
                
                response = generate_script(
                    model,
                    tokenizer,
                    prompt,
                    max_length=max_len,
                    temperature=temp,
                    top_p=p
                )
                
                # Save the interaction
                save_path = save_interaction(prompt, response, model_name)
                
                return response, save_path
            except Exception as e:
                return f"Error: {str(e)}", None
        
        # Event handlers
        refresh_btn.click(
            fn=refresh_models,
            outputs=[model_dropdown, model_info]
        )
        
        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
        
        clear_btn.click(
            fn=on_clear,
            outputs=[output_text, save_path]
        )
        
        generate_btn.click(
            fn=on_submit,
            inputs=[
                model_dropdown,
                input_prompt,
                temperature,
                top_p,
                max_length
            ],
            outputs=[output_text, save_path]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_ui()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 