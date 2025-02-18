import os
import json
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import shutil

class ModelManager:
    def __init__(self):
        self.models_dir = "saved_models"
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.current_model_name = None
        
    def list_models(self):
        """List all available models."""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for item in os.listdir(self.models_dir):
            if os.path.isdir(os.path.join(self.models_dir, item)):
                models.append(item)
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
        
        # Load new model
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

def generate_script(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate ErgoScript based on the prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    
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
            
            model_path = os.path.join("saved_models", model_name)
            info = {
                "Status": "Model selected",
                "Name": model_name,
                "Path": model_path,
                "Last Modified": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
            }
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