import os
import json
import shutil
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import socket

class ModelManager:
    """Model manager for actual inference."""
    def __init__(self):
        self.models_dir = "saved_models"
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.current_model_name = None
        self.base_model_name = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
        
    def ensure_tokenizer_files(self, model_path):
        """Ensure tokenizer files exist by copying from base model if needed."""
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
        missing_files = [f for f in tokenizer_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing_files:
            print(f"Missing tokenizer files: {missing_files}")
            print(f"Downloading tokenizer from base model: {self.base_model_name}")
            try:
                temp_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                temp_tokenizer.save_pretrained(model_path)
                print("Tokenizer files copied successfully!")
            except Exception as e:
                print(f"Error copying tokenizer files: {str(e)}")
                raise
    
    def fix_model_config(self, model_path):
        """Fix model configuration file."""
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Add required fields if missing
                if 'model_type' not in config:
                    config['model_type'] = 'llama'
                if 'architectures' not in config:
                    config['architectures'] = ['LlamaForCausalLM']
                
                # Save updated config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print("Model config updated successfully!")
            except Exception as e:
                print(f"Error fixing config: {str(e)}")
                raise
    
    def list_models(self):
        """List available trained models."""
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
            
        print(f"Loading model from {model_path}...")
            
        try:
            # Ensure tokenizer files exist
            self.ensure_tokenizer_files(model_path)
            
            # Fix model configuration
            self.fix_model_config(model_path)
            
            # Try loading with fixed configuration
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                max_memory={0: os.getenv("MAX_MEMORY", "16GB")}
            )

            # Patch the model: set max_seq_length if not present
            if not hasattr(model, 'max_seq_length'):
                model.max_seq_length = getattr(model.config, 'max_position_embeddings', 512)
                print(f"ModelManager: Patched model.max_seq_length to {model.max_seq_length}")
            
            # Ensure the model is prepared for inference using Unsloth
            from unsloth import FastLanguageModel
            model = FastLanguageModel.for_inference(model)
            if not hasattr(model, '_inference_ready'):
                print("DEBUG: ModelManager.load_model - _inference_ready attribute missing after for_inference, patching manually.")
                model._inference_ready = True
            
            self.loaded_model = model
            self.loaded_tokenizer = tokenizer
            self.current_model_name = model_name
            
            print("Model loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate response using the actual model."""
    # Prepare the prompt
    full_prompt = f"""Generate ErgoScript code or explanation for the following request:

{prompt}

Response:"""
    
    # Tokenize and generate
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    if hasattr(model, '_inference_ready'):
        print("DEBUG: In generate_response, model _inference_ready flag:", model._inference_ready)
    else:
        print("DEBUG: In generate_response, model NOT inference ready, _inference_ready attribute missing")
    
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

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Only basic CLI interface will be available.")

def save_interaction(prompt, response, model_name):
    """Save interaction to file."""
    save_dir = "output/interactions"
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

def create_cli_interface():
    """Create a simple CLI interface."""
    print("SharkNet - ErgoScript Assistant (CLI Mode)")
    print("----------------------------------------")
    
    # Initialize model manager
    model_manager = ModelManager()
    available_models = model_manager.list_models()
    
    if not available_models:
        print("Error: No models found in saved_models directory!")
        return
    
    # Select model
    print("\nAvailable models:")
    for i, model_name in enumerate(available_models):
        print(f"{i+1}. {model_name}")
    
    model_idx = int(input("\nSelect model number: ")) - 1
    model_name = available_models[model_idx]
    
    # Load model
    model, tokenizer = model_manager.load_model(model_name)
    
    print(f"\nUsing model: {model_name}")
    print("Type 'quit' to exit, 'params' to adjust generation parameters")
    
    # Default parameters
    params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_length": 512
    }
    
    while True:
        prompt = input("\nEnter your prompt: ")
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'params':
            print("\nCurrent parameters:", json.dumps(params, indent=2))
            params["temperature"] = float(input("Enter temperature (0.1-1.0): "))
            params["top_p"] = float(input("Enter top_p (0.1-1.0): "))
            params["max_length"] = int(input("Enter max_length (64-2048): "))
            continue
        
        response = generate_response(
            model, 
            tokenizer, 
            prompt,
            max_length=params["max_length"],
            temperature=params["temperature"],
            top_p=params["top_p"]
        )
        save_path = save_interaction(prompt, response, model_name)
        
        print("\nGenerated Response:")
        print("------------------")
        print(response)
        print(f"\nInteraction saved to: {save_path}")

def create_ui():
    """Create the Gradio interface."""
    model_manager = ModelManager()
    available_models = model_manager.list_models()
    
    if not available_models:
        raise ValueError("No models found in saved_models directory!")
    
    default_model = available_models[0]
    
    with gr.Blocks(title="SharkNet - ErgoScript Assistant") as interface:
        gr.Markdown("""
        # SharkNet - ErgoScript Assistant
        
        Generate ErgoScript code and explanations using trained models.
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
                
                # Patch the model in case 'max_seq_length' is missing
                if not hasattr(model, 'max_seq_length'):
                    model.max_seq_length = getattr(model.config, 'max_position_embeddings', 512)
                    print(f"Patched model.max_seq_length to {model.max_seq_length} in on_submit")
                
                response = generate_response(
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

# New helper functions for loading and running test cases
import os
import json

# Function to load test cases from a JSON file

def load_test_cases():
    test_cases_file = os.path.join('prototype', 'test_cases.json')
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    return test_cases


# Function to run inference on each test case and print the responses

def run_inference_on_test_cases(model, tokenizer, test_cases):
    # Ensure the model is set to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    for difficulty, questions in test_cases.items():
        print(f"\n--- Difficulty: {difficulty.upper()} ---")
        for idx, question in enumerate(questions, start=1):
            print(f"\nTest {idx}: {question}")
            # Prepare input for the model
            inputs = tokenizer(question, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")

def get_free_port(start_port=7860, end_port=7870):
    """Find a free port in the given range."""
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start_port}-{end_port - 1}")

# Updated __main__ block to integrate test cases into offline inference
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inference Standalone CLI")
    parser.add_argument('--run_mode', choices=['test', 'ui'], default='test', help='Choose run mode: "test" to run offline test cases; "ui" to launch the Gradio UI interface')
    args = parser.parse_args()
    
    from pipeline.model import create_model
    # Create the model and tokenizer
    model, tokenizer = create_model()
    
    # Patch the model for inference using Unsloth
    from unsloth import FastLanguageModel
    model = FastLanguageModel.for_inference(model)
    print("DEBUG: After for_inference call. Model type:", type(model))
    if hasattr(model, '_inference_ready'):
        print("DEBUG: Model is inference ready, _inference_ready flag:", model._inference_ready)
    else:
        print("DEBUG: Model is NOT inference ready, missing _inference_ready attribute")
    
    # Patch the model: set max_seq_length if not present
    if not hasattr(model, 'max_seq_length'):
        model.max_seq_length = getattr(model.config, 'max_position_embeddings', 512)
        print(f"Patched model.max_seq_length to {model.max_seq_length}")
    
    if args.run_mode == 'test':
        # Load test cases from the JSON file and run inference
        test_cases = load_test_cases()
        run_inference_on_test_cases(model, tokenizer, test_cases)
    elif args.run_mode == 'ui':
        if GRADIO_AVAILABLE:
            free_port = get_free_port()
            print(f"Starting Gradio interface on port {free_port}...")
            interface = create_ui()
            interface.launch(server_name="0.0.0.0", server_port=free_port)
        else:
            print("Gradio is not installed. Falling back to CLI interface...")
            create_cli_interface() 