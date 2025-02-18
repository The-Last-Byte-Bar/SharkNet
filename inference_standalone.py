import os
import json
from datetime import datetime

def mock_generate_response(prompt):
    """Mock generation function for testing without model dependencies."""
    return f"""Here's a sample response for: {prompt}
    
// This is a mock response for testing the UI and interaction
{{
    // Sample ErgoScript code
    val recipientPK = PK("9f1...")
    val deadline = HEIGHT > 100000
    
    recipientPK && deadline
}}"""

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Only basic CLI interface will be available.")

class SimpleModelManager:
    """Simplified model manager for testing."""
    def __init__(self):
        self.models_dir = "saved_models"
        
    def list_models(self):
        """List models or return mock data if directory doesn't exist."""
        if not os.path.exists(self.models_dir):
            return ["mock_model_1", "mock_model_2"]  # Mock data for testing
        
        models = []
        for item in os.listdir(self.models_dir):
            if os.path.isdir(os.path.join(self.models_dir, item)):
                models.append(item)
        return sorted(models, reverse=True)

def save_interaction(prompt, response, model_name="test_model"):
    """Save interaction to file."""
    save_dir = "output/interactions"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interaction = {
        "timestamp": timestamp,
        "model": model_name,
        "prompt": prompt,
        "response": response
    }
    
    filename = os.path.join(save_dir, f"interaction_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(interaction, f, indent=2)
    
    return filename

def create_cli_interface():
    """Create a simple CLI interface for testing."""
    print("SharkNet - ErgoScript Assistant (CLI Mode)")
    print("----------------------------------------")
    
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        response = mock_generate_response(prompt)
        save_path = save_interaction(prompt, response)
        
        print("\nGenerated Response:")
        print("------------------")
        print(response)
        print(f"\nInteraction saved to: {save_path}")

def create_ui():
    """Create the Gradio interface for testing."""
    model_manager = SimpleModelManager()
    available_models = model_manager.list_models()
    default_model = available_models[0] if available_models else None
    
    with gr.Blocks(title="SharkNet - ErgoScript Assistant (Test Mode)") as interface:
        gr.Markdown("""
        # SharkNet - ErgoScript Assistant (Test Mode)
        
        This is a simplified version for testing the UI without model dependencies.
        All responses are mock data.
        """)
        
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="Select Model (Mock)",
                    info="Mock model selection for testing"
                )
                
                input_prompt = gr.Textbox(
                    lines=4,
                    label="Your Question/Prompt",
                    placeholder="Ask about ErgoScript or describe the smart contract you want to create..."
                )
                
                generate_btn = gr.Button("Generate")
                
                output_text = gr.Textbox(
                    lines=10,
                    label="Generated Response (Mock)",
                    show_copy_button=True
                )
        
        # Example prompts
        examples = gr.Examples(
            examples=[
                ["Create a simple token lock contract"],
                ["Explain how to implement a time-locked vault"],
            ],
            inputs=input_prompt
        )
        
        def on_submit(model_name, prompt):
            response = mock_generate_response(prompt)
            save_interaction(prompt, response, model_name)
            return response
        
        generate_btn.click(
            fn=on_submit,
            inputs=[model_dropdown, input_prompt],
            outputs=output_text
        )
    
    return interface

if __name__ == "__main__":
    if GRADIO_AVAILABLE:
        print("Starting Gradio interface...")
        interface = create_ui()
        interface.launch(server_name="0.0.0.0", server_port=7860)
    else:
        create_cli_interface() 