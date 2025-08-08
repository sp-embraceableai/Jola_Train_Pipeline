import sys, os
# Add the jola directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
import torch

# Import from the local jola package
from jola.modeling_olmo2 import JoLAOlmo2Model
from jola.configuration_olmo2 import Olmo2Config

def run_olmo2_0425_1b_sft_inference():
    """
    Simple inference script for allenai/OLMo-2-0425-1B-SFT model using JOLA implementation
    """
    model_name = "allenai/OLMo-2-0425-1B-SFT"
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = Olmo2Config.from_pretrained(model_name)
    
    # Load model using JOLA implementation
    model = JoLAOlmo2Model.jola_from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=None,  # Use default cache directory
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"Device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence."
    ]
    
    print("\n" + "="*50)
    print("Running inference examples:")
    print("="*50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nExample {i}:")
        print(f"Prompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()  # Remove the input prompt
        
        print(f"Response: {response}")
        print("-" * 30)

if __name__ == "__main__":
    run_olmo2_0425_1b_sft_inference()
