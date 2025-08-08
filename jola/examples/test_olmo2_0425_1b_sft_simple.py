import sys, os
# Add the jola directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
import torch

# Import from the local jola package
from jola.modeling_olmo2 import JoLAOlmo2Model
from jola.configuration_olmo2 import Olmo2Config

def test_olmo2_0425_1b_sft():
    """
    Simple test script for allenai/OLMo-2-0425-1B-SFT model using JOLA implementation
    """
    model_name = "allenai/OLMo-2-0425-1B-SFT"
    
    print(f"Testing model: {model_name}")
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = Olmo2Config.from_pretrained(model_name)
    
    print(f"Config loaded: {config}")
    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    
    # Load model using JOLA implementation
    model = JoLAOlmo2Model.jola_from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=None,  # Use default cache directory
        config=config,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None  # Don't use device_map for CPU
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test basic forward pass
    test_text = "Hello, how are you?"
    print(f"\nTesting forward pass with: '{test_text}'")
    
    # Tokenize input
    inputs = tokenizer(test_text, return_tensors="pt")
    print(f"Input tokens: {inputs}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Output logits sample: {outputs.logits[0, -1, :10]}")  # Last token, first 10 logits
    
    print("\nModel test completed successfully!")

if __name__ == "__main__":
    test_olmo2_0425_1b_sft()
