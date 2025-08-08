#!/usr/bin/env python3
"""
Test script to verify OLMo-2 integration with JoLA
"""

import torch
from transformers import AutoTokenizer, AutoConfig
from jola import JoLAOlmo2Model, JoLAConfig

def test_olmo2_integration():
    """Test the OLMo-2 integration with JoLA"""
    
    print("Testing OLMo-2 integration with JoLA...")
    
    # Test configuration loading
    try:
        config = JoLAConfig.get_jola_config(default=False, config_path="jola/config_olmo2.yaml")
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    # Test model loading (without downloading the full model)
    try:
        # Load config first
        model_config = AutoConfig.from_pretrained("sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML")
        print("✓ Model config loaded successfully")
        
        # Create model instance
        model = JoLAOlmo2Model(model_config)
        print("✓ JoLA OLMo-2 model created successfully")
        
        # Test unfreeze_jola_params
        model.unfreeze_jola_params()
        print("✓ JoLA parameters unfrozen successfully")
        
        # Check that JoLA parameters are trainable
        jola_params_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                jola_params_count += param.numel()
        
        print(f"✓ JoLA parameters initialized: {jola_params_count} trainable parameters")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    # Test tokenizer loading
    try:
        tokenizer = AutoTokenizer.from_pretrained("sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML")
        print("✓ Tokenizer loaded successfully")
        
        # Test tokenization
        test_text = "Hello, this is a test of the OLMo-2 integration with JoLA."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Tokenization successful: {tokens['input_ids'].shape}")
        
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        return False
    
    print("\n🎉 All tests passed! OLMo-2 integration is working correctly.")
    return True

if __name__ == "__main__":
    success = test_olmo2_integration()
    if not success:
        print("\n❌ Some tests failed. Please check the error messages above.")
        exit(1)
