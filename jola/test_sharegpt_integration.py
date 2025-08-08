#!/usr/bin/env python3
"""
Test script to verify ShareGPT dataset integration with JoLA
"""

import json
import tempfile
import os
from jola import ShareGPTDataset, create_sharegpt_dataset

def create_sample_sharegpt_data():
    """Create sample ShareGPT data for testing"""
    sample_data = [
        {
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris."}
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Explain photosynthesis."},
                {"from": "gpt", "value": "Photosynthesis is the process by which plants convert sunlight into energy through chemical reactions."}
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "What is 2+2?"},
                {"from": "gpt", "value": "2+2 equals 4."}
            ]
        }
    ]
    return sample_data

def test_sharegpt_dataset():
    """Test the ShareGPT dataset loading"""
    
    print("Testing ShareGPT dataset integration...")
    
    # Create temporary file with sample data
    sample_data = create_sample_sharegpt_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
        temp_file = f.name
    
    try:
        # Test basic dataset loading
        dataset = ShareGPTDataset(
            data_path=temp_file,
            train_size=2,
            valid_size=1,
            test_size=0,
            format_type="chatml"
        )
        
        data = dataset.load_data()
        
        print("✓ ShareGPT dataset loaded successfully")
        print(f"✓ Train examples: {len(data['train'])}")
        print(f"✓ Valid examples: {len(data['valid'])}")
        print(f"✓ Test examples: {len(data['test'])}")
        
        # Test ChatML formatting
        if len(data['train']) > 0:
            sample_text = data['train'][0]['text']
            print(f"✓ Sample ChatML format: {sample_text[:100]}...")
            
            # Check if it contains ChatML tokens
            if "<|im_start|>" in sample_text and "<|im_end|>" in sample_text:
                print("✓ ChatML formatting is correct")
            else:
                print("✗ ChatML formatting is incorrect")
                return False
        
        # Test instruction formatting
        dataset_instruction = ShareGPTDataset(
            data_path=temp_file,
            train_size=2,
            format_type="instruction"
        )
        
        data_instruction = dataset_instruction.load_data()
        
        if len(data_instruction['train']) > 0:
            sample_text = data_instruction['train'][0]['text']
            print(f"✓ Sample instruction format: {sample_text[:100]}...")
            
            # Check if it contains instruction format
            if "### Instruction:" in sample_text and "### Response:" in sample_text:
                print("✓ Instruction formatting is correct")
            else:
                print("✗ Instruction formatting is incorrect")
                return False
        
        # Test convenience function
        data_conv = create_sharegpt_dataset(
            data_path=temp_file,
            train_size=2,
            format_type="chatml"
        )
        
        print("✓ Convenience function works correctly")
        
        print("\n🎉 All ShareGPT dataset tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ ShareGPT dataset test failed: {e}")
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_sharegpt_with_large_sample():
    """Test with a larger sample to verify sampling works"""
    
    print("\nTesting ShareGPT dataset with larger sample...")
    
    # Create larger sample data
    sample_data = []
    for i in range(100):
        sample_data.append({
            "conversations": [
                {"from": "human", "value": f"Question {i}?"},
                {"from": "gpt", "value": f"Answer {i}."}
            ]
        })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
        temp_file = f.name
    
    try:
        # Test with 2000 samples (as requested)
        dataset = ShareGPTDataset(
            data_path=temp_file,
            train_size=2000,  # Should sample all available
            valid_size=10,
            test_size=10,
            format_type="chatml"
        )
        
        data = dataset.load_data()
        
        print(f"✓ Large dataset loaded: {len(data['train'])} training examples")
        print(f"✓ Validation examples: {len(data['valid'])}")
        print(f"✓ Test examples: {len(data['test'])}")
        
        # Verify we got the expected number of examples
        expected_train = min(2000, len(sample_data))  # 100 conversations = 100 training examples
        if len(data['train']) == expected_train:
            print("✓ Sampling worked correctly")
        else:
            print(f"✗ Expected {expected_train} training examples, got {len(data['train'])}")
            return False
        
        print("🎉 Large sample test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Large sample test failed: {e}")
        return False
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == "__main__":
    success1 = test_sharegpt_dataset()
    success2 = test_sharegpt_with_large_sample()
    
    if success1 and success2:
        print("\n🎉 All tests passed! ShareGPT integration is working correctly.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        exit(1)
