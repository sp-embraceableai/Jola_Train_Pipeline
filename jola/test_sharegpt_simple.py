#!/usr/bin/env python3
"""
Simple test script to verify ShareGPT dataset loading
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from jola.dataset_sharegpt import ShareGPTDataset

def test_sharegpt_loading():
    """Test ShareGPT dataset loading with the actual file"""
    
    print("Testing ShareGPT dataset loading with actual file...")
    
    # Check if file exists
    data_file = "sharegpt_training_data_combined_final_solution_correctness_100.jsonl"
    if not os.path.exists(data_file):
        print(f"✗ File not found: {data_file}")
        return False
    
    print(f"✓ Found data file: {data_file}")
    
    try:
        # Load dataset with smaller sample for testing
        dataset = ShareGPTDataset(
            data_path=data_file,
            train_size=10,  # Small sample for testing
            valid_size=5,
            test_size=5,
            format_type="chatml"
        )
        
        data = dataset.load_data()
        
        print("✓ ShareGPT dataset loaded successfully")
        print(f"✓ Train examples: {len(data['train'])}")
        print(f"✓ Valid examples: {len(data['valid'])}")
        print(f"✓ Test examples: {len(data['test'])}")
        
        # Show a sample
        if len(data['train']) > 0:
            sample_text = data['train'][0]['text']
            print(f"✓ Sample text (first 200 chars): {sample_text[:200]}...")
            
            # Check if it contains ChatML tokens
            if "<|im_start|>" in sample_text and "<|im_end|>" in sample_text:
                print("✓ ChatML formatting is correct")
            else:
                print("✗ ChatML formatting is incorrect")
                return False
        
        print("\n🎉 ShareGPT dataset loading test passed!")
        return True
        
    except Exception as e:
        print(f"✗ ShareGPT dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sharegpt_loading()
    if not success:
        print("\n❌ Test failed.")
        exit(1)
    else:
        print("\n✅ All tests passed!")
        exit(0)
