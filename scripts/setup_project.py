#!/usr/bin/env python3
"""
Setup script to organize the project structure and prepare data
"""

import os
import shutil
import argparse
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        'configs',
        'models',
        'scripts',
        'data',
        'logs',
        'outputs',
        'outputs/olmo2_13b_instruct_16k',
        'checkpoints'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_data():
    """Setup the data directory"""
    # Check if the ShareGPT dataset exists
    source_file = "sharegpt_training_data_combined_final_solution_correctness_100.jsonl"
    target_file = "data/sharegpt_training_data_combined_final_solution_correctness_100.jsonl"
    
    if os.path.exists(source_file):
        if not os.path.exists(target_file):
            shutil.copy2(source_file, target_file)
            print(f"Copied dataset to: {target_file}")
        else:
            print(f"Dataset already exists at: {target_file}")
    else:
        print(f"Warning: Source dataset not found at {source_file}")
        print("Please ensure the ShareGPT dataset is available in the project root")

def main():
    parser = argparse.ArgumentParser(description="Setup JOLA OLMo2 13B training project")
    parser.add_argument("--skip-data", action="store_true", help="Skip data setup")
    
    args = parser.parse_args()
    
    print("Setting up JOLA OLMo2 13B training project...")
    
    # Create directory structure
    create_directory_structure()
    
    # Setup data
    if not args.skip_data:
        setup_data()
    
    print("\nProject setup completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start training: python scripts/train_olmo2_13b.py")
    print("3. Monitor training: Check logs/ directory and W&B dashboard")

if __name__ == "__main__":
    main()
