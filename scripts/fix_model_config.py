#!/usr/bin/env python3
"""
Utility script to fix model architecture in config.json after JOLA training.
Converts JoLAOlmo2Model back to Olmo2ForCausalLM for compatibility.
"""

import json
import os
import argparse
import glob
from pathlib import Path

def fix_config_file(config_path):
    """Fix a single config.json file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if the architecture needs fixing
        if 'architectures' in config and config['architectures']:
            if 'JoLAOlmo2Model' in config['architectures']:
                # Replace JoLAOlmo2Model with Olmo2ForCausalLM
                config['architectures'] = ['Olmo2ForCausalLM']
                
                # Write back the fixed config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"Fixed: {config_path}")
                return True
            else:
                print(f"No fix needed: {config_path}")
                return False
    except Exception as e:
        print(f"Error processing {config_path}: {e}")
        return False

def find_and_fix_configs(directory, recursive=True):
    """Find and fix all config.json files in a directory"""
    if recursive:
        pattern = os.path.join(directory, "**/config.json")
        config_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, "config.json")
        config_files = glob.glob(pattern)
    
    fixed_count = 0
    total_count = len(config_files)
    
    print(f"Found {total_count} config.json files")
    
    for config_file in config_files:
        if fix_config_file(config_file):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} out of {total_count} config files")
    return fixed_count

def main():
    parser = argparse.ArgumentParser(
        description="Fix model architecture in config.json files after JOLA training"
    )
    parser.add_argument(
        "path",
        help="Path to config.json file or directory containing config files"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively in subdirectories"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file() and path.name == "config.json":
        # Fix a single config file
        if fix_config_file(str(path)):
            print("Config file fixed successfully!")
        else:
            print("Config file didn't need fixing or encountered an error.")
    elif path.is_dir():
        # Fix all config files in directory
        fixed_count = find_and_fix_configs(str(path), not args.no_recursive)
        if fixed_count > 0:
            print(f"Successfully fixed {fixed_count} config files!")
        else:
            print("No config files needed fixing.")
    else:
        print(f"Error: {path} is not a valid file or directory")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
