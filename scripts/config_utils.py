import yaml
import os
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    jola_config: Dict[str, Any]
    hardware_config: Dict[str, Any]
    logging_config: Dict[str, Any]

def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TrainingConfig(**config_dict)

def validate_gpu_requirements(config: TrainingConfig) -> bool:
    """Validate GPU requirements for the configuration"""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Training will use CPU.")
        return False
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    print(f"Available GPU memory: {gpu_memory:.1f} GB")
    
    # Estimate memory requirements for 13B model
    model_size_gb = 13 * 2  # 13B parameters * 2 bytes per parameter (bfloat16)
    batch_size = config.training_config['per_device_train_batch_size']
    gradient_accumulation = config.training_config.get('gradient_accumulation_steps', 1)
    effective_batch_size = batch_size * gradient_accumulation
    
    estimated_memory = model_size_gb * (1 + effective_batch_size * 0.1)  # Rough estimate
    
    print(f"Estimated memory requirement: {estimated_memory:.1f} GB")
    
    if gpu_memory < estimated_memory:
        print(f"Warning: GPU memory ({gpu_memory:.1f} GB) may be insufficient for estimated requirement ({estimated_memory:.1f} GB)")
        return False
    
    return True

def setup_wandb(config: TrainingConfig) -> None:
    """Setup Weights & Biases logging"""
    try:
        import wandb
        wandb.init(
            project=config.logging_config['project_name'],
            name=config.logging_config['run_name'],
            tags=config.logging_config['tags'],
            config={
                'model_config': config.model_config,
                'training_config': config.training_config,
                'data_config': config.data_config,
                'jola_config': config.jola_config,
                'hardware_config': config.hardware_config
            }
        )
        print("Weights & Biases logging initialized successfully")
    except ImportError:
        print("Warning: wandb not installed. Logging will be disabled.")
        config.training_config['report_to'] = None

def print_config_summary(config: TrainingConfig) -> None:
    """Print a summary of the training configuration"""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"\nModel: {config.model_config['pretrained_model_name_or_path']}")
    print(f"Device: {config.model_config['device']}")
    
    print(f"\nTraining Parameters:")
    print(f"  Learning Rate: {config.training_config['learning_rate']}")
    print(f"  Batch Size: {config.training_config['per_device_train_batch_size']}")
    print(f"  Gradient Accumulation: {config.training_config.get('gradient_accumulation_steps', 1)}")
    print(f"  Effective Batch Size: {config.training_config['per_device_train_batch_size'] * config.training_config.get('gradient_accumulation_steps', 1)}")
    print(f"  Epochs: {config.training_config['num_train_epochs']}")
    print(f"  Warmup Steps: {config.training_config['warmup_steps']}")
    
    print(f"\nData Configuration:")
    print(f"  Train Size: {config.data_config['train_size']}")
    print(f"  Valid Size: {config.data_config['valid_size']}")
    print(f"  Max Length: {config.data_config['max_length']}")
    
    print(f"\nHardware Configuration:")
    print(f"  Precision: {config.hardware_config['torch_dtype']}")
    print(f"  Flash Attention: {config.hardware_config['use_flash_attention']}")
    print(f"  Gradient Checkpointing: {config.hardware_config['use_gradient_checkpointing']}")
    
    print(f"\nOutput Directory: {config.training_config['output_dir']}")
    print("="*60 + "\n")
