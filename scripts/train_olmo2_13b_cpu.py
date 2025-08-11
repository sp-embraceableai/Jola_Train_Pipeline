#!/usr/bin/env python3
"""
CPU Training script for embraceableAI/OLMo-2-1124-13B-Instruct-16k-Context-FT-ChatML
Optimized for CPU training (slower but compatible)
"""

import sys
import os
import torch
import argparse
import json
from pathlib import Path

# Add the parent directory to the path to import JOLA modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from jola.modeling_olmo2 import JoLAOlmo2Model
from jola.trainers import JoLATrainer, make_data_collator
from jola.dataset_sharegpt import ShareGPTDatasetFromFile

def fix_model_config(model_dir):
    """Fix model config to use standard architecture name"""
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'architectures' in config and config['architectures']:
                if 'JoLAOlmo2Model' in config['architectures']:
                    config['architectures'] = ['Olmo2ForCausalLM']
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    print(f"Fixed model config: {config_path}")
                else:
                    print(f"Model config already correct: {config_path}")
        except Exception as e:
            print(f"Warning: Could not fix model config: {e}")

def setup_environment():
    """Setup CPU training environment"""
    # Set environment variables for CPU training
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
    
    # Disable CUDA for CPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    print("Environment configured for CPU training")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")

def load_model_and_tokenizer():
    """Load model and tokenizer optimized for CPU"""
    model_name = "embraceableAI/OLMo-2-1124-13B-Instruct-16k-Context-FT-ChatML"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding side for training
    tokenizer.padding_side = 'right'
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print("Loading model for CPU...")
    model = JoLAOlmo2Model.jola_from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None,  # No device mapping for CPU
        use_flash_attention_2=False,  # Not available on CPU
        use_cache=False,  # Disable KV cache during training
    )
    
    # Resize embeddings if we added a pad token
    if tokenizer.pad_token == '[PAD]':
        model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")
    
    # Unfreeze JOLA parameters
    model.unfreeze_jola_params()
    
    # Move model to CPU explicitly
    model = model.to('cpu')
    
    # Set model to training mode
    model.train()
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """Prepare the dataset for CPU training"""
    print("Loading ShareGPT dataset...")
    
    dataset = ShareGPTDatasetFromFile(
        data_path="data/sharegpt_training_data_combined_final_solution_correctness_100.jsonl",
        train_size=100,  # Much smaller for CPU
        valid_size=20,
        test_size=20,
        format_type="chatml",
        max_length=1024  # Shorter sequences for CPU
    )
    
    data = dataset.data_from_file()
    
    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(data['train'])}")
    print(f"Validation samples: {len(data['valid'])}")
    print(f"Test samples: {len(data['test'])}")
    
    return data

def create_trainer(model, tokenizer, data):
    """Create the JOLA trainer optimized for CPU"""
    # Create data collator
    data_collator = make_data_collator(tokenizer=tokenizer)
    
    # Create training arguments optimized for CPU
    training_args = TrainingArguments(
        # Learning Rate & Optimization
        learning_rate=0.0001,
        lr_scheduler_type='cosine',
        warmup_steps=50,  # Reduced for CPU
        weight_decay=0.01,
        
        # Batch Sizes (optimized for CPU)
        per_device_train_batch_size=1,  # Small batch for CPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Compensate with accumulation
        
        # Training Duration
        num_train_epochs=3,  # Reduced for CPU testing
        
        # Evaluation & Saving
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=False,  # Disable to save memory
        save_total_limit=1,  # Keep only latest
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        
        # Logging & Monitoring
        report_to=None,  # Disable W&B for CPU
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        
        # Reproducibility
        seed=42,
        dataloader_pin_memory=False,  # Disable for CPU
        
        # Training Mode
        do_train=True,
        do_eval=True,
        do_predict=False,
        
        # Precision & Memory Optimization
        bf16=False,  # Not supported on CPU
        fp16=False,  # Not supported on CPU
        dataloader_num_workers=0,  # Single thread
        
        # Gradient Clipping
        max_grad_norm=1.0,
        
        # Output
        output_dir='./outputs/olmo2_13b_instruct_16k_cpu',
        overwrite_output_dir=True,
        
        # Checkpointing
        save_steps=1000,
        eval_steps=1000,
        
        # Early Stopping
        early_stopping_patience=2,
        early_stopping_threshold=0.001
    )
    
    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.001
    )
    
    # Create JOLA trainer
    trainer = JoLATrainer(
        model=model,
        train_dataset=data['train'],
        eval_dataset=data['valid'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        callbacks=[early_stopping_callback],
        gate_scheduler="expon"
    )
    
    # Set gate schedule
    trainer.gated_lambda = 0.00004
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Train OLMo2 13B model with JOLA on CPU")
    
    args = parser.parse_args()
    
    # Print configuration summary
    print("\n" + "="*60)
    print("CPU TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print("Model: embraceableAI/OLMo-2-1124-13B-Instruct-16k-Context-FT-ChatML")
    print("Hardware: CPU optimized")
    print("Batch Size: 1 (with gradient accumulation of 8 = effective batch size 8)")
    print("Precision: float32")
    print("Memory Optimizations: Gradient Checkpointing")
    print("Max Sequence Length: 1,024 tokens")
    print("Training Data: 100 examples")
    print("Validation Data: 20 examples")
    print("Output Directory: ./outputs/olmo2_13b_instruct_16k_cpu")
    print("="*60 + "\n")
    
    print("WARNING: CPU training will be significantly slower than GPU training!")
    print("Consider using smaller models for CPU or using GPU for better performance.\n")
    
    # Setup environment
    setup_environment()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare dataset
    data = prepare_dataset(tokenizer)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, data)
    
    # Start training
    print("Starting CPU training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Save final model
        final_output_dir = os.path.join('./outputs/olmo2_13b_instruct_16k_cpu', 'final')
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        print(f"Final model saved to {final_output_dir}")
        
        # Fix model config to use standard architecture name
        fix_model_config(final_output_dir)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=data['test'])
    print(f"Test results: {test_results}")

if __name__ == "__main__":
    main()
