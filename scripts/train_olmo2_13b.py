#!/usr/bin/env python3
"""
Training script for embraceableAI/OLMo-2-1124-13B-Instruct-16k-Context-FT-ChatML
Optimized for A100 80GB GPU training with JOLA framework
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add the parent directory to the path to import JOLA modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from jola.modeling_olmo2 import JoLAOlmo2Model
from jola.trainers import JoLATrainer, make_data_collator
from jola.dataset_sharegpt import ShareGPTDatasetFromFile

def setup_environment():
    """Setup training environment"""
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Enable flash attention if available
    try:
        import flash_attn
        print("Flash Attention available and enabled")
    except ImportError:
        print("Flash Attention not available, using standard attention")

def load_model_and_tokenizer():
    """Load model and tokenizer with optimal settings for A100"""
    model_name = "embraceableAI/OLMo-2-1124-13B-Instruct-16k-Context-FT-ChatML"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding side for training
    tokenizer.padding_side = 'right'
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print("Loading model...")
    model = JoLAOlmo2Model.jola_from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_flash_attention_2=True,
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
    
    # Set model to training mode
    model.train()
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """Prepare the dataset for training"""
    print("Loading ShareGPT dataset...")
    
    dataset = ShareGPTDatasetFromFile(
        data_path="data/sharegpt_training_data_combined_final_solution_correctness_100.jsonl",
        train_size=5000,
        valid_size=500,
        test_size=500,
        format_type="chatml",
        max_length=30000  # Extended context length
    )
    
    data = dataset.data_from_file()
    
    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(data['train'])}")
    print(f"Validation samples: {len(data['valid'])}")
    print(f"Test samples: {len(data['test'])}")
    
    return data

def create_trainer(model, tokenizer, data):
    """Create the JOLA trainer"""
    # Create data collator
    data_collator = make_data_collator(tokenizer=tokenizer)
    
    # Create training arguments optimized for A100
    training_args = TrainingArguments(
        # Learning Rate & Optimization
        learning_rate=0.0001,
        lr_scheduler_type='cosine',
        warmup_steps=100,
        weight_decay=0.01,
        
        # Batch Sizes (optimized for A100 80GB)
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        
        # Training Duration
        num_train_epochs=10,
        
        # Evaluation & Saving
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_total_limit=3,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        
        # Logging & Monitoring
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        
        # Reproducibility
        seed=42,
        dataloader_pin_memory=True,
        
        # Training Mode
        do_train=True,
        do_eval=True,
        do_predict=False,
        
        # Precision & Memory Optimization
        bf16=True,  # Use bfloat16 for A100
        fp16=False,
        dataloader_num_workers=4,
        
        # Gradient Clipping
        max_grad_norm=1.0,
        
        # Output
        output_dir='./outputs/olmo2_13b_instruct_16k',
        overwrite_output_dir=True,
        
        # Checkpointing
        save_steps=500,
        eval_steps=500,
        
        # Early Stopping
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
        
        # Max Sequence Length
        max_length=30000
    )
    
    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
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
    parser = argparse.ArgumentParser(description="Train OLMo2 13B model with JOLA")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Print configuration summary
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print("Model: embraceableAI/OLMo-2-1124-13B-Instruct-16k-Context-FT-ChatML")
    print("Hardware: A100 80GB GPU optimized")
    print("Batch Size: 4 (with gradient accumulation of 4 = effective batch size 16)")
    print("Precision: bfloat16")
    print("Memory Optimizations: Flash Attention, Gradient Checkpointing")
    print("Max Sequence Length: 30,000 tokens")
    print("Training Data: 5000 examples")
    print("Validation Data: 500 examples")
    print("Output Directory: ./outputs/olmo2_13b_instruct_16k")
    print("="*60 + "\n")
    
    # Setup environment
    setup_environment()
    
    # Setup Weights & Biases
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project="jola-olmo2-13b-training",
                name="olmo2-13b-instruct-16k-sharegpt",
                tags=["olmo2", "13b", "jola", "sharegpt"]
            )
            print("Weights & Biases logging initialized successfully")
        except ImportError:
            print("Warning: wandb not installed. Logging will be disabled.")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare dataset
    data = prepare_dataset(tokenizer)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, data)
    
    # Start training
    print("Starting training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Save final model
        final_output_dir = os.path.join('./outputs/olmo2_13b_instruct_16k', 'final')
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        print(f"Final model saved to {final_output_dir}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=data['test'])
    print(f"Test results: {test_results}")

if __name__ == "__main__":
    main()
