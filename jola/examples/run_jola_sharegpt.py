from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback

import sys, os, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import from local development version
from jola.config import JoLAConfig
from jola.modeling_olmo2 import JoLAOlmo2Model
from jola.trainers import JoLATrainer, make_data_collator
from jola.dataset_sharegpt import ShareGPTDataset

## if you have already install jola through pip, you can directly import them
# from jola import JoLAConfig, JoLAOlmo2Model, JoLATrainer, ShareGPTDataset, make_data_collator

# set the jola config through a yaml file
jola_config = JoLAConfig.get_jola_config(default=False, config_path="jola/config_sharegpt.yaml")

jola_tokenizer = AutoTokenizer.from_pretrained(**jola_config["model_config"])

# Use right padding for training
jola_tokenizer.padding_side = 'right'

# Load models
jola_model = JoLAOlmo2Model.jola_from_pretrained(**jola_config["model_config"])

# unfreeze jola parameters
jola_model.unfreeze_jola_params()

# set in training mode
jola_model.model.train()

# Define padding token if needed
if jola_tokenizer.pad_token is None:
    jola_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    jola_model.resize_token_embeddings(jola_model.config.vocab_size + 1)

# data setting, data loader
data_collator = make_data_collator(tokenizer=jola_tokenizer)

# dataset setting - ShareGPT dataset
jola_dataset = ShareGPTDataset(
    data_path=jola_config["data_config"]["data_path"],
    train_size=jola_config["data_config"]["train_size"],
    valid_size=jola_config["data_config"]["valid_size"],
    test_size=jola_config["data_config"]["test_size"],
    format_type=jola_config["data_config"]["format_type"]
)
jola_data = jola_dataset.load_data()

# early stop according to the performance from validation set
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.0
)

training_args = TrainingArguments(**jola_config["training_config"])

# trainer
jola_trainer = JoLATrainer(
    jola_model,
    train_dataset=jola_data['train'],
    eval_dataset=jola_data['valid'],
    tokenizer=jola_tokenizer,
    data_collator=data_collator,
    args=training_args,
    callbacks=[early_stopping_callback],
    gate_scheduler=jola_config["jola_config"]["gate_scheduler"]
)

torch.autograd.set_detect_anomaly(True)

# set gate schedule
if not jola_config["jola_config"]["gate_scheduler"]:
    jola_trainer.gated_lambda = jola_config['training_config']["gate_lambda"]

print("Starting training with ShareGPT data...")
print(f"Training examples: {len(jola_data['train'])}")
print(f"Validation examples: {len(jola_data['valid'])}")
print(f"Test examples: {len(jola_data['test'])}")

jola_trainer.train()

print("Training completed!")

# Optional: Save the trained model
output_dir = jola_config["training_config"]["output_dir"]
jola_model.save_pretrained(f"{output_dir}/final_model")
jola_tokenizer.save_pretrained(f"{output_dir}/final_model")
print(f"Model saved to {output_dir}/final_model")
