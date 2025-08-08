import sys, os, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from jola.config import JoLAConfig
from jola.modeling_olmo2 import JoLAOlmo2Model
from jola.trainers import JoLATrainer, make_data_collator
from jola.dataset import JoLADataset

## if you have already install jola through pip, you can directly import them
# from jola import JoLAConfig, JoLAOlmo2Model, JoLATrainer, JoLADataset, make_data_collator

# Configuration for allenai/OLMo-2-0425-1B-SFT model
model_config = {
    "pretrained_model_name_or_path": "allenai/OLMo-2-0425-1B-SFT",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_dir": None,  # Use default cache directory
    "applied_module": 'attention',
    "base_model_name": 'olmo2_0425_1b_sft'
}

training_config = {
    "learning_rate": 0.005,
    "lr_scheduler_type": 'cosine',
    "warmup_steps": 50,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "num_train_epochs": 20,
    "evaluation_strategy": 'epoch',
    "save_strategy": 'epoch',
    "load_best_model_at_end": True,
    "save_total_limit": 1,
    "report_to": "wandb",
    "logging_strategy": "epoch",
    "seed": 42,
    "do_train": True,
    "do_eval": True,
    "bf16": True if torch.cuda.is_available() else False,
    "output_dir": './output_olmo2_0425_1b_sft'
}

data_config = {
    "train_size": 200,
    "task_name": "common_reason",
    "data_path": "/dss/dssmcmlfs01/pn39je/pn39je-dss-0001/go52tox/lavine_prj/paper_prj/rep_pruning/dataset/data_with_instruct/commonsense/ARC-c"
}

jola_config = {
    "gate_lambda": 0.00004,
    "gate_scheduler": "expon"
}

# Create combined config
jola_config_dict = {
    "model_config": model_config,
    "training_config": training_config,
    "data_config": data_config,
    "jola_config": jola_config
}

jola_tokenizer = AutoTokenizer.from_pretrained(**jola_config_dict["model_config"])

# Use right padding for training
jola_tokenizer.padding_side = 'right'

# Load models
jola_model = JoLAOlmo2Model.jola_from_pretrained(**jola_config_dict["model_config"])

# unfreeze jola parameters
jola_model.unfreeze_jola_params()

# set in training mode
jola_model.model.train()

# Define padding
if jola_tokenizer.pad_token is None:
    jola_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    jola_model.resize_token_embeddings(jola_model.config.vocab_size + 1)

# data setting, data loader
data_collator = make_data_collator(tokenizer=jola_tokenizer)

# dataset setting
jola_dataset = JoLADataset(data_path=jola_config_dict["data_config"]["data_path"])
jola_data = jola_dataset.data_from_file()

# early stop according to the performance from validation set
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.0
)

training_args = TrainingArguments(**jola_config_dict["training_config"])

# trainer
jola_trainer = JoLATrainer(
    jola_model,
    train_dataset=jola_data['train'],
    eval_dataset = jola_data['valid'],
    tokenizer=jola_tokenizer,
    data_collator = data_collator,
    args=training_args,
    callbacks=[early_stopping_callback],
    gate_scheduler=jola_config_dict["jola_config"]["gate_scheduler"]
)

torch.autograd.set_detect_anomaly(True)

# set gate schedule
if not jola_config_dict["jola_config"]["gate_scheduler"]:
    jola_trainer.gated_lambda = jola_config_dict['training_config']["gate_lambda"]

print(f"Starting training with model: {model_config['pretrained_model_name_or_path']}")
print(f"Device: {model_config['device']}")
print(f"Output directory: {training_config['output_dir']}")

jola_trainer.train()

# do evaluation ** double check
# jola_trainer.test(fname=args.output_file_name, task=args.task, subtask=args.subtask, eval_dataset=jola_data['test'],model_name = args.base_model_name)
