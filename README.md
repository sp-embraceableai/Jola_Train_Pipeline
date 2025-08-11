# JOLA: Joint Localization and Activation Editing for Low-Resource Fine-Tuning

Code for the paper "JoLA: Joint Localization and Activation Editing for Low-Resource Fine-Tuning (ICML 2025)"

Paper: http://arxiv.org/abs/2502.01179

Authors: [Wen Lai](https://wenlai-lavine.github.io/)$^{1,2}$, [Alexander Fraser](https://alexfraser.github.io/)$^{1,2}$, [Ivan Titov](https://ivan-titov.org/)$^{3,4}$

$^1$ Technical University of Munich, $^2$ Munich Center for Machine Learning, $^3$ University of Edinburgh, $^4$ University of Amsterdam

Email: wen.lai@tum.de

## Overview
![JoLA](./images/framework.png)
> Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, are commonly used to adapt LLMs. However, their effectiveness is often limited in low-resource scenarios with only a few hundred examples. Recent advances in interpretability research have inspired the emergence of activation editing techniques, which modify the activations of specific model components. These methods, due to their extremely small parameter counts, show promise for small datasets. However, their performance is highly dependent on identifying the correct modules to edit and often lacks stability across different datasets. In this paper, we propose Joint Localization and Activation Editing (JoLA), a method that jointly learns (1) which heads in the Transformer to edit; (2) whether the intervention should be additive, multiplicative, or both and (3) the intervention parameters themselves - vectors applied as additive offsets or multiplicative scalings to the head output. Through evaluations on three benchmarks spanning commonsense reasoning, natural language understanding, and natural language generation, we demonstrate that JoLA consistently outperforms existing methods.

## Table of Contents
- [Installation](#installation)
- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
- [Quick Start](#quick-start)
- [OLMo2 13B Training Project](#olmo2-13b-training-project)
- [Training Examples](#training-examples)
- [Experiments in Paper](#experiments-in-paper)
- [Citation](#citation)

## Installation
+ Install **`jola`** from pip:
```bash
pip install jola
```
+ or, install our latest **`jola`** from pip+git:
```bash
pip install git+https://github.com/wenlai-lavine/jola.git
```

## Supported Models
JoLA currently supports the following model architectures:
- **LLaMA Models**: Full support for LLaMA-2 and LLaMA-3 variants
- **Qwen2 Models**: Full support for Qwen2 variants  
- **OLMo-2 Models**: Full support for OLMo-2 variants including the [OLMo-2-1124-13B-Instruct-32k-Context-ChatML](https://huggingface.co/sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML) model

## Supported Datasets
JoLA supports various dataset formats:
- **Instruction-Response**: Standard instruction-following format
- **ShareGPT**: Conversation-style datasets in JSONL format
- **ChatML**: Structured conversation format for OLMo-2 models

## Quick Start

### For LLaMA Models
```py
from jola import JoLAConfig, JoLAModel, JoLATrainer, data_from_list
from transformers import AutoTokenizer, TrainingArguments

# set default=False, if you want to specify the parameters in JoLA; you need to set the configuration by providing a yaml file (example: config.yaml).
jola_config = JoLAConfig(default=True)

# load tokenizer
jola_tokenizer = AutoTokenizer.from_pretrained(**jola_config["model_config"])

# Load models
jola_model = JoLAModel.jola_from_pretrained(**jola_config["model_config"])

# unfreeze jola parameters
jola_model.unfreeze_jola_params()

# set in training mode
jola_model.model.train()

# examples from ARC-e
training_examples = [
    "Which statement best explains why photosynthesis is the foundation of most food webs?\n\nAnswer1: Sunlight is the source of energy for nearly all ecosystems. Answer2: Most ecosystems are found on land instead of in water. Answer3: Carbon dioxide is more available than other gases. Answer4: The producers in all ecosystems are plants.", "answer1",
    "Which piece of safety equipment is used to keep mold spores from entering the respiratory system?\n\nAnswer1: safety goggles Answer2: breathing mask Answer3: rubber gloves Answer4: lead apron.", "answer2",
    ... ...
]

# data setting, data loader
data_collator = make_data_collator(tokenizer=jola_tokenizer)

# dataset setting
jola_dataset = JoLADataset(train_list=training_examples)
jola_data = jola_dataset.data_from_list()
# training arguments
training_args = TrainingArguments(**jola_config["training_config"])
# trainer
jola_trainer = JoLATrainer(
    jola_model,
    train_dataset=jola_data['train'],
    eval_dataset = jola_data['valid'],
    tokenizer=jola_tokenizer,
    data_collator = data_collator,
    args=training_args,
    callbacks=[early_stopping_callback],
    gate_scheduler=jola_config["jola_config"]["gate_scheduler"]
)

jola_trainer.train()
```

### For OLMo-2 Models
```py
from jola import JoLAConfig, JoLAOlmo2Model, JoLATrainer, data_from_list
from transformers import AutoTokenizer, TrainingArguments

# Use OLMo-2 specific configuration
jola_config = JoLAConfig.get_jola_config(default=False, config_path="jola/config_olmo2.yaml")

# load tokenizer
jola_tokenizer = AutoTokenizer.from_pretrained(**jola_config["model_config"])

# Load OLMo-2 model
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

# dataset setting
jola_dataset = JoLADataset(data_path=jola_config["data_config"]["data_path"])
jola_data = jola_dataset.data_from_file()

# training arguments
training_args = TrainingArguments(**jola_config["training_config"])

# trainer
jola_trainer = JoLATrainer(
    jola_model,
    train_dataset=jola_data['train'],
    eval_dataset = jola_data['valid'],
    tokenizer=jola_tokenizer,
    data_collator = data_collator,
    args=training_args,
    callbacks=[early_stopping_callback],
    gate_scheduler=jola_config["jola_config"]["gate_scheduler"]
)

jola_trainer.train()
```

### For ShareGPT Data with OLMo-2
```py
from jola import JoLAConfig, JoLAOlmo2Model, JoLATrainer, ShareGPTDataset
from transformers import AutoTokenizer, TrainingArguments

# Use ShareGPT specific configuration
jola_config = JoLAConfig.get_jola_config(default=False, config_path="jola/config_sharegpt.yaml")

# load tokenizer
jola_tokenizer = AutoTokenizer.from_pretrained(**jola_config["model_config"])

# Load OLMo-2 model
jola_model = JoLAOlmo2Model.jola_from_pretrained(**jola_config["model_config"])

# unfreeze jola parameters
jola_model.unfreeze_jola_params()

# dataset setting - ShareGPT dataset
jola_dataset = ShareGPTDataset(
    data_path=jola_config["data_config"]["data_path"],
    train_size=jola_config["data_config"]["train_size"],  # 2000 samples
    valid_size=jola_config["data_config"]["valid_size"],
    test_size=jola_config["data_config"]["test_size"],
    format_type=jola_config["data_config"]["format_type"]  # "chatml"
)
jola_data = jola_dataset.load_data()

# trainer
jola_trainer = JoLATrainer(
    jola_model,
    train_dataset=jola_data['train'],
    eval_dataset=jola_data['valid'],
    tokenizer=jola_tokenizer,
    data_collator=data_collator,
    args=training_args,
    gate_scheduler=jola_config["jola_config"]["gate_scheduler"]
)

jola_trainer.train()
```

---

# OLMo2 13B Training Project

This project contains the training setup for the `embraceableAI/OLMo-2-1124-13B-Instruct-16k-Context-FT-ChatML` model using the JOLA framework, optimized for A100 80GB GPU training.

## Project Structure

```
JOLA-LLM/
├── configs/                    # Configuration files
│   └── olmo2_13b_instruct_16k_config.yaml
├── scripts/                    # Training and utility scripts
│   ├── train_olmo2_13b.py     # Main training script
│   ├── config_utils.py         # Configuration utilities
│   └── setup_project.py        # Project setup script
├── data/                       # Dataset files
│   └── sharegpt_training_data_combined_final_solution_correctness_100.jsonl
├── outputs/                    # Training outputs
│   └── olmo2_13b_instruct_16k/
├── logs/                       # Training logs
├── checkpoints/                # Model checkpoints
├── jola/                       # JOLA framework code
├── examples/                   # Example scripts
└── requirements.txt            # Python dependencies
```

## Model Information

- **Model**: `embraceableAI/OLMo-2-1124-13B-Instruct-16k-Context-FT-ChatML`
- **Architecture**: OLMo2 (13B parameters)
- **Context Length**: 30,000 tokens (extended from 16k)
- **Format**: ChatML
- **Base Model**: OLMo-2-1124-13B

## Hardware Requirements

### Recommended Configuration (GPU)
- **GPU**: NVIDIA A100 80GB (or similar high-memory GPU)
- **System RAM**: 128GB+
- **Storage**: 100GB+ free space for model and checkpoints
- **CUDA**: 11.8+ with cuDNN

### CPU Configuration (Alternative)
- **CPU**: Modern multi-core processor (16+ cores recommended)
- **System RAM**: 64GB+ (32GB minimum)
- **Storage**: 50GB+ free space
- **Note**: Training will be 10-100x slower than GPU

### Memory Optimization Features
- **Precision**: bfloat16 (optimized for A100)
- **Flash Attention**: Enabled for memory efficiency
- **Gradient Checkpointing**: Enabled to reduce memory usage
- **Gradient Accumulation**: 4 steps (effective batch size = 16)
- **Extended Context**: 30,000 token sequence length support

## Quick Start for 13B Training

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Project Structure

```bash
python scripts/setup_project.py
```

### 3. Start Training

For GPU training (recommended):
```bash
python scripts/train_olmo2_13b.py
```

For CPU training (slower, for testing/compatibility):
```bash
python scripts/train_olmo2_13b_cpu.py
```

### 4. Monitor Training

Training progress is logged to Weights & Biases. Set up your W&B account:

```bash
wandb login
```

## Training Configuration

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.0001 | Optimized for 13B model |
| Batch Size | 4 | Per device batch size |
| Gradient Accumulation | 4 | Effective batch size = 16 |
| Epochs | 10 | Training duration |
| Warmup Steps | 100 | Learning rate warmup |
| Max Sequence Length | 30000 | Extended context window |
| Training Data | 5000 examples | ShareGPT dataset |
| Validation Data | 500 examples | Evaluation set |

### Memory Optimizations

- **bfloat16 Precision**: Reduces memory usage by 50% vs float32
- **Flash Attention**: Efficient attention computation
- **Gradient Checkpointing**: Trades compute for memory
- **Device Map**: Automatic model distribution across GPU memory

## Dataset

The training uses the ShareGPT dataset with the following structure:

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "System instruction..."
    },
    {
      "from": "human", 
      "value": "User question..."
    },
    {
      "from": "gpt",
      "value": "Assistant response..."
    }
  ]
}
```

- **Total Conversations**: 23,813
- **Training Samples**: 5,000 (configurable)
- **Validation Samples**: 500
- **Test Samples**: 500

## Training Scripts

### Main Training Script (`scripts/train_olmo2_13b.py`)

Features:
- Automatic GPU memory validation
- Weights & Biases integration
- Early stopping with patience
- Model checkpointing
- Comprehensive logging

Usage:
```bash
# Standard GPU training
python scripts/train_olmo2_13b.py

# GPU training without W&B logging
python scripts/train_olmo2_13b.py --no-wandb
```

### CPU Training Script (`scripts/train_olmo2_13b_cpu.py`)

Features:
- CPU-optimized configuration
- Reduced batch sizes and shorter sequences
- Memory-efficient settings
- Smaller dataset for testing

Usage:
```bash
# CPU training (much slower)
python scripts/train_olmo2_13b_cpu.py
```

**CPU Training Specifications:**
- Batch Size: 1 (with gradient accumulation of 8)
- Max Sequence Length: 1,024 tokens
- Training Data: 100 examples
- Validation Data: 20 examples
- Precision: float32
- No Flash Attention or mixed precision

### Configuration Utilities (`scripts/config_utils.py`)

Provides:
- YAML configuration loading
- GPU requirements validation
- W&B setup utilities
- Configuration summary printing

### Model Config Fix Utility (`scripts/fix_model_config.py`)

Fixes model architecture in config.json files after JOLA training:
- Converts "JoLAOlmo2Model" to "Olmo2ForCausalLM" for compatibility
- Can process single files or entire directories
- Automatically called by training script

Usage:
```bash
# Fix single config file
python scripts/fix_model_config.py path/to/config.json

# Fix all configs in directory
python scripts/fix_model_config.py path/to/model/directory

# Fix without recursive search
python scripts/fix_model_config.py path/to/directory --no-recursive
```

## Outputs

### Model Checkpoints
- **Location**: `outputs/olmo2_13b_instruct_16k/`
- **Format**: HuggingFace model format
- **Contents**: Model weights, tokenizer, configuration

### Final Model
- **Location**: `outputs/olmo2_13b_instruct_16k/final/`
- **Usage**: Ready for inference and deployment

### Logs
- **Training Logs**: `logs/` directory
- **W&B Dashboard**: Real-time metrics and plots
- **Console Output**: Progress and error messages

## Performance Monitoring

### Weights & Biases Metrics
- Training loss
- Validation loss
- Learning rate schedule
- GPU memory usage
- Training speed (tokens/second)

### Key Metrics to Monitor
- **Loss Convergence**: Should decrease steadily
- **Memory Usage**: Should stay within GPU limits
- **Training Speed**: Tokens processed per second
- **Validation Performance**: Prevents overfitting

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller sequence length

2. **Slow Training**
   - Check GPU utilization
   - Verify Flash Attention installation
   - Monitor data loading bottlenecks

3. **Model Not Loading**
   - Verify model name is correct
   - Check internet connection for model download
   - Ensure sufficient disk space

4. **Model Config Architecture Issue**
   - After JOLA training, config.json may show "JoLAOlmo2Model" instead of "Olmo2ForCausalLM"
   - This is automatically fixed by the training script
   - Manual fix: `python scripts/fix_model_config.py path/to/model/directory`

### Performance Tips

1. **Memory Optimization**
   - Use bfloat16 precision
   - Enable gradient checkpointing
   - Optimize batch size for your GPU

2. **Speed Optimization**
   - Use Flash Attention
   - Increase dataloader workers
   - Use SSD storage for data

3. **Stability**
   - Use gradient clipping
   - Implement early stopping
   - Monitor validation loss

## Advanced Configuration

### Custom Training Parameters

Edit the training script to modify:
- Learning rate schedule
- Batch sizes
- Training duration
- Evaluation frequency
- Model saving strategy

### Multi-GPU Training

For multi-GPU setups:
- Use `accelerate` for distributed training
- Adjust batch sizes per device
- Configure gradient accumulation

---

## Experiments in Paper

#### Dataset
We evaluate on commonsense reasoning, natural language understanding and natural language generation benchmarks.
+ Commonsense Reasoning
    - The same as [Hu et al., 2023](https://aclanthology.org/2023.emnlp-main.319/)
    - 8 Tasks: ARC-c / ARC-e / BoolQ / HellaSwag / OBQA / PIQA / SIQA / WinoGrande
    - [Download Link](https://github.com/AGI-Edgerunners/LLM-Adapters)
+ Natural Language Understanding
    - We use MMLU-Pro Benchmark ([Wang et al., 2024](https://arxiv.org/abs/2406.01574))
    - 14 Domains: Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, and Others
    - [Download Link](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
+ Natural Language Generation
    - We use [GEM Benchmark](https://gem-benchmark.com/) ([Gehrmann et al., 2022](https://arxiv.org/abs/2206.11249))
    - We use 4 tasks: CommonGen / E2E_NLG / Web_NLG / Xsum
    - [Download Link](https://huggingface.co/datasets/GEM/gem)
    - Prompt Template from [PromptSource](https://github.com/bigscience-workshop/promptsource)

#### Training and Evaluation
+ please use your own ```config.yaml```
+ ```python examples/run_jola.py```

## Citation

[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2502.01179-green?color=FF8000?color=009922)](https://doi.org/10.48550/arXiv.2502.01179)

Please cite our paper if it's helpful to your work!
```bibtex
@article{lai2025joint,
  title={Joint Localization and Activation Editing for Low-Resource Fine-Tuning},
  author={Lai, Wen and Fraser, Alexander and Titov, Ivan},
  journal={arXiv preprint arXiv:2502.01179},
  year={2025}
}
```

## License

This project uses the JOLA framework and OLMo2 model. Please refer to their respective licenses.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the JOLA documentation
3. Monitor GPU memory usage
4. Check training logs for errors
