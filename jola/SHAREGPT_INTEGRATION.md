# ShareGPT Integration with JoLA

This document describes the integration of ShareGPT-style datasets with the JoLA framework for training OLMo-2 models.

## Overview

ShareGPT datasets contain conversation-style data in JSONL format, making them ideal for training conversational AI models. The integration supports both ChatML and instruction-response formats.

## ShareGPT Dataset Format

### Standard ShareGPT Format
```json
{
  "conversations": [
    {"from": "human", "value": "What is the capital of France?"},
    {"from": "gpt", "value": "The capital of France is Paris."}
  ]
}
```

### Alternative Formats
The loader also supports:
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

## Dataset Structure

### File Organization
```
sharegpt_training_data_combined_final_solution_correctness_100.jsonl
```

Each line contains a JSON object with conversation data.

### Data Processing

1. **Loading**: Reads JSONL file line by line
2. **Parsing**: Extracts conversation pairs (user → assistant)
3. **Formatting**: Converts to ChatML or instruction format
4. **Splitting**: Creates train/valid/test splits
5. **Sampling**: Applies size limits (2000 train, 200 valid, 200 test)

## Usage

### Basic Usage

```python
from jola import ShareGPTDataset, JoLAOlmo2Model, JoLATrainer

# Load ShareGPT dataset
dataset = ShareGPTDataset(
    data_path="sharegpt_training_data_combined_final_solution_correctness_100.jsonl",
    train_size=2000,
    valid_size=200,
    test_size=200,
    format_type="chatml"  # or "instruction"
)

data = dataset.load_data()
```

### With JoLA Training

```python
from jola import JoLAConfig, JoLAOlmo2Model, JoLATrainer, ShareGPTDataset

# Load configuration
config = JoLAConfig.get_jola_config(default=False, config_path="jola/config_sharegpt.yaml")

# Load model and tokenizer
model = JoLAOlmo2Model.jola_from_pretrained(**config["model_config"])
model.unfreeze_jola_params()

# Load ShareGPT dataset
dataset = ShareGPTDataset(
    data_path=config["data_config"]["data_path"],
    train_size=config["data_config"]["train_size"],
    valid_size=config["data_config"]["valid_size"],
    test_size=config["data_config"]["test_size"],
    format_type=config["data_config"]["format_type"]
)
data = dataset.load_data()

# Setup training
trainer = JoLATrainer(
    model=model,
    train_dataset=data['train'],
    eval_dataset=data['valid'],
    # ... other arguments
)

trainer.train()
```

## Configuration

### ShareGPT Configuration (`config_sharegpt.yaml`)

```yaml
model_config:
  pretrained_model_name_or_path: "sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML"
  applied_module: 'attention'

training_config:
  learning_rate: 0.003
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  bf16: True

data_config:
  train_size: 2000
  valid_size: 200
  test_size: 200
  format_type: "chatml"
  data_path: "sharegpt_training_data_combined_final_solution_correctness_100.jsonl"

jola_config:
  gate_scheduler: "expon"
  gate_lambda: 0.00004
```

## Data Formatting

### ChatML Format (Recommended for OLMo-2)
```
<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
The capital of France is Paris.
<|im_end|>
```

### Instruction Format
```
### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
```

## Features

### Automatic Processing
- **JSONL Support**: Handles large JSONL files efficiently
- **Format Detection**: Automatically detects ShareGPT format variations
- **Conversation Parsing**: Extracts user-assistant pairs from conversations
- **Smart Sampling**: Random sampling with configurable sizes

### Split Management
- **Train/Valid/Test**: Automatic split creation (80%/10%/10% by default)
- **Size Limits**: Configurable sample sizes for each split
- **Random Shuffling**: Ensures representative splits

### Error Handling
- **Robust Parsing**: Handles malformed JSON lines gracefully
- **Content Validation**: Filters out empty or invalid conversations
- **Encoding Support**: Handles UTF-8 encoding properly

## Example Scripts

### Quick Start
```bash
python examples/run_jola_sharegpt.py
```

### Testing Integration
```bash
python test_sharegpt_integration.py
```

## Performance Considerations

### Memory Efficiency
- **Streaming**: Processes large files without loading everything into memory
- **Batch Processing**: Efficient handling of large datasets
- **Sampling**: Reduces memory usage for large datasets

### Training Optimization
- **Gradient Accumulation**: Handles large effective batch sizes
- **Mixed Precision**: BF16 support for memory efficiency
- **Early Stopping**: Prevents overfitting

## Dataset Statistics

For the ShareGPT dataset with 2000 training samples:
- **Training Examples**: 2000 conversation pairs
- **Validation Examples**: 200 conversation pairs  
- **Test Examples**: 200 conversation pairs
- **Total Parameters**: ~1M JoLA parameters
- **Memory Usage**: ~4MB additional parameters

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure the JSONL file path is correct
2. **Empty Dataset**: Check if the file contains valid JSON lines
3. **Memory Issues**: Reduce batch size or use gradient accumulation
4. **Format Errors**: Verify the ShareGPT format is correct

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Test dataset loading
dataset = ShareGPTDataset(data_path="your_file.jsonl", train_size=10)
data = dataset.load_data()
print(f"Loaded {len(data['train'])} training examples")
```

## Integration with OLMo-2

The ShareGPT integration is specifically optimized for OLMo-2 models:
- **ChatML Format**: Native support for OLMo-2's conversation format
- **32k Context**: Handles long conversations efficiently
- **Parameter Efficiency**: Only ~1M trainable parameters
- **Low-Resource**: Effective with 2000 training examples

## References

- [JoLA Paper](http://arxiv.org/abs/2502.01179)
- [OLMo-2 Model](https://huggingface.co/sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML)
- [ShareGPT Dataset Format](https://github.com/dome272/ShareGPT)
