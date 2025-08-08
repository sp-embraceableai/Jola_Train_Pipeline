# OLMo-2 Integration with JoLA

This document describes the integration of the OLMo-2 model architecture with the JoLA (Joint Localization and Activation Editing) framework.

## Overview

The OLMo-2 integration extends JoLA's parameter-efficient fine-tuning capabilities to OLMo-2 models, including the [OLMo-2-1124-13B-Instruct-32k-Context-ChatML](https://huggingface.co/sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML) model.

## Model Architecture

OLMo-2 is a decoder-only transformer model with the following key characteristics:
- **Architecture**: Decoder-only transformer
- **Attention**: Multi-head attention with RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm
- **Activation**: SiLU (Swish)
- **Position Embeddings**: RoPE with configurable theta

### Key Model Parameters (OLMo-2-1124-13B-Instruct-32k-Context-ChatML)
```json
{
  "hidden_size": 5120,
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "intermediate_size": 13824,
  "max_position_embeddings": 32768,
  "vocab_size": 100352,
  "rope_theta": 4134231.132028111
}
```

## JoLA Integration

### Core Components

1. **JoLAOlmo2Model**: Main model class that extends OLMo-2 with JoLA functionality
2. **Olmo2Attention**: Modified attention layer with JoLA parameters
3. **Gate Mechanism**: Learnable gates for selective activation editing

### JoLA Parameters

For each attention head, JoLA adds:
- **attn_A**: Scaling factors (head_dim parameters)
- **attn_v**: Bias vectors (head_dim parameters)  
- **log_g1**: Gate parameters for additive interventions
- **log_g2**: Gate parameters for multiplicative interventions

### Parameter Count

For the OLMo-2-1124-13B-Instruct-32k-Context-ChatML model:
- **Total JoLA parameters**: 40 layers × 40 heads × (128 + 128 + 1 + 1) = 1,040,000 parameters
- **Memory efficient**: Only ~4MB of additional parameters vs full fine-tuning

## Usage

### Basic Usage

```python
from jola import JoLAOlmo2Model, JoLAConfig, JoLATrainer
from transformers import AutoTokenizer

# Load configuration
config = JoLAConfig.get_jola_config(default=False, config_path="jola/config_olmo2.yaml")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML")
model = JoLAOlmo2Model.jola_from_pretrained(**config["model_config"])

# Unfreeze JoLA parameters
model.unfreeze_jola_params()

# Setup training
trainer = JoLATrainer(
    model=model,
    # ... other training arguments
    gate_scheduler="expon"  # or "linear", "cyclic", "perform"
)

trainer.train()
```

### Configuration

The OLMo-2 specific configuration is in `jola/config_olmo2.yaml`:

```yaml
model_config:
  pretrained_model_name_or_path: "sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML"
  applied_module: 'attention'
  device: "cuda"

training_config:
  learning_rate: 0.005
  per_device_train_batch_size: 4  # Reduced for 13B model
  bf16: True  # Recommended for OLMo-2

jola_config:
  gate_scheduler: "expon"
  gate_lambda: 0.00004
```

## Gate Scheduling Strategies

1. **Linear**: Gradually increases gate lambda from 0.0 to 0.2
2. **Cyclic**: Oscillates gate values in sinusoidal pattern
3. **Performance-based**: Adjusts based on performance improvement
4. **Exponential decay**: Gradually decreases gate values (recommended for OLMo-2)

## Training Considerations

### Memory Optimization
- Use `bf16=True` for reduced memory usage
- Smaller batch sizes (4-8) recommended for 13B model
- Gradient checkpointing available if needed

### Hyperparameter Tuning
- **Learning rate**: 0.005 (default), may need adjustment based on dataset
- **Gate lambda**: 0.00004 (default), controls regularization strength
- **Temperature**: 0.33 (default), controls gate sampling sharpness

### Data Format
OLMo-2 uses ChatML format. Ensure your data follows this pattern:
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
{assistant_response}
<|im_end|>
```

## Example Scripts

### Quick Start
```bash
python examples/run_jola_olmo2.py
```

### Testing Integration
```bash
python test_olmo2_integration.py
```

## Performance

JoLA with OLMo-2 provides:
- **Parameter efficiency**: Only ~1M trainable parameters vs full fine-tuning
- **Low-resource optimization**: Effective with few hundred examples
- **Selective intervention**: Learns which attention heads to modify
- **Adaptive gates**: Automatically determines intervention type and strength

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Tokenizer Issues**: Ensure proper ChatML formatting
3. **Gate Convergence**: Try different gate scheduling strategies
4. **Model Loading**: Verify transformers version compatibility

### Debug Mode
Enable anomaly detection for debugging:
```python
torch.autograd.set_detect_anomaly(True)
```

## References

- [JoLA Paper](http://arxiv.org/abs/2502.01179)
- [OLMo-2 Model](https://huggingface.co/sp-embraceable/OLMo-2-1124-13B-Instruct-32k-Context-ChatML)
- [OLMo-2 Architecture](https://arxiv.org/abs/2402.00810)
