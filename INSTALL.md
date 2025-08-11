# Installation Guide

This guide provides multiple ways to install and use the JOLA Training Pipeline.

## Method 1: Direct Installation from GitHub (Recommended)

### For GPU Training:
```bash
# Install with GPU support
pip install "git+https://github.com/sp-embraceableai/Jola_Train_Pipeline.git[gpu]"
```

### For CPU Training:
```bash
# Install without GPU dependencies
pip install "git+https://github.com/sp-embraceableai/Jola_Train_Pipeline.git"
```

### For Development:
```bash
# Install with development tools
pip install "git+https://github.com/sp-embraceableai/Jola_Train_Pipeline.git[dev]"
```

## Method 2: Clone and Install

```bash
# Clone the repository
git clone https://github.com/sp-embraceableai/Jola_Train_Pipeline.git
cd Jola_Train_Pipeline

# Install in development mode
pip install -e .

# Or install with GPU support
pip install -e ".[gpu]"
```

## Method 3: Manual Setup

```bash
# Clone the repository
git clone https://github.com/sp-embraceableai/Jola_Train_Pipeline.git
cd Jola_Train_Pipeline

# Install dependencies
pip install -r requirements.txt

# Run scripts directly
python scripts/train_olmo2_13b.py
```

## Verify Installation

After installation, you can verify it works by running:

```bash
# Check if command-line tools are available
jola-train --help
jola-train-cpu --help
jola-fix-config --help
```

Or test the Python imports:

```python
import scripts.train_olmo2_13b
import scripts.fix_model_config
print("Installation successful!")
```

## Requirements

### Minimum Requirements:
- Python 3.8+
- 32GB RAM (for CPU training)
- 50GB storage

### Recommended for GPU Training:
- Python 3.9+
- NVIDIA A100 80GB GPU
- 128GB RAM
- 100GB storage
- CUDA 11.8+

## Troubleshooting

### Common Issues:

1. **ImportError: No module named 'jola'**
   ```bash
   pip install jola
   ```

2. **CUDA not available**
   ```bash
   # Use CPU training instead
   jola-train-cpu
   ```

3. **Out of memory errors**
   ```bash
   # Reduce batch size or use CPU training
   # Edit configs/olmo2_13b_cpu_config.yaml
   ```

4. **Permission errors on Windows**
   ```bash
   # Run as administrator or use virtual environment
   python -m venv venv
   venv\Scripts\activate
   pip install "git+https://github.com/sp-embraceableai/Jola_Train_Pipeline.git"
   ```

## Next Steps

After installation:

1. **Setup the project structure:**
   ```bash
   python scripts/setup_project.py
   ```

2. **Start training:**
   ```bash
   # GPU training
   jola-train
   
   # CPU training  
   jola-train-cpu
   ```

3. **Monitor training:**
   ```bash
   # Setup Weights & Biases
   wandb login
   ```

For more details, see the main [README.md](README.md) file.
