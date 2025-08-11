"""
JOLA Training Pipeline

A comprehensive training pipeline for the OLMo2 13B model using the JOLA framework,
optimized for A100 80GB GPU training with extended 30k token context support.

This package provides:
- Training scripts for GPU and CPU
- Configuration management utilities
- Model configuration fixing tools
- Comprehensive documentation and examples

Example usage:
    >>> from scripts.train_olmo2_13b import main as train_gpu
    >>> from scripts.train_olmo2_13b_cpu import main as train_cpu
    >>> from scripts.fix_model_config import main as fix_config
"""

__version__ = "1.0.0"
__author__ = "sp-embraceable"
__email__ = "contact@embraceable.ai"
__license__ = "MIT"
__url__ = "https://github.com/sp-embraceableai/Jola_Train_Pipeline"

# Package metadata
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "__url__",
]
