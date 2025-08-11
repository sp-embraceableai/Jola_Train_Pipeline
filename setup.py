#!/usr/bin/env python3
"""
Setup script for JOLA Training Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="jola-train-pipeline",
    version="1.0.0",
    author="sp-embraceable",
    author_email="contact@embraceable.ai",
    description="JOLA Training Pipeline for OLMo2 13B model with A100 GPU optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sp-embraceableai/Jola_Train_Pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
        "gpu": [
            "flash-attn>=2.3.0",
            "xformers>=0.0.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "jola-train=scripts.train_olmo2_13b:main",
            "jola-train-cpu=scripts.train_olmo2_13b_cpu:main",
            "jola-fix-config=scripts.fix_model_config:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "deep learning",
        "transformers",
        "language models",
        "fine-tuning",
        "jola",
        "olmo2",
        "parameter efficient",
        "low resource",
    ],
    project_urls={
        "Bug Reports": "https://github.com/sp-embraceableai/Jola_Train_Pipeline/issues",
        "Source": "https://github.com/sp-embraceableai/Jola_Train_Pipeline",
        "Documentation": "https://github.com/sp-embraceableai/Jola_Train_Pipeline#readme",
    },
)
