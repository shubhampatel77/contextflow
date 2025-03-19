#!/bin/bash

# Create contextflow environment
conda create -n contextflow python=3.10 -y
conda activate contextflow

# Install PyTorch with specific CUDA version
pip3 install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements except flash-attention
pip install -r requirements.txt

# Install flash-attention with correct CUDA version
pip install ninja
pip install flash-attn --no-build-isolation