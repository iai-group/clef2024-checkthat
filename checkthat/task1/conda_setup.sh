#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100
#SBATCH --time=1:00:00
#SBATCH --job-name=conda_setup
#SBATCH --output=conda_setup.out

module load cuda/12.2.0 cudnn/8.8.0 # Load CUDA and cuDNN modules

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Create and activate the Conda environment
conda create -n CLEF_checkthat2024 -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.1 -c nvidia -y
conda activate CLEF_checkthat2024

# Install Python packages
pip install torch torchvision torchaudio
pip install transformers[torch]
pip install -r requirements.txt