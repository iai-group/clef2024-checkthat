#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=gpuA100 
#SBATCH --time=1:00:00
#SBATCH --job-name=CLEF2024_task1_training
#SBATCH --output=start_train_all.out
 
# Activate environment
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39
conda activate transformer_cuda12
PATH=~/.local/bin:$PATH
echo $PATH
# Run the Python script that uses the GPU
TOKENIZERS_PARALLELISM=false python -u main.py