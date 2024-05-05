#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=1:00:00
#SBATCH --job-name=CLEF2024_task1_training
#SBATCH --output=start_train_all.out

# Load CUDA and cuDNN modules
module load cuda/12.2.0 cudnn/8.8.0

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
#conda activate ~/bhome/env/checkthat2024_env

conda ~/.conda/envs/CLEF_checkthat2024


# Add user's local bin directory to the PATH
PATH=~/.local/bin:$PATH
echo $PATH

# Disable tokenizers parallelism for better GPU utilization
export TOKENIZERS_PARALLELISM=false

# Run the Python script that uses the GPU
python -u main_train_all.py