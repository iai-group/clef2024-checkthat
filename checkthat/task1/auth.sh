#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100
#SBATCH --time=1:00:00
#SBATCH --job-name=setup_authenticator.sh
#SBATCH --output=hf_test.out
# Load necessary modules, if required
# module load python/3.9  # Adjust this according to your environment

# Activate your Python environment
# source ~/bhome/env/checkthat2024_env/bin/activate

# Explicitly specify the path to the correct Python executable
# PYTHON="~/bhome/env/checkthat2024_env/bin/"
PYTHON="~/.conda/envs/CLEF_checkthat2024/bin"
uenv miniconda3-py39

# Activate the Conda environment
#conda activate ~/bhome/env/checkthat2024_env
conda activate ~/.conda/envs/CLEF_checkthat2024

export HF_HOME=~/bhome/clef2024-checkthat/checkthat/task1
# Create necessary directories
mkdir -p $HF_HOME $WANDB_CACHE_DIR

# Store the Hugging Face token
echo 'KEY' > $HF_HOME/token
chmod 600 $HF_HOME/token

# Log in to wandb
export WANDB_API_KEY='KEY'
export WANDB_CACHE_DIR=~/bhome/clef2024-checkthat/checkthat/task1
wandb login KEY

# Test the Hugging Face API with a Python script
#$PYTHON test_start.py
python -u test_hf_login.py
