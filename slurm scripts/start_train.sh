#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=gpuA100
#SBATCH --time=24:00:00
#SBATCH --job-name=checkthat_training
#SBATCH --output=checkthat_training.out

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

PROJECT_NAME="EN-SWEEP-no-data-alter-main" 

run_sweep_and_agent () {
  # Ensure the PROJECT_NAME environment variable is set
  if [[ -z "$PROJECT_NAME" ]]; then
    echo "Error: PROJECT_NAME must be set."
    return 1
  fi
  
  echo "Initializing sweep using sweep.yaml in project: $PROJECT_NAME..."
  
  # Run the wandb sweep command using a fixed file path
  wandb sweep --project "$PROJECT_NAME" "sweep.yaml" > temp_output.txt 2>&1
  
  # Check if the wandb sweep command succeeded
  if [ $? -ne 0 ]; then
    echo "Error: Failed to initialize sweep. See output below:"
    cat temp_output.txt
    return 1
  fi

  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)
  
  # Check if the sweep ID was extracted successfully
  if [[ -z "$SWEEP_ID" ]]; then
    echo "Error: Failed to extract sweep ID from output."
    cat temp_output.txt
    return 1
  fi

  # Cleanup: Remove the temporary output file
  rm temp_output.txt
  
  # Run the wandb agent command
  echo "Starting wandb agent for sweep ID: $SWEEP_ID"
  wandb agent $SWEEP_ID
}

run_sweep_and_agent = "" # Create name and run the sweep and agent function