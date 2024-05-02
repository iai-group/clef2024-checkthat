import os
from transformers import AutoModelForSequenceClassification
import torch

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in the given directory."""
    checkpoint_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and 'checkpoint' in d]
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found in the given model directory.")

    # Sort directories to find the one with the highest step (assuming naming convention includes "checkpoint-<step>")
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
    return latest_checkpoint

def load_model_from_dir(base_dir):
    """Load models from a structured directory of models.
    Args:
        base_dir (str): Directory containing subdirectories of models named like 'FacebookAI/xlm-roberta-base_10_en'
    Returns:
        models (dict): Dictionary with keys as model names and values as loaded model objects.
    """
    models = {}
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        
        try:
            latest_checkpoint = find_latest_checkpoint(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint)
            models[model_name] = model
            print(f"Loaded model from {latest_checkpoint}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {str(e)}")

    return models

    