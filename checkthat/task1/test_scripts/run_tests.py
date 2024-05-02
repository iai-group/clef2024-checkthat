import torch
import pandas as pd
from models.custom_model import CustomModel
from tokenization.tokenizer import TextDataset
import wandb
from metrics.compute_metrics import compute_metrics


import os
from transformers import AutoModelForSequenceClassification
import torch

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in the given directory."""
    checkpoint_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and 'checkpoint' in d]
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found in the given model directory.")
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
    return latest_checkpoint

def find_models_with_checkpoints(base_dir):
    """Find models and their latest checkpoints in a structured directory.
    Args:
        base_dir (str): Directory containing subdirectories of models
    Returns:
        model_info (list): List of tuples containing model name and path to the latest checkpoint.
    """
    model_info = []
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        
        try:
            latest_checkpoint = find_latest_checkpoint(model_path)
            model_info.append((model_name, latest_checkpoint))
            print(f"Model: {model_name}, Latest Checkpoint: {latest_checkpoint}")
        except Exception as e:
            print(f"Failed to find checkpoint for model {model_name}: {str(e)}")

    return model_info



def run_prediction(model_name, dataset, tokenizer, model_named_trained, has_labels: bool):
    """Run prediction on the dataset, compute metrics if labels are present, and write results to a .tsv file."""
    
    device = 'cuda'
    label_map = {0: 'no', 1: 'yes'}  # Ensure this mapping is correct for your model

    # Initialize Weights & Biases
    run_name = f"TEST__{model_named_trained}"
    wandb.init(project="Clef2024", entity="aarnes", name=run_name)

    model = CustomModel(model_name=model_name, num_labels=len(label_map), device=device)
    model.eval()

    # Load the dataset, with or without labels
    test_dataset = TextDataset(dataset["test"], tokenizer, None if not has_labels else label_map)
    all_logits = []
    all_labels = []
    results = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataset):
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            predictions = logits.argmax(-1).cpu().numpy()

            # Collect logits and labels for metric calculation if labels are present
            if has_labels and 'labels' in batch:
                labels = batch['labels'].cpu().numpy()
                all_logits.append(logits)
                all_labels.append(torch.tensor(labels))
                for label, pred in zip(labels, predictions):
                    results.append((i, label_map[pred], model_named_trained))
            else:
                for pred in predictions:
                    results.append((i, label_map[pred], model_named_trained))

    # If labels were present, calculate metrics
    if has_labels:
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        predictions = all_logits.argmax(-1)
        metrics = compute_metrics((predictions, all_labels))
        wandb.log(metrics)

    # Save results to a .tsv file
    df = pd.DataFrame(results, columns=['sentence_id', 'prediction', 'model_name'])
    df.to_csv(f"{model_named_trained}_predictions.tsv", sep='\t', index=False)

    # Finish Weights & Biases logging
    wandb.finish()
