import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from task1.models.custom_model import CustomModel
from task1.tokenization.tokenizer import TextDataset
from task1.metrics.compute_metrics import compute_metrics
import wandb
from tokenization.tokenizer import TextDataset

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in the given directory."""
    checkpoint_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and 'checkpoint' in d]
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found in the given model directory.")
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
    return latest_checkpoint

def find_models_with_checkpoints(base_dir):
    """Find models and their latest checkpoints in a structured directory."""
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

def run_prediction(model_name, dataset_list, tokenizer, model_path, has_labels: bool):
    """Run prediction on the dataset, compute metrics if labels are present, and write results to a .tsv file."""
    device = 'cuda'
    label_map = {0: 'no', 1: 'yes'}
    
    # Detect language from model name and select dataset
    lang = model_name.split('_')[-2]  # Assumes format like 'modelname_lang_'
    dataset = dataset_list[lang]

    # Initialize Weights & Biases
    run_name = f"TEST__{model_path}"
    wandb.init(project="Clef2024", entity="aarnes", name=run_name)

    # Load the model from the checkpoint
    model = CustomModel.from_pretrained(model_path)
    model.to(device)
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
                    results.append((i, label_map[pred], model_name))
            else:
                for pred in predictions:
                    results.append((i, label_map[pred], model_name))

    # If labels were present, calculate metrics
    if has_labels:
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        predictions = all_logits.argmax(-1)
        metrics = compute_metrics((predictions, all_labels))
        wandb.log(metrics)

    # Save results to a .tsv file
    df = pd.DataFrame(results, columns=['sentence_id', 'prediction', 'model_name'])
    df.to_csv(f"{model_path}_predictions.tsv", sep='\t', index=False)

    # Finish Weights & Biases logging
    wandb.finish()


if __name__ == "__main__":
    # Define the dataset list for each language
    dataset_list = {
        "en":"iai-group/clef2024_checkthat_task1_en",
        "ar":"iai-group/clef2024_checkthat_task1_ar",
        "es":"iai-group/clef2024_checkthat_task1_es",
        "nl":"iai-group/clef2024_checkthat_task1_nl",
    }

    label_map = {"Yes": 1, "No": 0}


    # Load models and run prediction
    base_dir = "./trained_models"
    i = 0

    model_info = find_models_with_checkpoints(base_dir)
    for model_name, checkpoint_path in model_info:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)  # General tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)  # Model for prediction
        tokenized_data = TextDataset(dataset_list.values()[i], tokenizer, label_map)
        run_prediction(model_name, dataset_list, tokenizer, checkpoint_path, has_labels=True)
        i += 1
        
    
