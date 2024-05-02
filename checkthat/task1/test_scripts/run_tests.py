import torch
import wandb
from tokenization.tokenizer import TextDataset
from models.custom_model import CustomModel
from metrics.compute_metrics import compute_metrics

def run_testing(model_name, dataset, tokenizer, label_map, model_named_trained):
    """Run testing on the given model and dataset."""
     
    run_name = f"TEST__{model_named_trained}"
    wandb.init(project="Clef2024", entity="aarnes", name=run_name)

    # Assuming TextDataset provides the input in the correct format
    test_dataset = TextDataset(dataset["test"], tokenizer, label_map)
    model = CustomModel(model_name=model_name, num_labels=len(label_map), device='cuda')
    model.eval()

    logits = []
    labels = []

    with torch.no_grad():
        for batch in test_dataset:
            input_ids, attention_mask, label = batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda'), batch['labels'].to('cuda')
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits.append(output.logits)  # Adjust according to how outputs are structured
            labels.append(label)

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        predictions = logits.argmax(-1)
        metrics = compute_metrics((predictions, labels))
        wandb.log(metrics)
        wandb.finish()

