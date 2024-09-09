from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tokenization.normalize_DatasetDict_featues import rename_features
from tokenization.tokenizer import TextDataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def test_predict() -> None:
    # """Run training."""
    label_map = {"Yes": 1, "No": 0}

    # Load tokenizer

    dataset_name = "iai-group/clef2024_checkthat_task1_en"  # Replace this with the name of your dataset
    dataset = load_dataset(dataset_name)
    if "tweet_text" in dataset["test"].column_names:
        dataset = rename_features(dataset)

    """Saved model and tokenizer paths."""
    saved_model_path = (
        "FacebookAI/roberta-large"  # Replace this with the path to your model
    )
    saved_tokenizer_path = "FacebookAI/roberta-large"  # Replace this with the path to your tokenizer
    tokenizer_dataset = AutoTokenizer.from_pretrained(saved_tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(saved_tokenizer_path)

    test_dataset = TextDataset(dataset["test"], tokenizer, label_map)

    hf_model = AutoModelForSequenceClassification.from_pretrained(
        saved_model_path, num_labels=len(label_map)
    )

    # Assuming you're using a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model.to(device)
    hf_model.eval()

    # Assuming TextDataset is properly returning tokenized outputs and compatible with DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer_dataset)

    # Creating a DataLoader for batch processing
    test_loader = DataLoader(
        test_dataset, batch_size=16, collate_fn=data_collator
    )

    # Initialize lists to store predictions and labels
    predictions = []
    labels = []

    for batch in test_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = hf_model(**batch)
            logits = outputs.logits
            pred_labels = torch.argmax(logits, dim=-1)
            predictions.extend(pred_labels.cpu().numpy())
            labels.extend(
                batch["labels"].cpu().numpy()
            )  # assuming labels are part of your batch

    # Calculate metrics

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    acc = accuracy_score(labels, predictions)

    print(
        f"Accuracy: {acc}, F1: {f1}, Precision: {precision}, Recall: {recall}"
    )
