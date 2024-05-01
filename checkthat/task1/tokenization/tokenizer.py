"""Tokenizer for the task1 datasets."""
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Takes a list of dictionaries containing text and class labels.

    Args:
        Dataset: Dataset class from torch.utils.data
    """

    def __init__(self, data, tokenizer, label_map):
        """Initialize the TextDataset class."""
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Tokenize the text and return a dictionary containing the
        tokenized."""
        item = self.data[idx]
        encoded = self.tokenizer.encode_plus(
            item["Text"],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        label_id = self.label_map[item["class_label"]]
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id),
        }
