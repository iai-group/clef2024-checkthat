"""Module for tokenizing text data."""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Takes a list of dictionaries containing text and labels."""

    def __init__(self, data, tokenizer, label_map=None):
        """Initialize the TextDataset class."""
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Tokenize the text and return a dictionary data with tokenization."""
        item = self.data[idx]
        encoded = self.tokenizer.encode_plus(
            item["Text"],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if "class_label" in item and self.label_map is not None:
            label_id = self.label_map[item["class_label"]]
            result["labels"] = torch.tensor(label_id)

        return result
