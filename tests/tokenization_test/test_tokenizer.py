import pytest
from checkthat.task1.tokenization.tokenizer import TextDataset
from transformers import AutoTokenizer


def test_text_dataset_length():
    """Test the length of the TextDataset.

    The length of the dataset should be equal to the number of data
    samples.
    """
    data = [{"Text": "Example text", "class_label": "Yes"}]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    label_map = {"Yes": 1}

    dataset = TextDataset(data, tokenizer, label_map)
    assert len(dataset) == 1


def test_text_dataset_getitem():
    """Test the __getitem__ method of the TextDataset.

    The __getitem__ method should return a dictionary with the keys
    'input_ids', 'attention_mask', and 'labels'.
    """
    data = [{"Text": "Example text", "class_label": "Yes"}]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    label_map = {"Yes": 1}

    dataset = TextDataset(data, tokenizer, label_map)
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
