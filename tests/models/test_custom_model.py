"""Tests for the CustomModel class."""
import pytest
from unittest.mock import patch
from transformers import AutoModelForSequenceClassification
import torch
from checkthat.task1.models.custom_model import CustomModel


def test_custom_model_initialization():
    """Test for the initialization of the CustomModel class.

    The model should be initialized with the correct model name and
    number of labels.
    """
    model_name = "bert-base-uncased"
    num_labels = 2
    with patch.object(
        AutoModelForSequenceClassification, "from_pretrained", return_value=None
    ) as mock_method:
        model = CustomModel(model_name, num_labels)
        mock_method.assert_called_once_with(model_name, num_labels=num_labels)


def test_custom_model_forward():
    """Test for the forward method of the CustomModel class.

    The forward method should return a dictionary with the key 'loss'
    when labels are provided.
    """
    model_name = "bert-base-uncased"
    num_labels = 2
    model = CustomModel(model_name, num_labels)
    input_ids = torch.randint(0, 1000, (1, 10))
    attention_mask = torch.ones(1, 10)
    labels = torch.tensor([1])

    with patch.object(
        AutoModelForSequenceClassification, "from_pretrained", return_value=None
    ):
        output = model.forward(input_ids, attention_mask, labels)
        assert "loss" in output.keys()
