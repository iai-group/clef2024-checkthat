import sys
from unittest.mock import MagicMock, patch

import sys
# Clean up the sys.path to remove incorrect entries
sys.path = [p for p in sys.path if not p.endswith('\n')]

# Optionally, append the correct path explicitly if it's not already included
correct_path = '/Users/aarnes/Desktop/github_clones/clef2024-checkthat'
if correct_path not in sys.path:
    sys.path.insert(0, correct_path)

import sys
sys.path.insert(0, '/Users/aarnes/Desktop/github_clones/clef2024-checkthat/checkthat')

print()
print(sys.path)
print()
import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from checkthat.task1.predict_test import main


@pytest.fixture
def mock_environment():
    with patch("checkthat.task1.predict_test.load_dataset") as mock_load_dataset, \
         patch("checkthat.task1.predict_test.AutoTokenizer.from_pretrained") as mock_from_pretrained, \
         patch("checkthat.task1.predict_test.AutoModelForSequenceClassification.from_pretrained") as mock_model_pretrained, \
         patch("checkthat.task1.predict_test.TextDataset") as mock_text_dataset, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.device"), \
         patch("torch.utils.data.DataLoader") as mock_dataloader, \
         patch("checkthat.task1.predict_test.rename_features") as mock_rename_features:
        
        # Setup mock for dataset loading
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {'column_names': ['tweet_text']}
        mock_load_dataset.return_value = {"test": mock_dataset}

        # Mock model and tokenizer
        tokenizer_instance = MagicMock(spec=AutoTokenizer)
        model_instance = MagicMock(spec=AutoModelForSequenceClassification)
        mock_from_pretrained.return_value = tokenizer_instance
        mock_model_pretrained.return_value = model_instance

        # Mock TextDataset
        mock_dataset_instance = MagicMock()
        mock_text_dataset.return_value = mock_dataset_instance

        # Mock DataLoaderéé
        mock_dataloader.return_value = iter([{'input_ids': torch.tensor([1]), 'labels': torch.tensor([1])}])

        # Setup mock for renaming features if needed
        mock_rename_features.return_value = mock_dataset

        yield

def test_main_runs_through(mock_environment):
    # Execute the main function
    main()

    # Here, you would typically add assertions to verify that the correct calls were made,
    # For instance, check if models and tokenizers were loaded with the correct arguments,
    # or if the DataLoader was called correctly. Since `main` does not return anything,
    # you are limited to checking side effects and interactions.
    # Examples:
    # mock_from_pretrained.assert_called_with('path_to_tokenizer')
    # mock_model_pretrained.assert_called_with('path_to_model', num_labels=2)