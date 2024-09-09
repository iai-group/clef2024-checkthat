import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import the main function from your script
from checkthat.task1.predict.predict_unlabeled import main

@pytest.fixture
def mock_data():
    return pd.DataFrame({
        'Text': ['This is a test sentence.', 'Another test sentence.'],
        'Sentence_id': [1, 2]
    })

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    return tokenizer

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    model.return_value = mock_outputs
    return model

@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=Dataset)
    dataset.map.return_value = dataset
    dataset.remove_columns.return_value = dataset
    dataset.set_format.return_value = None
    return dataset

@pytest.fixture
def mock_dataloader():
    dataloader = MagicMock()
    dataloader.__iter__.return_value = [
        {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'Sentence_id': torch.tensor([1])
        },
        {
            'input_ids': torch.tensor([[4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'Sentence_id': torch.tensor([2])
        }
    ]
    return dataloader

def test_main(mock_data, mock_tokenizer, mock_model, mock_dataset, mock_dataloader):
    with patch('pandas.read_csv', return_value=mock_data), \
         patch('datasets.Dataset.from_pandas', return_value=mock_dataset), \
         patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
         patch('torch.device', return_value='cpu'), \
         patch('torch.utils.data.DataLoader', return_value=mock_dataloader), \
         patch('builtins.open', MagicMock()), \
         patch('torch.argmax', return_value=torch.tensor([0, 1])):

        # Run the main function
        main()

        # Assert that the model was called
        mock_model.assert_called()

        # Check if the model's eval method was called
        mock_model.eval.assert_called_once()

        # Check if the tokenizer was called
        mock_tokenizer.assert_called()

        # Check if the output file was attempted to be written
        open.assert_called_with('unlabeled_test_results.tsv', 'w')

if __name__ == "__main__":
    pytest.main()