import pytest
from unittest.mock import patch, MagicMock
import torch
from checkthat.task1.predict.predict_test import test_predict

@pytest.fixture
def mock_dependencies():
    with patch('datasets.load_dataset') as mock_load_dataset, \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
         patch('torch.device') as mock_device, \
         patch('torch.cuda.is_available') as mock_cuda_available, \
         patch('torch.utils.data.DataLoader') as mock_dataloader, \
         patch('tokenization.normalize_DatasetDict_featues.rename_features') as mock_rename_features, \
         patch('tokenization.tokenizer.TextDataset') as mock_text_dataset, \
         patch('transformers.DataCollatorWithPadding') as mock_data_collator, \
         patch('torch.no_grad') as mock_no_grad, \
         patch('sklearn.metrics.precision_recall_fscore_support') as mock_prf, \
         patch('sklearn.metrics.accuracy_score') as mock_accuracy:
        
        # Set up mock returns
        mock_load_dataset.return_value = {
            "test": MagicMock(column_names=["text", "label"])
        }
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_device.return_value = MagicMock()
        mock_cuda_available.return_value = False

        # Mock TextDataset to include labels in the batch
        mock_text_dataset_instance = MagicMock()
        mock_text_dataset.return_value = mock_text_dataset_instance

        # Create a fake DataLoader batch
        fake_batch = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]]),
            'labels': torch.tensor([0, 1])  # Ensure labels are included
        }
        mock_dataloader.return_value = [fake_batch]  # Mock DataLoader as an iterable returning the batch

        yield {
            'load_dataset': mock_load_dataset,
            'tokenizer': mock_tokenizer,
            'model': mock_model,
            'device': mock_device,
            'cuda_available': mock_cuda_available,
            'dataloader': mock_dataloader,
            'rename_features': mock_rename_features,
            'text_dataset': mock_text_dataset,
            'data_collator': mock_data_collator,
            'no_grad': mock_no_grad,
            'prf': mock_prf,
            'accuracy': mock_accuracy
        }

def test_test_predict(mock_dependencies, capsys):
    # Set up mock model output
    mock_output = MagicMock()
    mock_output.logits = torch.randn(2, 2)  # Example tensor with batch size 2 and 2 classes
    mock_dependencies['model'].return_value.return_value = mock_output

    # Call the function
    test_predict()

    # Verify that key functions were called
    mock_dependencies['load_dataset'].assert_called_once()
    mock_dependencies['tokenizer'].assert_called()
    mock_dependencies['model'].assert_called_once()
    mock_dependencies['text_dataset'].assert_called_once()
    mock_dependencies['dataloader'].assert_called_once()
    mock_dependencies['prf'].assert_called_once()
    mock_dependencies['accuracy'].assert_called_once()