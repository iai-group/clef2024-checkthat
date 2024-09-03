import pytest
from unittest.mock import patch, MagicMock

# Import the function to be tested
from checkthat.task1.predict.predict_test import test_predict

@pytest.fixture
def mock_dependencies():
    with patch('torch.device') as mock_device, \
         patch('torch.cuda.is_available') as mock_cuda_available, \
         patch('torch.utils.data.DataLoader') as mock_dataloader, \
         patch('torch.no_grad') as mock_no_grad, \
         patch('sklearn.metrics.precision_recall_fscore_support') as mock_prf, \
         patch('sklearn.metrics.accuracy_score') as mock_accuracy:
        
        # Set up minimal mock returns
        mock_device.return_value = MagicMock()
        mock_cuda_available.return_value = False
        mock_dataloader.return_value = [
            {
                'input_ids': MagicMock(),
                'attention_mask': MagicMock(),
                'labels': MagicMock()
            }
        ]
        mock_no_grad.return_value.__enter__.return_value = None
        mock_prf.return_value = (0.5, 0.5, 0.5, None)
        mock_accuracy.return_value = 0.5

        yield {
            'dataloader': mock_dataloader,
            'prf': mock_prf,
            'accuracy': mock_accuracy
        }

def test_test_predict(mock_dependencies, capsys):
    # Call the function
    test_predict()

    # Check that the function ran and produced output
    captured = capsys.readouterr()
    assert "Accuracy:" in captured.out
    assert "F1:" in captured.out
    assert "Precision:" in captured.out
    assert "Recall:" in captured.out

    # Verify that key metric calculations were called
    mock_dependencies['prf'].assert_called_once()
    mock_dependencies['accuracy'].assert_called_once()

