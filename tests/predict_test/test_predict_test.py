import pytest
from unittest.mock import patch, MagicMock
import torch

# Import the function you want to test
from checkthat.task1.predict.predict_test import test_predict


@pytest.fixture
def mock_dependencies():
    with patch("datasets.load_dataset") as mock_load_dataset, patch(
        "transformers.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer, patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained"
    ) as mock_model, patch(
        "torch.utils.data.DataLoader"
    ) as mock_dataloader, patch(
        "sklearn.metrics.precision_recall_fscore_support"
    ) as mock_prf, patch(
        "sklearn.metrics.accuracy_score"
    ) as mock_accuracy:

        # Set up basic mock returns
        mock_load_dataset.return_value = {
            "test": MagicMock(column_names=["text", "label"])
        }
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        # Create a simple mock DataLoader batch
        mock_dataloader.return_value = [
            {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                "labels": torch.tensor([0, 1]),
            }
        ]

        # Set up mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(2, 2)
        mock_model.return_value.return_value = mock_output

        yield {
            "load_dataset": mock_load_dataset,
            "tokenizer": mock_tokenizer,
            "model": mock_model,
            "dataloader": mock_dataloader,
            "prf": mock_prf,
            "accuracy": mock_accuracy,
        }


def test_test_predict(mock_dependencies):
    # Call the function
    test_predict()

    # Verify that key functions were called
    mock_dependencies["load_dataset"].assert_called_once()
    mock_dependencies["tokenizer"].assert_called()
    mock_dependencies["model"].assert_called()
    mock_dependencies["dataloader"].assert_called()
    mock_dependencies["prf"].assert_called_once()
    mock_dependencies["accuracy"].assert_called_once()


if __name__ == "__main__":
    pytest.main()
