"""Tests for the predict_unlabeled.py script."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import argparse

# Assuming the main script is in a file named 'predict_unlabeled.py'
from checkthat.task1.predict.predict_unlabeled import main


@pytest.fixture
def mock_data():
    """Mock data for testing."""
    return pd.DataFrame(
        {
            "Text": ["This is a test sentence.", "Another test sentence."],
            "Sentence_id": [1, 2],
        }
    )


@pytest.fixture
def mock_args():
    """Mock arguments for testing."""
    args = argparse.Namespace()
    args.model_name = "roberta-large"
    args.input_data = "test_data.tsv"
    return args


def test_main(mock_data, mock_args):
    """Test the main function."""
    # Mock all external functions and classes
    with patch("pandas.read_csv", return_value=mock_data), patch(
        "datasets.Dataset.from_pandas"
    ), patch("transformers.AutoTokenizer.from_pretrained"), patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained"
    ), patch(
        "torch.device"
    ), patch(
        "torch.utils.data.DataLoader"
    ), patch(
        "builtins.open", MagicMock()
    ):

        # Run the main function with mock arguments
        main(mock_args)

        # Check if the output file was attempted to be written
        open.assert_called_with("unlabeled_test_results_roberta-large.tsv", "w")


if __name__ == "__main__":
    pytest.main()
