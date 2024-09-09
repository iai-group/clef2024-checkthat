import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch


from checkthat.task1.predict.predict_unlabeled import main


@pytest.fixture
def mock_data():
    return pd.DataFrame(
        {
            "Text": ["This is a test sentence.", "Another test sentence."],
            "Sentence_id": [1, 2],
        }
    )


def test_main(mock_data):
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

        # Run the main function
        main()

        # Check if the output file was attempted to be written
        open.assert_called_with("unlabeled_test_results.tsv", "w")


if __name__ == "__main__":
    pytest.main()
