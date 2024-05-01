"""Test cases for the normalize_DatasetDict_features function."""
import pytest
from datasets import DatasetDict, Dataset
from checkthat.task1.tokenization.normalize_DatasetDict_featues import (
    rename_features,
)


@pytest.fixture
def sample_data():
    """Fixture providing sample data."""
    # Create a DatasetDict with sample data
    train_data = {
        "tweet_id": [1, 2, 3],
        "tweet_text": ["text1", "text2", "text3"],
        "class_label": [0, 1, 0],
    }
    validation_data = {
        "tweet_id": [4, 5],
        "tweet_text": ["text4", "text5"],
        "class_label": [1, 0],
    }
    test_data = {
        "tweet_id": [6, 7],
        "tweet_text": ["text6", "text7"],
        "class_label": [0, 1],
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(train_data),
            "validation": Dataset.from_dict(validation_data),
            "test": Dataset.from_dict(test_data),
        }
    )


@pytest.fixture
def expected_data():
    """Fixture providing the expected data after renaming 'tweet_text' to
    'Text'."""
    # Define the expected result after renaming 'tweet_text' to 'Text'
    train_data = {
        "tweet_id": [1, 2, 3],
        "Text": ["text1", "text2", "text3"],
        "class_label": [0, 1, 0],
    }
    validation_data = {
        "tweet_id": [4, 5],
        "Text": ["text4", "text5"],
        "class_label": [1, 0],
    }
    test_data = {
        "tweet_id": [6, 7],
        "Text": ["text6", "text7"],
        "class_label": [0, 1],
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(train_data),
            "validation": Dataset.from_dict(validation_data),
            "test": Dataset.from_dict(test_data),
        }
    )


def test_rename_features(sample_data, expected_data):
    """Test for the rename_features function."""
    # Call the function to rename features
    result = rename_features(sample_data)

    # Compare individual datasets within result and expected_data
    for split_name in sample_data.keys():
        result_dataset = result[split_name]
        expected_dataset = expected_data[split_name]

        # Check if feature names are the same
        assert result_dataset.features == expected_dataset.features

        # Check if number of rows is the same
        assert len(result_dataset) == len(expected_dataset)

        # Check if each row in result matches corresponding row in expected_data
        for result_row, expected_row in zip(result_dataset, expected_dataset):
            assert result_row == expected_row
