"""Test train_config.py."""

import pytest
from unittest.mock import mock_open, patch
from transformers import TrainingArguments
from checkthat.task1.training_scripts.train_config import (
    load_config,
    get_training_arguments,
    get_language,
)


# Test load_config
def test_load_config():
    """Test load_config function."""
    mock_data = """
    training_arguments:
        learning_rate: 5e-5
        num_train_epochs: 3
        per_device_train_batch_size: 8
    """
    with patch("builtins.open", mock_open(read_data=mock_data)) as mock_file:
        with patch(
            "yaml.safe_load",
            return_value={
                "training_arguments": {
                    "learning_rate": 5e-5,
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 8,
                }
            },
        ) as mock_yaml:
            config = load_config("dummy_path")
            assert config == {
                "training_arguments": {
                    "learning_rate": 5e-5,
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 8,
                }
            }
            mock_file.assert_called_once_with("dummy_path", "r")
            mock_yaml.assert_called_once()


# Test get_training_arguments
def test_get_training_arguments():
    """Test get_training_arguments function."""
    config = {
        "training_arguments": {
            "learning_rate": 5e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "output_dir": "",
        }
    }
    with patch(
        "checkthat.task1.training_scripts.train_config.load_config",
        return_value=config,
    ) as mock_load_config:
        training_args = get_training_arguments(
            "bert-base-multilingual-cased", 42, "dataset_fr"
        )
        assert isinstance(training_args, TrainingArguments)
        assert training_args.learning_rate == 5e-5
        assert training_args.num_train_epochs == 3
        assert training_args.per_device_train_batch_size == 8
        assert (
            training_args.output_dir
            == "./results/bert-base-multilingual-cased_seed_42_fr"
        )


# Test get_language
def test_get_language():
    """Test get_language function."""
    assert get_language("dataset_en") == "en"
    assert get_language("dataset_fr") == "fr"
