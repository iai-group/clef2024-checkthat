import pytest
from unittest.mock import patch, mock_open
from checkthat.task1.training_scripts.train_config import (
    load_config,
    get_training_arguments,
)


def test_load_config():
    """Test for load_config."""
    with patch("builtins.open", mock_open(read_data="training_arguments: {}")):
        config = load_config("dummy_path")
        assert config == {"training_arguments": {}}


def test_get_training_arguments():
    """Test for get_training_arguments."""
    with patch(
        "checkthat.task1.training_scripts.train_config.load_config",
        return_value={"training_arguments": {"output_dir": "test"}},
    ):
        training_args = get_training_arguments()
        assert training_args.output_dir == "test"
