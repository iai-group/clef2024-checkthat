import pytest
from unittest.mock import patch, MagicMock
from checkthat.task1.training_scripts.training import run_training


@patch("checkthat.task1.training_scripts.training.wandb.init")
@patch("checkthat.task1.training_scripts.training.Trainer")
@patch(
    "checkthat.task1.training_scripts.train_config.load_config"
)  # Mock load_config function
def test_run_training(mock_load_config, mock_trainer, mock_wandb_init):
    # Setup the mocks
    mock_trainer.return_value.train.return_value = None
    mock_wandb_init.return_value = None
    mock_load_config.return_value = {
        "training_arguments": {
            "output_dir": "some/path",
            "evaluation_strategy": "steps",
            # Add other needed arguments
        }
    }

    # Call the function
    run_training(
        seed=42,
        dataset={"train": [], "validation": [], "test": []},
        model_name="bert-base-uncased",
        tokenizer=MagicMock(),
        label_map={},
    )

    # Assertions
    mock_wandb_init.assert_called_once()
    mock_trainer.assert_called_once()
