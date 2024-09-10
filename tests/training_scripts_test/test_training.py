"""Tests for training script."""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from transformers import EvalPrediction
from checkthat.task1.training_scripts.training import (
    run_training,
    compute_metrics,
    load_config,
)


@pytest.fixture
def mock_env():
    """Mock environment for training script."""
    with patch(
        "checkthat.task1.training_scripts.training.wandb.init"
    ) as mock_wandb_init, patch(
        "checkthat.task1.training_scripts.training.wandb.log"
    ) as mock_wandb_log, patch(
        "builtins.open",
        mock_open(
            read_data="parameters:\n  learning_rate:\n    values: [0.01]\n  batch_size:\n    values: [16]"
        ),
    ), patch(
        "builtins.open", new_callable=MagicMock
    ) as mock_file, patch(
        "yaml.safe_load",
        return_value={
            "parameters": {
                "learning_rate": {"values": [0.01]},
                "batch_size": {"values": [16]},
            }
        },
    ) as mock_yaml_load, patch(
        "checkthat.task1.training_scripts.training.AutoModelForSequenceClassification"
    ) as mock_model, patch(
        "checkthat.task1.training_scripts.training.AutoTokenizer"
    ) as mock_tokenizer, patch(
        "checkthat.task1.training_scripts.training.Trainer"
    ) as mock_trainer:

        # Set up mock returns for wandb.init
        mock_wandb_run = MagicMock()
        mock_wandb_run.name = "test_run"
        mock_wandb_init.return_value = mock_wandb_run

        # Mock specific model functions
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Setup trainer behavior
        mock_trainer_instance.train.return_value = None
        mock_trainer_instance.evaluate.return_value = {"eval_accuracy": 0.85}
        mock_trainer_instance.predict.return_value = MagicMock(
            metrics={"test_accuracy": 0.90}
        )

        yield {
            "wandb_init": mock_wandb_init,
            "wandb_log": mock_wandb_log,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "trainer": mock_trainer,
            "model_instance": mock_model_instance,
            "tokenizer_instance": mock_tokenizer_instance,
            "trainer_instance": mock_trainer_instance,
        }


def test_run_training_calls_external_functions(mock_env):
    """Test that run_training calls external functions."""
    # Define dummy datasets and parameters
    train_dataset = MagicMock()
    eval_dataset = MagicMock()
    test_dataset = MagicMock()
    label_map = {"positive": 0, "negative": 1}

    # Call the function
    model_path, tokenizer_path = run_training(
        train_dataset,
        eval_dataset,
        "bert-base-uncased",
        label_map,
        "en",
        test_dataset,
    )

    # Verify external functions are called
    assert mock_env["wandb_init"].called
    assert mock_env["wandb_log"].called
    assert mock_env["model"].from_pretrained.called
    assert mock_env["tokenizer"].from_pretrained.called
    assert mock_env["trainer_instance"].train.called
    assert mock_env["trainer_instance"].evaluate.called
    assert mock_env["trainer_instance"].predict.called


def test_compute_metrics():
    """Test compute_metrics function."""
    # Dummy predictions and labels
    predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0])
    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

    # Compute metrics
    results = compute_metrics(eval_pred)

    # Verify results contain expected keys
    assert "accuracy" in results
    assert "f1" in results
    assert "precision" in results
    assert "recall" in results
