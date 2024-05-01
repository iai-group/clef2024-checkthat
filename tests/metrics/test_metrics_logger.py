"""Test for the metrics_logger module."""

import numpy as np
from unittest.mock import patch
from checkthat.task1.metrics.metrics_logger import (
    compute_custom_metrics,
    MetricsLoggerCallback,
    get_callbacks,
)  # Replace 'your_module' with the actual module name

# Test for compute_custom_metrics
def test_compute_custom_metrics():
    """Mock test for compute_custom_metrics."""
    # Define mock logits and labels
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.55, 0.45]])
    labels = np.array([1, 0, 1])

    # Expected results
    precision, recall, f1 = compute_custom_metrics(logits, labels)

    # Assert conditions
    assert precision >= 0, "Precision should be non-negative"
    assert recall >= 0, "Recall should be non-negative"
    assert f1 >= 0, "F1 score should be non-negative"
    # You can add more detailed assertions here based on known input and output


# Test for MetricsLoggerCallback
@patch(
    "checkthat.task1.metrics.metrics_logger.wandb.log"
)  # Mock the wandb.log method
def test_metrics_logger_callback(mock_log):
    """Mock test for MetricsLoggerCallback."""
    # Create an instance of MetricsLoggerCallback
    callback = MetricsLoggerCallback()

    # Create mock arguments
    args = None  # Depending on the real use case, populate this correctly
    state = type(
        "state", (object,), {"epoch": 1}
    )  # Mock state with an epoch attribute
    logits = np.array(
        [[10, 0], [0, 10]]
    )  # Very clear separation of class predictions
    labels = np.array([0, 1])  # Correct labels aligning with logits

    # Execute the on_evaluate method
    callback.on_evaluate(args, state, logits=logits, labels=labels)

    # Check that wandb.log was called with expected values
    mock_log.assert_called_with(
        {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "epoch": 1}
    )


# Test for get_callbacks
def test_get_callbacks():
    """Test for get_callbacks."""
    # Get callback instances
    callbacks = get_callbacks(["MetricsLoggerCallback"])
    # Check that the correct callbacks are returned
    assert len(callbacks) == 1 and isinstance(
        callbacks[0], MetricsLoggerCallback
    ), "Should return an instance of MetricsLoggerCallback"
