import pytest
import numpy as np
from unittest.mock import MagicMock
from evaluate import load
from checkthat.task1.metrics.compute_metrics import compute_metrics


@pytest.fixture
def mock_metrics(mocker):
    """Mock the load function to return predefined metric values.

    The mock function returns predefined values for accuracy, precision,
    recall, and f1.
    """
    mocker.patch(
        "evaluate.load",
        side_effect=lambda metric_name: MagicMock(
            compute=MagicMock(
                return_value={
                    "accuracy": 0.75,
                    "precision": 0.875,
                    "recall": 0.75,
                    "f1": 0.7666666666666667,
                }
                if metric_name in ["accuracy", "precision", "recall", "f1"]
                else None
            )
        ),
    )


def test_compute_metrics(mock_metrics):
    """Test the compute_metrics function."""
    logits = np.array(
        [
            [0.1, 0.9],  # Predicted class 1
            [0.8, 0.2],  # Predicted class 0
            [0.6, 0.4],  # Predicted class 0 (incorrect, should be 1)
            [0.3, 0.7],  # Predicted class 1 (correct)
        ]
    )
    labels = np.array([1, 0, 1, 1])
    eval_pred = (logits, labels)

    results = compute_metrics(eval_pred)

    assert results["accuracy"] == 0.75
    assert results["precision"] == pytest.approx(0.875)
    assert results["recall"] == 0.75
    assert results["f1"] == pytest.approx(0.7666666666666667)
