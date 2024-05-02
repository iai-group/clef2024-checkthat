"""Sets up the logging for the metrics using Weights and Biases."""
import wandb
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import TrainerCallback


def compute_custom_metrics(logits, labels):
    """Compute precision, recall, and F1-score from model logits and true
    labels.

    Args:
    logits (np.array): Logits returned by the model. Shape (num_samples, num_classes).
    labels (np.array): True labels. Shape (num_samples,).

    Returns:
    tuple: precision, recall, F1-score
    """

    predictions = np.argmax(logits, axis=1)  # Convert logits to predictions

    # Calculate metrics
    precision = precision_score(labels, predictions, average="macro", pos_label=1)
    recall = recall_score(labels, predictions, average="macro", pos_label=1)
    f1 = f1_score(labels, predictions, average="macro", pos_label=1)

    return precision, recall, f1


class MetricsLoggerCallback(TrainerCallback):
    """Custom callback for logging additional metrics to wandb."""

    def on_evaluate(self, args, state, **kwargs):
        # Assuming 'logits' and 'labels' are part of the outputs collected during evaluation
        logits = kwargs["logits"]
        labels = kwargs["labels"]

        # Compute custom metrics
        precision, recall, f1 = compute_custom_metrics(logits, labels)

        # Log custom metrics to wandb
        wandb.log(
            {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "epoch": state.epoch,
            }
        )


callback_map = {
    "MetricsLoggerCallback": MetricsLoggerCallback,
}


def get_callbacks(callback_names):
    """Create a list of callback instances from a list of callback names."""
    return [
        callback_map[name]() for name in callback_names if name in callback_map
    ]
