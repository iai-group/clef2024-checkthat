"""Function to compute four metrics: accuracy, precision, recall, and F1-score.

Metrics will be passed to wandb for logging.
"""
from evaluate import load

"""Compute accuracy, precision, recall, and F1-score metrics."""
accuracy_metric = load("accuracy")
precision_metric = load("precision")
recall_metric = load("recall")
f1_metric = load("f1")


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1-score metrics.

    Args:
        eval_pred: Tuple of logits and labels.

    Returns:
        dict: Dictionary containing the computed metrics.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return {
        "accuracy": accuracy_metric.compute(
            predictions=predictions, references=labels
        )["accuracy"],
        "precision": precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["precision"],
        "recall": recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["recall"],
        "f1": f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"],
    }
