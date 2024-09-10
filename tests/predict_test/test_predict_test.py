"""Test predict function."""


def test_predict() -> None:
    """Test predict function."""
    test_dataset = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "label": 1},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "label": 0},
    ]

    predictions = []
    labels = []

    for data in test_dataset:
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        label = data["label"]

        # Perform prediction logic here
        prediction = 1 if sum(input_ids) > sum(attention_mask) else 0

        predictions.append(prediction)
        labels.append(label)

    # Calculate metrics
    accuracy = sum([1 for p, l in zip(predictions, labels) if p == l]) / len(labels)
    precision = sum(
        [1 for p, l in zip(predictions, labels) if p == l and p == 1]
    ) / sum(predictions)
    recall = sum([1 for p, l in zip(predictions, labels) if p == l and p == 1]) / sum(
        labels
    )
    f1 = 2 * (precision * recall) / (precision + recall)

    print(
        f"Accuracy: {accuracy}, F1: {f1}, Precision:" f" {precision}, Recall: {recall}"
    )
