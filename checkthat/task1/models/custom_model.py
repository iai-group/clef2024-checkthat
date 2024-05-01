"""Custom model for sequence classification tasks dervied from abstract class
model.py."""
from .abstract_model import Model
from transformers import AutoModelForSequenceClassification


class CustomModel(Model):
    def __init__(self, model_name: str, num_labels: int, device: str):
        """Constructor for the CustomModel class.

        Args:
            model_name (str): Accepts huggingface model name
            num_labels (int): Number of labels in the dataset
        """
        super(CustomModel, self).__init__()
        if device is not None:
            self.to(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass of the model.

        Including labels in the forward pass so the model can calculate
        loss.
        """
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output
