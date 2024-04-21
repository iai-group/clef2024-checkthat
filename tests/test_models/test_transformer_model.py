"""Tests for the transformer model."""

import pytest
import torch
from checkthat.task1.models.transformer_model import Model


class ConcreteModel(Model):
    """Concrete model for testing purposes."""

    def forward(self, x):
        """Forward pass of the model.

        This method takes an input tensor and returns an output tensor where
        each element is doubled.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor, where each element is doubled.
        """
        return x * 2


def test_model_cannot_be_instantiated():
    """Test if the Model class cannot be instantiated.

    This test checks whether attempting to instantiate the abstract
    Model class raises a TypeError, as expected for abstract classes.
    """
    with pytest.raises(TypeError):
        Model()  # Directly test instantiation without assignment


def test_concrete_model():
    """Test if the ConcreteModel class works correctly.

    This test verifies that the ConcreteModel class's forward method
    processes input tensors correctly by doubling each element.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    model = ConcreteModel()
    expected_output = torch.tensor([2.0, 4.0, 6.0])
    assert torch.equal(
        model(x), expected_output
    ), "The output tensor does not match the expected doubled values."
