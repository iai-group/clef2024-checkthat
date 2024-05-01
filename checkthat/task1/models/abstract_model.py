"""Module contains the transformer model for Task 1.

Abstract indended as blueprint custom masked language model class.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Model(ABC, nn.Module):
    def __init__(self) -> None:
        """Constructor for the Model class."""
        super(Model, self).__init__()
        return None

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        """Forward pass of the model."""
        return x
