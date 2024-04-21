"""Module contains the transformer model for Task 1.

Abstract class for later use.
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class Model(ABC, nn.Module):
    def __init__(self):
        """Constructor for the Model class."""
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor.
        """
        pass
