"""Register the dataset and edge-aware loss with MMSegmentation.

Import this package before building a model from the archived configuration.
The registration changes no prediction values; it only makes the custom
classes named in the configuration available to MMSegmentation's registry.
"""

from .mydata import MyDataset
from .my_loss import EdgeAwareLoss

__all__ = ["MyDataset", "EdgeAwareLoss"]
