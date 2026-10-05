"""Register the dataset and edge-aware loss with MMSegmentation.

Import this package before building a model from the archived configuration.
The registration changes no prediction values; it only makes the custom
classes named in the configuration available to MMSegmentation's registry.
"""

from .mydata import MyDataset
from .my_loss import EdgeAwareLoss
from .sobel_losses import SobelImageEdgeAwareLoss, SobelMaskEdgeAwareLoss
from .sobel_image_training import SobelImageEncoderDecoder, SobelImagePSPHead

__all__ = ["MyDataset", "EdgeAwareLoss", "SobelImageEdgeAwareLoss",
           "SobelMaskEdgeAwareLoss", "SobelImageEncoderDecoder", "SobelImagePSPHead"]
