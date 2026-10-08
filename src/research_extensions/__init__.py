"""Register the dataset, Sobel losses, and training adapters.

Import this package before building a model from the project configuration.
The registration changes no prediction values; it only makes the custom
classes named in the configuration available to MMSegmentation's registry.
"""

from .mydata import MyDataset
from .sobel_losses import SobelImageEdgeAwareLoss, SobelMaskEdgeAwareLoss
from .sobel_image_training import SobelImageEncoderDecoder, SobelImagePSPHead

__all__ = ["MyDataset", "SobelImageEdgeAwareLoss",
           "SobelMaskEdgeAwareLoss", "SobelImageEncoderDecoder", "SobelImagePSPHead"]
