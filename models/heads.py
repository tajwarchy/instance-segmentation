"""
Mask R-CNN Head Replacements
─────────────────────────────
Mask R-CNN has two heads that need replacing when fine-tuning on a custom
number of classes:

1. Box head (FastRCNNPredictor)
   - Input:  RoI-pooled features → 1024-dim fc
   - Output: class logits (num_classes,) + box deltas (num_classes * 4,)

2. Mask head (MaskRCNNPredictor)
   - Input:  RoI-aligned features → series of 256-channel convolutions
   - Output: per-class binary masks (num_classes, 28, 28)

Both heads are replaced so the output dimensions match our subset class count.
"""

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_box_predictor(in_features: int, num_classes: int) -> FastRCNNPredictor:
    """
    Replace the default box predictor head.

    Args:
        in_features:  size of input feature vector (typically 1024 for ResNet-50)
        num_classes:  number of classes INCLUDING background (e.g. 9 for 8 cats + bg)

    Returns:
        FastRCNNPredictor with correct output dims
    """
    return FastRCNNPredictor(in_features, num_classes)


def get_mask_predictor(
    in_channels: int,
    dim_reduced: int,
    num_classes: int
) -> MaskRCNNPredictor:
    """
    Replace the default mask predictor head.

    Args:
        in_channels:  number of input channels from mask head convolutions (256)
        dim_reduced:  hidden layer size (256 by convention)
        num_classes:  number of classes INCLUDING background

    Returns:
        MaskRCNNPredictor with correct output dims
    """
    return MaskRCNNPredictor(in_channels, dim_reduced, num_classes)