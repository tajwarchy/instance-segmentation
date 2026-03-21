import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_device() -> torch.device:
    """Return the best available device: MPS → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_backbone(backbone_name: str = "resnet50", trainable_layers: int = 3):
    """
    Build a ResNet + FPN backbone for Mask R-CNN.

    Args:
        backbone_name:      resnet50 | resnet101
        trainable_layers:   number of layers to unfreeze (0=all frozen, 5=all trainable)

    Returns:
        backbone with FPN, out_channels=256
    """
    supported = ["resnet50", "resnet101"]
    if backbone_name not in supported:
        raise ValueError(f"backbone_name must be one of {supported}, got '{backbone_name}'")

    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights="DEFAULT",
        trainable_layers=trainable_layers,
    )

    print(f"  Backbone:          {backbone_name} + FPN")
    print(f"  Trainable layers:  {trainable_layers}")
    print(f"  Output channels:   {backbone.out_channels}")

    return backbone


def get_device_info():
    device = get_device()
    print(f"  Device:            {device}")
    if device.type == "mps":
        print(f"  MPS built:         {torch.backends.mps.is_built()}")
    return device