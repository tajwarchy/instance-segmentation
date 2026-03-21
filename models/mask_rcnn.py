import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

from models.heads import get_box_predictor, get_mask_predictor
from models.backbone_selection import get_backbone, get_device


def build_model(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 0,   # 0 = fully frozen for Phase A
    min_size: int = 512,
    max_size: int = 512,
) -> MaskRCNN:
    """
    Build a Mask R-CNN model with custom heads for num_classes.

    Architecture:
        Backbone:   ResNet-50 + FPN  (pretrained on ImageNet)
        RPN:        Region Proposal Network
        RoI heads:  Box predictor + Mask predictor (replaced for custom classes)

    Args:
        num_classes:                  number of classes + background
        pretrained:                   load COCO pretrained weights
        trainable_backbone_layers:    0 = frozen (Phase A), 5 = full (Phase B)
        min_size / max_size:          input image size constraints

    Returns:
        MaskRCNN model ready for fine-tuning
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # ── Replace box predictor head ──
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = get_box_predictor(in_features, num_classes)

    # ── Replace mask predictor head ──
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.out_channels
    dim_reduced      = 256
    model.roi_heads.mask_predictor = get_mask_predictor(
        in_channels_mask, dim_reduced, num_classes
    )

    return model


def freeze_backbone(model: MaskRCNN):
    """Freeze all backbone + FPN parameters (Phase A training)."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("  Backbone frozen ✅")


def unfreeze_backbone(model: MaskRCNN):
    """Unfreeze all backbone + FPN parameters (Phase B training)."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("  Backbone unfrozen ✅")


def count_parameters(model: MaskRCNN) -> dict:
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def load_checkpoint(model: MaskRCNN, path: str, device: torch.device) -> dict:
    """Load a saved checkpoint into the model. Returns the checkpoint dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


def save_checkpoint(model: MaskRCNN, path: str, epoch: int, metrics: dict):
    """Save model checkpoint with metadata."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "metrics":          metrics,
    }, path)
    print(f"  Checkpoint saved → {path}")