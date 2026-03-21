"""
Standalone model architecture test.
Verifies forward pass, output keys, and head replacement.
Run: python models/model_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.mask_rcnn import build_model, freeze_backbone, count_parameters
from models.backbone_selection import get_device

NUM_CLASSES = 9   # 8 categories + background


def test_forward_pass(model, device):
    model.eval()
    dummy = [torch.rand(3, 512, 512).to(device)]
    with torch.no_grad():
        out = model(dummy)

    print("\n=== Forward Pass Output ===")
    print(f"  Output keys:   {list(out[0].keys())}")
    print(f"  Boxes shape:   {out[0]['boxes'].shape}")
    print(f"  Labels shape:  {out[0]['labels'].shape}")
    print(f"  Scores shape:  {out[0]['scores'].shape}")
    print(f"  Masks shape:   {out[0]['masks'].shape}")

    assert set(out[0].keys()) == {"boxes", "labels", "scores", "masks"}, \
        "Missing expected output keys"
    assert out[0]["masks"].shape[1] == 1, \
        f"Expected masks dim 1 = 1, got {out[0]['masks'].shape[1]}"

    print("  ✅ Forward pass assertions passed")


def test_training_pass(model, device):
    model.train()
    images = [torch.rand(3, 512, 512).to(device)]
    targets = [{
        "boxes":    torch.tensor([[50., 50., 200., 200.]], dtype=torch.float32).to(device),
        "labels":   torch.tensor([1], dtype=torch.int64).to(device),
        "masks":    torch.zeros((1, 512, 512), dtype=torch.uint8).to(device),
    }]
    targets[0]["masks"][0, 50:200, 50:200] = 1

    loss_dict = model(images, targets)

    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    assert "loss_mask" in loss_dict
    assert "loss_classifier" in loss_dict
    assert "loss_box_reg" in loss_dict

    print("  ✅ Training pass assertions passed")

def main():
    print("=== Mask R-CNN Model Test ===")

    device = get_device()
    print(f"\n  Device: {device}")

    # ── Build model ──
    print("\n=== Building Model ===")
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # ── Parameter counts ──
    params = count_parameters(model)
    print(f"\n=== Parameter Count ===")
    print(f"  Total:     {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen:    {params['frozen']:,}")

    # ── Verify head replacement ──
    print("\n=== Head Verification ===")
    box_out  = model.roi_heads.box_predictor.cls_score.out_features
    mask_out = model.roi_heads.mask_predictor.mask_fcn_logits.out_channels
    print(f"  Box predictor output classes:  {box_out}  (expected {NUM_CLASSES})")
    print(f"  Mask predictor output classes: {mask_out} (expected {NUM_CLASSES})")
    assert box_out  == NUM_CLASSES, f"Box head mismatch: {box_out} != {NUM_CLASSES}"
    assert mask_out == NUM_CLASSES, f"Mask head mismatch: {mask_out} != {NUM_CLASSES}"
    print("  ✅ Head replacement verified")

    # ── Freeze test ──
    print("\n=== Freeze/Unfreeze Test ===")
    freeze_backbone(model)
    params_frozen = count_parameters(model)
    print(f"  Trainable after freeze: {params_frozen['trainable']:,}")

    # ── Forward pass (eval) ──
    test_forward_pass(model, device)

    # ── Training pass ──
    print("\n=== Training Pass Losses (CPU — MPS too memory-heavy for this test) ===")
    model_cpu = model.to("cpu")
    test_training_pass(model_cpu, torch.device("cpu"))
    model = model.to(device)  # move back to MPS

    print("\n✅ All model tests passed. Phase 3 complete.")


if __name__ == "__main__":
    main()