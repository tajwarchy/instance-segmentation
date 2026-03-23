"""
Compare bounding box AP vs mask AP side by side.
Run: python results/metrics/compare_box_vs_mask.py
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml, torch
from data.coco_dataset import COCOSubsetDataset, get_val_transforms, collate_fn
from models.mask_rcnn import build_model, load_checkpoint
from training.evaluate import evaluate_one_epoch
from training.metrics import compute_coco_metrics, IOU_THRESHOLDS, CATEGORIES

with open("configs/mask_rcnn_config.yaml") as f:
    cfg = yaml.safe_load(f)

WEIGHTS = "weights/mask_rcnn_best.pth"
OUT_DIR = Path("results/metrics")


def main():
    device = torch.device("cpu")

    print("Loading model ...")
    model = build_model(num_classes=cfg["model"]["num_classes"], pretrained=False).to(device)
    load_checkpoint(model, WEIGHTS, device)

    print("Loading val dataset ...")
    val_ds = COCOSubsetDataset(
        cfg["dataset"]["val_images"],
        cfg["dataset"]["val_ann"],
        transforms=get_val_transforms(cfg["model"]["min_size"]),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=collate_fn)

    print("Running evaluation ...")
    preds, targets = evaluate_one_epoch(model, val_loader, device)

    # Compute both
    mask_metrics = compute_coco_metrics(preds, targets, use_masks=True)
    box_metrics  = compute_coco_metrics(preds, targets, use_masks=False)

    print(f"\n  Box  AP50: {box_metrics['AP50']:.4f}  mAP: {box_metrics['mAP']:.4f}")
    print(f"  Mask AP50: {mask_metrics['AP50']:.4f}  mAP: {mask_metrics['mAP']:.4f}")

    # ── Plot 1: AP at each threshold ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Bounding Box AP vs Mask AP", fontsize=14, fontweight="bold")

    thrs      = [float(k.replace("AP", "")) / 100
                 for k in mask_metrics["AP_per_threshold"].keys()]
    mask_vals = list(mask_metrics["AP_per_threshold"].values())
    box_vals  = list(box_metrics["AP_per_threshold"].values())

    axes[0].plot(thrs, box_vals,  label="Box AP",  color="#3498db",
                 linewidth=2, marker="s")
    axes[0].plot(thrs, mask_vals, label="Mask AP", color="#e74c3c",
                 linewidth=2, marker="o")
    axes[0].fill_between(thrs, box_vals,  alpha=0.1, color="#3498db")
    axes[0].fill_between(thrs, mask_vals, alpha=0.1, color="#e74c3c")
    axes[0].set_title("AP at Each IoU Threshold")
    axes[0].set_xlabel("IoU Threshold")
    axes[0].set_ylabel("AP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.45, 1.0)

    # ── Plot 2: Per-category comparison ──
    cats      = list(CATEGORIES.values())
    box_cat   = [box_metrics["per_category_AP50"].get(c, 0)  for c in cats]
    mask_cat  = [mask_metrics["per_category_AP50"].get(c, 0) for c in cats]
    x         = np.arange(len(cats))
    w         = 0.35

    axes[1].bar(x - w/2, box_cat,  w, label="Box AP50",  color="#3498db", alpha=0.85)
    axes[1].bar(x + w/2, mask_cat, w, label="Mask AP50", color="#e74c3c", alpha=0.85)
    axes[1].set_title("Per-Category AP50: Box vs Mask")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats, rotation=30, ha="right")
    axes[1].set_ylabel("AP50")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = OUT_DIR / "box_vs_mask_ap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved → {out}")
    plt.show()

    # Save JSON
    comparison = {"box": box_metrics, "mask": mask_metrics}
    with open(OUT_DIR / "box_vs_mask_metrics.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"✅ JSON saved → {OUT_DIR / 'box_vs_mask_metrics.json'}")


if __name__ == "__main__":
    main()