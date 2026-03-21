"""
Standalone sanity check — loads 5 samples and renders boxes + masks.
Run: python data/verify_dataset.py
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coco_dataset import COCOSubsetDataset, get_val_transforms

ANN_FILE = "data/coco/annotations/subset_val2017.json"
IMG_DIR  = "data/coco/val2017"
N        = 5

CATEGORY_NAMES = {
    1: "person", 2: "car",     3: "dog",    4: "bicycle",
    5: "cat",    6: "chair",   7: "bottle", 8: "laptop",
}

COLORS = plt.cm.get_cmap("tab10", 8)


def show_sample(img_tensor, target, ax, title=""):
    img = img_tensor.permute(1, 2, 0).numpy()
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img * std + mean).clip(0, 1)

    overlay = img.copy()

    for i, mask in enumerate(target["masks"]):
        color = COLORS(i % 8)[:3]
        m     = mask.numpy().astype(bool)
        overlay[m] = overlay[m] * 0.5 + np.array(color) * 0.5

    ax.imshow(overlay)

    for i, box in enumerate(target["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        color   = COLORS(i % 8)
        label   = CATEGORY_NAMES.get(target["labels"][i].item(), "?")
        rect    = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, label, color=color, fontsize=8,
                fontweight="bold", backgroundcolor="black")

    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    print("Loading dataset ...")
    ds = COCOSubsetDataset(IMG_DIR, ANN_FILE, transforms=get_val_transforms())
    print(f"Dataset size: {len(ds)} images")

    indices = random.sample(range(len(ds)), min(N, len(ds)))
    fig, axes = plt.subplots(1, N, figsize=(4 * N, 4))

    for ax, idx in zip(axes, indices):
        img, target = ds[idx]
        n_inst = len(target["boxes"])
        show_sample(img, target, ax, title=f"idx={idx} | {n_inst} instances")
        print(f"  Sample {idx}: {n_inst} instances, labels={target['labels'].tolist()}")

    plt.tight_layout()
    out = Path("results/visualizations/verify_dataset.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n✅ Saved verification image → {out}")
    plt.show()


if __name__ == "__main__":
    main()