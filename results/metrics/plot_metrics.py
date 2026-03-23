"""
Plot training history and final evaluation metrics.
Run: python results/metrics/plot_metrics.py
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

HISTORY_FILE = "results/metrics/training_history.json"
OUT_DIR      = Path("results/metrics")

CATEGORIES = ["person", "car", "dog", "bicycle", "cat", "chair", "bottle", "laptop"]
COLORS     = plt.cm.get_cmap("tab10", len(CATEGORIES))


def load_history():
    with open(HISTORY_FILE) as f:
        return json.load(f)


def plot_loss_curves(history: list, ax):
    epochs     = [e["epoch"] for e in history]
    loss_keys  = ["loss_classifier", "loss_box_reg", "loss_mask",
                  "loss_objectness", "loss_rpn_box_reg"]
    colors     = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for key, color in zip(loss_keys, colors):
        vals = [e["train_losses"].get(key, 0) for e in history]
        ax.plot(epochs, vals, label=key.replace("loss_", ""), color=color,
                linewidth=2, marker="o", markersize=4)

    # Phase boundary
    phase_b = next((e["epoch"] for e in history
                    if e["epoch"] > 1 and e.get("lr", 1) < history[0].get("lr", 1)), None)
    if phase_b:
        ax.axvline(x=phase_b, color="gray", linestyle="--", alpha=0.6, label="Phase B start")

    ax.set_title("Training Loss Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)


def plot_ap_over_epochs(history: list, ax):
    eval_epochs = [e for e in history if e.get("metrics")]
    if not eval_epochs:
        ax.text(0.5, 0.5, "No evaluation data", ha="center", va="center")
        return

    epochs = [e["epoch"] for e in eval_epochs]
    ap50   = [e["metrics"].get("AP50", 0) for e in eval_epochs]
    ap75   = [e["metrics"].get("AP75", 0) for e in eval_epochs]
    mAP    = [e["metrics"].get("mAP",  0) for e in eval_epochs]

    ax.plot(epochs, ap50, label="AP50",        color="#e74c3c", linewidth=2, marker="o")
    ax.plot(epochs, ap75, label="AP75",        color="#3498db", linewidth=2, marker="s")
    ax.plot(epochs, mAP,  label="mAP@[.5:.95]",color="#2ecc71", linewidth=2, marker="^")

    ax.set_title("AP Metrics Over Training", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xticks(epochs)


def plot_per_category_ap(history: list, ax):
    eval_epochs = [e for e in history if e.get("metrics")]
    if not eval_epochs:
        return

    last = eval_epochs[-1]["metrics"].get("per_category_AP50", {})
    cats = list(last.keys())
    vals = list(last.values())
    colors = [COLORS(i) for i in range(len(cats))]

    bars = ax.barh(cats, vals, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_title("Per-Category AP50 (Final)", fontsize=13, fontweight="bold")
    ax.set_xlabel("AP50")
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="x")


def plot_ap_per_threshold(history: list, ax):
    eval_epochs = [e for e in history if e.get("metrics")]
    if not eval_epochs:
        return

    last    = eval_epochs[-1]["metrics"].get("AP_per_threshold", {})
    thrs    = [float(k.replace("AP", "")) / 100 for k in last.keys()]
    vals    = list(last.values())

    ax.plot(thrs, vals, color="#e74c3c", linewidth=2, marker="o", markersize=5)
    ax.fill_between(thrs, vals, alpha=0.15, color="#e74c3c")
    ax.axvline(x=0.5,  color="gray", linestyle="--", alpha=0.5, label="IoU=0.50")
    ax.axvline(x=0.75, color="gray", linestyle=":",  alpha=0.5, label="IoU=0.75")

    ax.set_title("AP at Each IoU Threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("AP")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0, max(vals) * 1.3 if vals else 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    print("Loading training history ...")
    history = load_history()
    print(f"  {len(history)} epochs found")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Mask R-CNN Training & Evaluation Report",
                 fontsize=16, fontweight="bold", y=0.98)

    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[1, 0])
    ax4  = fig.add_subplot(gs[1, 1])

    plot_loss_curves(history, ax1)
    plot_ap_over_epochs(history, ax2)
    plot_per_category_ap(history, ax3)
    plot_ap_per_threshold(history, ax4)

    out = OUT_DIR / "training_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✅ Report saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()