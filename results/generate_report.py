"""
Generate a detailed segmentation report for any inference run.
Reads batch_report.json from predict.py and produces a summary.
Run: python results/generate_report.py
     python results/generate_report.py --input results/visualizations/predictions/batch_report.json
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("results/metrics")


def load_batch_report(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def analyze_report(reports: list) -> dict:
    total_images    = len(reports)
    total_instances = sum(r["n_instances"] for r in reports)
    inference_times = [r["inference_ms"] for r in reports]

    # Per-class counts and scores
    class_counts = defaultdict(int)
    class_scores = defaultdict(list)
    mask_areas   = defaultdict(list)

    for r in reports:
        for inst in r.get("instances", []):
            cls = inst["class"]
            class_counts[cls] += 1
            class_scores[cls].append(inst["score"])
            if "mask_area_px" in inst:
                mask_areas[cls].append(inst["mask_area_px"])

    per_class = {}
    for cls in class_counts:
        scores = class_scores[cls]
        areas  = mask_areas[cls]
        per_class[cls] = {
            "count":          class_counts[cls],
            "mean_score":     round(float(np.mean(scores)), 4),
            "min_score":      round(float(np.min(scores)),  4),
            "max_score":      round(float(np.max(scores)),  4),
            "mean_mask_area": round(float(np.mean(areas)),  1) if areas else 0,
            "min_mask_area":  int(np.min(areas))  if areas else 0,
            "max_mask_area":  int(np.max(areas))  if areas else 0,
        }

    return {
        "generated_at":        datetime.now().isoformat(),
        "total_images":        total_images,
        "total_instances":     total_instances,
        "avg_instances_per_image": round(total_instances / max(total_images, 1), 2),
        "inference_ms": {
            "mean": round(float(np.mean(inference_times)), 1),
            "min":  round(float(np.min(inference_times)),  1),
            "max":  round(float(np.max(inference_times)),  1),
        },
        "per_class": per_class,
    }


def save_txt_report(analysis: dict, path: Path):
    lines = [
        "=" * 55,
        "       MASK R-CNN SEGMENTATION REPORT",
        "=" * 55,
        f"  Generated:          {analysis['generated_at']}",
        f"  Total images:       {analysis['total_images']}",
        f"  Total instances:    {analysis['total_instances']}",
        f"  Avg inst/image:     {analysis['avg_instances_per_image']}",
        "",
        "  Inference Time (ms):",
        f"    Mean: {analysis['inference_ms']['mean']}",
        f"    Min:  {analysis['inference_ms']['min']}",
        f"    Max:  {analysis['inference_ms']['max']}",
        "",
        "-" * 55,
        f"  {'Class':<12} {'Count':>6}  {'Mean Score':>10}  {'Mean Mask Area':>14}",
        "-" * 55,
    ]

    for cls, stats in sorted(analysis["per_class"].items(),
                             key=lambda x: x[1]["count"], reverse=True):
        lines.append(
            f"  {cls:<12} {stats['count']:>6}  "
            f"{stats['mean_score']:>10.4f}  "
            f"{stats['mean_mask_area']:>14.1f}"
        )

    lines += ["=" * 55, ""]
    text = "\n".join(lines)

    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"  ✅ TXT report saved → {path}")


def plot_report(analysis: dict, out_dir: Path):
    per_class = analysis["per_class"]
    if not per_class:
        print("  ⚠️  No instance data to plot")
        return

    classes    = list(per_class.keys())
    counts     = [per_class[c]["count"]          for c in classes]
    scores     = [per_class[c]["mean_score"]     for c in classes]
    mask_areas = [per_class[c]["mean_mask_area"] for c in classes]

    colors = plt.cm.get_cmap("tab10", len(classes))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Segmentation Report", fontsize=14, fontweight="bold")

    # ── Instance count per class ──
    bars = axes[0].bar(classes, counts,
                       color=[colors(i) for i in range(len(classes))],
                       edgecolor="white")
    for bar, val in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(val), ha="center", fontsize=9)
    axes[0].set_title("Instance Count per Class")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(classes, rotation=30, ha="right")
    axes[0].grid(True, alpha=0.3, axis="y")

    # ── Mean confidence score per class ──
    bars = axes[1].bar(classes, scores,
                       color=[colors(i) for i in range(len(classes))],
                       edgecolor="white")
    for bar, val in zip(bars, scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", fontsize=9)
    axes[1].set_title("Mean Confidence Score per Class")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xticklabels(classes, rotation=30, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")

    # ── Mean mask area per class ──
    bars = axes[2].bar(classes, mask_areas,
                       color=[colors(i) for i in range(len(classes))],
                       edgecolor="white")
    for bar, val in zip(bars, mask_areas):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:.0f}", ha="center", fontsize=8)
    axes[2].set_title("Mean Mask Area (px) per Class")
    axes[2].set_ylabel("Pixels")
    axes[2].set_xticklabels(classes, rotation=30, ha="right")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = out_dir / "segmentation_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ✅ Plot saved → {out}")
    plt.show()


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading report from {args.input} ...")
    reports  = load_batch_report(args.input)
    analysis = analyze_report(reports)

    # Save JSON
    json_path = OUT_DIR / "segmentation_report.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  ✅ JSON saved → {json_path}\n")

    # Save TXT
    save_txt_report(analysis, OUT_DIR / "segmentation_report.txt")

    # Plot
    plot_report(analysis, OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/visualizations/predictions/batch_report.json",
        help="Path to batch_report.json from predict.py"
    )
    args = parser.parse_args()
    main(args)