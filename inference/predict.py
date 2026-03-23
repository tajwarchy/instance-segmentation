"""
Single image or folder inference.
Usage:
    python inference/predict.py --input path/to/image.jpg
    python inference/predict.py --input path/to/folder/ --output results/visualizations/
"""

import sys
import argparse
import time
import yaml
import torch
import cv2
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mask_rcnn import build_model, load_checkpoint
from inference.postprocess import postprocess_outputs
from training.visualization import draw_instances, CATEGORY_NAMES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, weights: str, device: torch.device):
    model = build_model(
        num_classes = cfg["model"]["num_classes"],
        pretrained  = False,
        min_size    = cfg["model"]["min_size"],
        max_size    = cfg["model"]["max_size"],
    ).to(device)
    load_checkpoint(model, weights, device)
    model.eval()
    return model


def preprocess_image(img_bgr, size: int = 512):
    """BGR numpy → normalized float tensor (C, H, W)."""
    import numpy as np
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rsz = cv2.resize(img_rgb, (size, size))
    tensor  = torch.from_numpy(img_rsz).permute(2, 0, 1).float() / 255.0
    tensor  = (tensor - mean) / std
    return tensor


@torch.no_grad()
def predict_single(model, img_tensor, device, cfg) -> dict:
    """Run inference on a single preprocessed image tensor."""
    t0      = time.time()
    inputs  = [img_tensor.to(device)]
    outputs = model(inputs)
    elapsed = time.time() - t0

    processed = postprocess_outputs(
        outputs,
        score_thr   = cfg["inference"]["score_threshold"],
        mask_thr    = cfg["inference"]["mask_threshold"],
        nms_iou_thr = cfg["inference"]["nms_iou_threshold"],
        img_size    = (img_tensor.shape[1], img_tensor.shape[2]),
    )
    return processed[0], elapsed


def build_report(out: dict, elapsed: float, img_name: str) -> dict:
    """Build a structured prediction report for one image."""
    labels  = out["labels"].tolist()
    scores  = out["scores"].tolist()
    masks   = out["masks"]

    instances = []
    for i, (lbl, scr) in enumerate(zip(labels, scores)):
        mask_area = int(masks[i].sum().item()) if i < len(masks) else 0
        instances.append({
            "instance_id":  i,
            "class":        CATEGORY_NAMES.get(lbl, f"cls{lbl}"),
            "class_id":     lbl,
            "score":        round(scr, 4),
            "mask_area_px": mask_area,
            "box":          out["boxes"][i].tolist() if i < len(out["boxes"]) else [],
        })

    return {
        "image":          img_name,
        "n_instances":    len(instances),
        "inference_ms":   round(elapsed * 1000, 1),
        "instances":      instances,
    }


def run_on_image(img_path: Path, model, device, cfg, out_dir: Path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  ⚠️  Could not read {img_path}")
        return

    size      = cfg["model"]["min_size"]
    tensor    = preprocess_image(img_bgr, size)
    out, elapsed = predict_single(model, tensor, device, cfg)

    # Visualize
    vis = draw_instances(
        tensor,
        out["boxes"],
        out["labels"],
        out["scores"],
        out["masks"].unsqueeze(1).float(),
        score_thr = cfg["inference"]["score_threshold"],
        mask_thr  = 0.5,
    )

    # Save annotated image
    out_path = out_dir / f"{img_path.stem}_pred.jpg"
    cv2.imwrite(str(out_path), vis)

    # Save report
    report      = build_report(out, elapsed, img_path.name)
    report_path = out_dir / f"{img_path.stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  {img_path.name}: {report['n_instances']} instances "
          f"| {report['inference_ms']} ms → {out_path.name}")
    return report


def main(args):
    cfg    = load_config(args.config)
    device = torch.device("cpu")

    print(f"\nDevice:  {device}")
    print(f"Weights: {args.weights}")

    model   = load_model(cfg, args.weights, device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if input_path.is_dir():
        img_paths = [p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS]
        print(f"\nRunning on {len(img_paths)} images in {input_path} ...\n")
        reports = []
        for p in sorted(img_paths):
            r = run_on_image(p, model, device, cfg, out_dir)
            if r:
                reports.append(r)

        # Save combined report
        combined = out_dir / "batch_report.json"
        with open(combined, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\n✅ Batch report saved → {combined}")

    elif input_path.is_file():
        print(f"\nRunning on {input_path} ...\n")
        run_on_image(input_path, model, device, cfg, out_dir)
        print("\n✅ Done.")
    else:
        print(f"❌ Input not found: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True,
                        help="Path to image or folder")
    parser.add_argument("--output",  default="results/visualizations/predictions",
                        help="Output directory")
    parser.add_argument("--weights", default="weights/mask_rcnn_best.pth")
    parser.add_argument("--config",  default="configs/mask_rcnn_config.yaml")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override score threshold from config")
    args = parser.parse_args()

    if args.threshold is not None:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["inference"]["score_threshold"] = args.threshold

    main(args)