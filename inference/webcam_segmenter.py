"""
Real-time webcam instance segmentation.
Usage:
    python inference/webcam_segmenter.py

Controls:
    Q — quit
    S — screenshot
    T — toggle mask / box-only mode
    R — toggle report overlay
    + / - — raise / lower score threshold
"""

import sys
import time
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mask_rcnn import build_model, load_checkpoint
from inference.postprocess import postprocess_outputs
from training.visualization import INSTANCE_COLORS, CATEGORY_NAMES

SCREENSHOT_DIR = Path("results/visualizations/webcam_screenshots")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess_frame(frame_bgr, size: int = 512):
    mean   = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std    = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - mean) / std
    return tensor


def draw_hud(canvas, fps, n_inst, score_thr, mode, show_report, instances):
    h, w = canvas.shape[:2]

    # ── Top bar ──
    cv2.rectangle(canvas, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
    cv2.putText(canvas, f"Instances: {n_inst}", (130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    cv2.putText(canvas, f"Thr: {score_thr:.2f}", (310, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    mode_str = "MASK" if mode == "mask" else "BOX"
    cv2.putText(canvas, mode_str, (430, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # ── Controls reminder ──
    cv2.putText(canvas, "Q:quit  S:save  T:mode  R:report  +/-:threshold",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # ── Per-instance report panel ──
    if show_report and instances:
        panel_w = 200
        panel_h = min(len(instances) * 26 + 14, h - 60)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (w - panel_w - 8, 44),
                      (w - 8, 44 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.putText(canvas, "Detections", (w - panel_w, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, inst in enumerate(instances[:min(len(instances), 12)]):
            color   = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
            color_b = (color[2], color[1], color[0])
            text    = f"{inst['class']} {inst['score']:.2f}"
            y       = 78 + i * 24
            cv2.circle(canvas, (w - panel_w - 2, y - 4), 5, color_b, -1)
            cv2.putText(canvas, text, (w - panel_w + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)


def render_frame(frame_bgr, out, mode, score_thr, mask_thr=0.5):
    """Draw instances on a BGR frame."""
    canvas  = frame_bgr.copy()
    overlay = canvas.copy()

    boxes  = out["boxes"]
    labels = out["labels"]
    scores = out["scores"]
    masks  = out["masks"]

    instances = []

    for i in range(len(boxes)):
        color     = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        color_bgr = (color[2], color[1], color[0])
        label_str = CATEGORY_NAMES.get(labels[i].item(), f"cls{labels[i].item()}")
        score_val = scores[i].item()

        instances.append({"class": label_str, "score": score_val})

        # ── Mask ──
        if mode == "mask" and i < len(masks):
            mask = masks[i].numpy().astype(bool)
            # Resize mask to frame size
            mask_rsz = cv2.resize(
                mask.astype(np.uint8),
                (canvas.shape[1], canvas.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            overlay[mask_rsz] = (
                overlay[mask_rsz] * 0.4 + np.array(color_bgr) * 0.6
            ).astype(np.uint8)

            # Boundary contour
            mask_u8    = mask_rsz.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color_bgr, 2)

        # ── Box ──
        x1, y1, x2, y2 = boxes[i].int().tolist()
        # Scale box from 512 to frame size
        fh, fw = canvas.shape[:2]
        x1 = int(x1 * fw / 512); x2 = int(x2 * fw / 512)
        y1 = int(y1 * fh / 512); y2 = int(y2 * fh / 512)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)

        text = f"{label_str} {score_val:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(canvas, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if mode == "mask":
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

    return canvas, instances


def main():
    cfg    = load_config("configs/mask_rcnn_config.yaml")
    device = torch.device("cpu")
    size   = cfg["model"]["min_size"]

    print(f"Device: {device}")
    print("Loading model ...")
    model = build_model(
        num_classes = cfg["model"]["num_classes"],
        pretrained  = False,
        min_size    = size,
        max_size    = size,
    ).to(device)
    load_checkpoint(model, "weights/mask_rcnn_best.pth", device)
    model.eval()
    print("✅ Model loaded\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── State ──
    score_thr   = cfg["inference"]["score_threshold"]
    mask_thr    = cfg["inference"]["mask_threshold"]
    mode        = "mask"        # "mask" | "box"
    show_report = True
    fps         = 0.0
    prev_time   = time.time()

    print("Webcam running ...")
    print("  Q — quit  |  S — screenshot  |  T — toggle mode")
    print("  R — report overlay  |  + / - — threshold\n")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame capture failed")
                break

            # ── Inference ──
            tensor  = preprocess_frame(frame, size)
            outputs = model([tensor.to(device)])
            out     = postprocess_outputs(
                outputs,
                score_thr   = score_thr,
                mask_thr    = mask_thr,
                nms_iou_thr = cfg["inference"]["nms_iou_threshold"],
                img_size    = (size, size),
            )[0]

            # ── Render ──
            canvas, instances = render_frame(frame, out, mode, score_thr, mask_thr)

            # ── FPS ──
            now       = time.time()
            fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            draw_hud(canvas, fps, len(instances), score_thr,
                     mode, show_report, instances)

            cv2.imshow("Instance Segmentation — Mask R-CNN", canvas)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = SCREENSHOT_DIR / f"screenshot_{ts}.jpg"
                cv2.imwrite(str(path), canvas)
                print(f"  📸 Screenshot saved → {path}")
            elif key == ord("t"):
                mode = "box" if mode == "mask" else "mask"
                print(f"  Mode: {mode.upper()}")
            elif key == ord("r"):
                show_report = not show_report
            elif key == ord("+") or key == ord("="):
                score_thr = min(0.95, round(score_thr + 0.05, 2))
                print(f"  Threshold: {score_thr:.2f}")
            elif key == ord("-"):
                score_thr = max(0.05, round(score_thr - 0.05, 2))
                print(f"  Threshold: {score_thr:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Webcam closed.")


if __name__ == "__main__":
    main()