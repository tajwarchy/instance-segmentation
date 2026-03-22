import cv2
import numpy as np
import torch
from pathlib import Path

# Visually distinct colors for up to 20 instances
INSTANCE_COLORS = [
    (255, 56,  56 ), (255, 157, 151), (255, 112, 31 ), (255, 178, 29 ),
    (207, 210, 49 ), (72,  249, 10 ), (146, 204, 23 ), (61,  219, 134),
    (26,  147, 52 ), (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236), (132, 56,  255),
    (82,  0,   133), (203, 56,  255), (255, 149, 200), (255, 55,  199),
]

CATEGORY_NAMES = {
    1: "person", 2: "car",    3: "dog",    4: "bicycle",
    5: "cat",    6: "chair",  7: "bottle", 8: "laptop",
}


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized image tensor (C,H,W) → uint8 BGR numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)
    img  = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_instances(
    image:      torch.Tensor,
    boxes:      torch.Tensor,
    labels:     torch.Tensor,
    scores:     torch.Tensor,
    masks:      torch.Tensor,
    score_thr:  float = 0.5,
    mask_thr:   float = 0.5,
) -> np.ndarray:
    """
    Draw instance masks, bounding boxes, and labels on an image.

    Args:
        image:     (C, H, W) normalized tensor
        boxes:     (N, 4) float tensor [x1, y1, x2, y2]
        labels:    (N,) int tensor
        scores:    (N,) float tensor
        masks:     (N, 1, H, W) float tensor (raw sigmoid output)
        score_thr: minimum score to display
        mask_thr:  threshold to binarize masks

    Returns:
        BGR uint8 numpy array with all annotations drawn
    """
    canvas = denormalize(image)
    overlay = canvas.copy()

    keep = scores >= score_thr
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks  = masks[keep]

    for i in range(len(boxes)):
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        color_bgr = (color[2], color[1], color[0])

        # ── Mask ──
        mask = masks[i, 0].cpu().numpy() > mask_thr
        overlay[mask] = (
            overlay[mask] * 0.45 + np.array(color_bgr) * 0.55
        ).astype(np.uint8)

        # ── Boundary ──
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color_bgr, 2)

        # ── Box ──
        x1, y1, x2, y2 = boxes[i].int().tolist()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)

        # ── Label ──
        label_str = CATEGORY_NAMES.get(labels[i].item(), f"cls{labels[i].item()}")
        text      = f"{label_str} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(canvas, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend overlay (masks) with canvas (boxes + contours)
    result = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
    return result


def save_prediction_grid(
    images:   list,
    outputs:  list,
    save_path: str,
    score_thr: float = 0.5,
    max_imgs:  int   = 4,
):
    """Save a grid of prediction visualizations."""
    rendered = []
    for img, out in zip(images[:max_imgs], outputs[:max_imgs]):
        vis = draw_instances(
            img,
            out["boxes"].cpu(),
            out["labels"].cpu(),
            out["scores"].cpu(),
            out["masks"].cpu(),
            score_thr=score_thr,
        )
        rendered.append(vis)

    if not rendered:
        return

    # Pad to same size if needed
    h = max(r.shape[0] for r in rendered)
    w = max(r.shape[1] for r in rendered)
    padded = [
        cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, w - r.shape[1],
                           cv2.BORDER_CONSTANT, value=0)
        for r in rendered
    ]

    grid = np.concatenate(padded, axis=1)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, grid)
    print(f"  Saved prediction grid → {save_path}")