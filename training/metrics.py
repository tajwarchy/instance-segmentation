import torch
import numpy as np
from typing import Optional


# ─── IoU ──────────────────────────────────────────────────────────────────────

def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    boxes: (N, 4) [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / union.clamp(min=1e-6)


def mask_iou(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of binary masks.
    masks: (N, H, W) bool tensors
    Returns: (N, M) IoU matrix
    """
    masks_a = masks_a.bool().flatten(1).float()   # (N, H*W)
    masks_b = masks_b.bool().flatten(1).float()   # (M, H*W)

    inter = torch.mm(masks_a, masks_b.t())        # (N, M)
    area_a = masks_a.sum(1)
    area_b = masks_b.sum(1)
    union  = area_a[:, None] + area_b[None, :] - inter

    return inter / union.clamp(min=1e-6)


# ─── AP at single IoU threshold ───────────────────────────────────────────────

def compute_ap(
    preds:     list[dict],
    targets:   list[dict],
    iou_thr:   float,
    use_masks: bool = True,
    category:  Optional[int] = None,
) -> float:
    """
    Compute Average Precision at a single IoU threshold.

    Args:
        preds:     list of {boxes, labels, scores, masks} per image
        targets:   list of {boxes, labels, masks} per image
        iou_thr:   IoU threshold (e.g. 0.5)
        use_masks: use mask IoU instead of box IoU
        category:  if set, evaluate only this class

    Returns:
        AP as float
    """
    all_scores, all_tp, all_fp = [], [], []
    n_gt = 0

    for pred, target in zip(preds, targets):
        p_boxes  = pred["boxes"]
        p_labels = pred["labels"]
        p_scores = pred["scores"]
        p_masks  = pred["masks"]

        t_boxes  = target["boxes"]
        t_labels = target["labels"]
        t_masks  = target["masks"]

        # Filter by category
        if category is not None:
            p_keep = p_labels == category
            t_keep = t_labels == category
            p_boxes, p_labels, p_scores = p_boxes[p_keep], p_labels[p_keep], p_scores[p_keep]
            p_masks  = p_masks[p_keep]
            t_boxes, t_labels = t_boxes[t_keep], t_labels[t_keep]
            t_masks  = t_masks[t_keep]

        n_gt += len(t_boxes)

        if len(p_boxes) == 0:
            continue
        if len(t_boxes) == 0:
            all_scores.extend(p_scores.tolist())
            all_tp.extend([0] * len(p_boxes))
            all_fp.extend([1] * len(p_boxes))
            continue

        # Compute IoU matrix
        if use_masks and p_masks.shape[0] > 0 and t_masks.shape[0] > 0:
            # Resize masks to same shape if needed
            if p_masks.shape[-2:] != t_masks.shape[-2:]:
                iou_mat = box_iou(p_boxes, t_boxes)
            else:
                iou_mat = mask_iou(p_masks, t_masks)
        else:
            iou_mat = box_iou(p_boxes, t_boxes)

        matched_gt = set()
        # Sort predictions by score descending
        sorted_idx = p_scores.argsort(descending=True)

        for i in sorted_idx:
            best_iou, best_j = 0.0, -1
            for j in range(len(t_boxes)):
                if j in matched_gt:
                    continue
                iou_val = iou_mat[i, j].item()
                if iou_val > best_iou:
                    best_iou, best_j = iou_val, j

            all_scores.append(p_scores[i].item())
            if best_iou >= iou_thr and best_j >= 0:
                all_tp.append(1)
                all_fp.append(0)
                matched_gt.add(best_j)
            else:
                all_tp.append(0)
                all_fp.append(1)

    if n_gt == 0 or len(all_scores) == 0:
        return 0.0

    # Sort by score descending
    order  = np.argsort(all_scores)[::-1]
    tp_cum = np.cumsum(np.array(all_tp)[order])
    fp_cum = np.cumsum(np.array(all_fp)[order])

    precision = tp_cum / (tp_cum + fp_cum).clip(min=1e-6)
    recall    = tp_cum / max(n_gt, 1)

    # Append sentinel values
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall    = np.concatenate([[0.0], recall,    [1.0]])

    # Monotonically decreasing precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Area under PR curve
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap  = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return float(ap)


# ─── COCO-style metrics ────────────────────────────────────────────────────────

IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05).round(2).tolist()  # 0.50 … 0.95

CATEGORIES = {
    1: "person", 2: "car",    3: "dog",    4: "bicycle",
    5: "cat",    6: "chair",  7: "bottle", 8: "laptop",
}


def compute_coco_metrics(
    preds:     list[dict],
    targets:   list[dict],
    use_masks: bool = True,
) -> dict:
    """
    Compute COCO-style AP/AR metrics.

    Returns dict with:
        AP50, AP75, mAP (AP@[.5:.95])
        per_category_AP50
        AR (average recall @ 100 dets)
    """
    results = {}

    # ── AP at each IoU threshold ──
    ap_per_thr = []
    for thr in IOU_THRESHOLDS:
        ap = compute_ap(preds, targets, iou_thr=thr, use_masks=use_masks)
        ap_per_thr.append(ap)

    results["AP50"]  = ap_per_thr[0]
    results["AP75"]  = ap_per_thr[5]
    results["mAP"]   = float(np.mean(ap_per_thr))
    results["AP_per_threshold"] = {
        f"AP{int(t*100)}": v for t, v in zip(IOU_THRESHOLDS, ap_per_thr)
    }

    # ── Per-category AP50 ──
    per_cat = {}
    for cat_id, cat_name in CATEGORIES.items():
        ap = compute_ap(preds, targets, iou_thr=0.5,
                        use_masks=use_masks, category=cat_id)
        per_cat[cat_name] = round(ap, 4)
    results["per_category_AP50"] = per_cat

    # ── AR @ 100 detections ──
    recalls = []
    for pred, target in zip(preds, targets):
        if len(target["boxes"]) == 0:
            continue
        top100 = pred["scores"].argsort(descending=True)[:100]
        p_b = pred["boxes"][top100]
        t_b = target["boxes"]
        if len(p_b) == 0:
            recalls.append(0.0)
            continue
        iou_mat  = box_iou(p_b, t_b)
        matched  = (iou_mat.max(0).values >= 0.5).sum().item()
        recalls.append(matched / len(t_b))
    results["AR100"] = float(np.mean(recalls)) if recalls else 0.0

    return results


def format_metrics(metrics: dict) -> str:
    lines = [
        "┌─────────────────────────────────────┐",
        "│         Evaluation Metrics          │",
        "├─────────────────────────────────────┤",
        f"│  AP50:  {metrics['AP50']:.4f}                      │",
        f"│  AP75:  {metrics['AP75']:.4f}                      │",
        f"│  mAP:   {metrics['mAP']:.4f}  (AP@[.5:.95])       │",
        f"│  AR100: {metrics['AR100']:.4f}                      │",
        "├─────────────────────────────────────┤",
        "│  Per-Category AP50:                 │",
    ]
    for cat, ap in metrics["per_category_AP50"].items():
        lines.append(f"│    {cat:<12} {ap:.4f}               │")
    lines.append("└─────────────────────────────────────┘")
    return "\n".join(lines)