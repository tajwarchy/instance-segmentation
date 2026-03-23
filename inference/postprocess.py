import torch
import torch.nn.functional as F
from torchvision.ops import nms


def postprocess_outputs(
    outputs:     list[dict],
    score_thr:   float = 0.5,
    mask_thr:    float = 0.5,
    nms_iou_thr: float = 0.5,
    img_size:    tuple = (512, 512),
) -> list[dict]:
    """
    Post-process raw Mask R-CNN outputs.

    Steps:
        1. Score thresholding
        2. NMS across all classes
        3. Mask binarization + resize to full image size

    Args:
        outputs:     raw model outputs (list of dicts per image)
        score_thr:   minimum confidence score
        mask_thr:    threshold to binarize soft masks
        nms_iou_thr: IoU threshold for NMS
        img_size:    (H, W) of the original image

    Returns:
        list of cleaned dicts: {boxes, labels, scores, masks, colors}
    """
    results = []

    for out in outputs:
        boxes  = out["boxes"].cpu()
        labels = out["labels"].cpu()
        scores = out["scores"].cpu()
        masks  = out["masks"].cpu()   # (N, 1, H, W) soft masks

        # ── Score threshold ──
        keep   = scores >= score_thr
        boxes  = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        masks  = masks[keep]

        if len(boxes) == 0:
            results.append({
                "boxes": boxes, "labels": labels,
                "scores": scores, "masks": masks,
            })
            continue

        # ── NMS ──
        keep   = nms(boxes, scores, nms_iou_thr)
        boxes  = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        masks  = masks[keep]

        # ── Binarize + resize masks to img_size ──
        masks = F.interpolate(
            masks, size=img_size, mode="bilinear", align_corners=False
        )
        masks = (masks[:, 0] > mask_thr)   # (N, H, W) bool

        results.append({
            "boxes":  boxes,
            "labels": labels,
            "scores": scores,
            "masks":  masks,
        })

    return results