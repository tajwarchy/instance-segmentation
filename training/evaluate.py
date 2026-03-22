import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_one_epoch(
    model:      torch.nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    score_thr:  float = 0.05,   # low threshold — metrics.py will filter
) -> tuple[list, list]:
    """
    Run inference on the entire validation set.

    Returns:
        predictions: list of dicts {boxes, labels, scores, masks} per image
        targets:     list of ground truth dicts per image
    """
    model.eval()
    all_preds   = []
    all_targets = []

    for images, targets in tqdm(loader, desc="  Evaluating", leave=False):
        images  = [img.to(device) for img in images]
        outputs = model(images)

        for out in outputs:
            all_preds.append({
                "boxes":  out["boxes"].cpu(),
                "labels": out["labels"].cpu(),
                "scores": out["scores"].cpu(),
                "masks":  (out["masks"][:, 0] > 0.5).cpu(),  # binarized (N, H, W)
            })

        for t in targets:
            all_targets.append({
                "boxes":    t["boxes"].cpu(),
                "labels":   t["labels"].cpu(),
                "masks":    t["masks"].cpu(),
                "image_id": t["image_id"].cpu(),
            })

    return all_preds, all_targets