import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_train_transforms(size: int = 512):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.3,
    ))


def get_val_transforms(size: int = 512):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.3,
    ))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class COCOSubsetDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, transforms=None):
        self.img_dir    = Path(img_dir)
        self.transforms = transforms
        self.coco       = COCO(ann_file)
        self.img_ids    = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]

        # ── Load image ──
        img_path = self.img_dir / img_info["file_name"]
        img      = np.array(Image.open(img_path).convert("RGB"))
        h, w     = img.shape[:2]

        # ── Load annotations ──
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for ann in anns:
            # Bounding box — COCO stores as [x, y, w, h]
            x, y, bw, bh = ann["bbox"]
            x2, y2       = x + bw, y + bh

            # Skip degenerate boxes
            if bw < 1 or bh < 1:
                continue

            # Clamp to image bounds
            x  = max(0.0, x)
            y  = max(0.0, y)
            x2 = min(float(w), x2)
            y2 = min(float(h), y2)

            boxes.append([x, y, x2, y2])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

            # Binary mask from COCO RLE / polygon
            masks.append(self.coco.annToMask(ann))

        # ── Handle empty annotations ──
        if len(boxes) == 0:
            boxes   = torch.zeros((0, 4), dtype=torch.float32)
            labels  = torch.zeros((0,),   dtype=torch.int64)
            masks   = torch.zeros((0, h, w), dtype=torch.uint8)
            areas   = torch.zeros((0,),   dtype=torch.float32)
            iscrowd = torch.zeros((0,),   dtype=torch.int64)
        else:
            if self.transforms:
                # Albumentations expects numpy masks as (H, W) per mask
                transformed = self.transforms(
                    image=img,
                    masks=masks,
                    bboxes=boxes,
                    labels=labels,
                )
                img    = transformed["image"]           # (C, H, W) tensor
                masks  = transformed["masks"]           # list of (H, W) ndarrays
                boxes  = transformed["bboxes"]
                labels = transformed["labels"]

                if len(boxes) == 0:
                    h_new = img.shape[1]
                    w_new = img.shape[2]
                    boxes   = torch.zeros((0, 4), dtype=torch.float32)
                    labels  = torch.zeros((0,),   dtype=torch.int64)
                    masks   = torch.zeros((0, h_new, w_new), dtype=torch.uint8)
                    areas   = torch.zeros((0,),   dtype=torch.float32)
                    iscrowd = torch.zeros((0,),   dtype=torch.int64)
                else:
                    boxes   = torch.as_tensor(boxes,  dtype=torch.float32)
                    labels  = torch.as_tensor(labels, dtype=torch.int64)
                    masks   = torch.stack([
                        torch.as_tensor(np.array(m), dtype=torch.uint8) for m in masks
                    ])
                    areas   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            else:
                # No transforms — convert raw numpy to tensors
                img     = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                boxes   = torch.as_tensor(boxes,  dtype=torch.float32)
                labels  = torch.as_tensor(labels, dtype=torch.int64)
                masks   = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
                areas   = torch.as_tensor(areas,  dtype=torch.float32)
                iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "masks":    masks,
            "image_id": torch.tensor([img_id]),
            "area":     areas,
            "iscrowd":  iscrowd,
        }

        return img, target


# ─── Collate ──────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Required because each image may have a different number of instances."""
    return tuple(zip(*batch))