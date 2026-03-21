import os
import json
import shutil
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────

CATEGORIES = ["person", "car", "dog", "bicycle", "cat", "chair", "bottle", "laptop"]

DATA_DIR   = Path("data/coco")
ANN_DIR    = DATA_DIR / "annotations"
TRAIN_DIR  = DATA_DIR / "train2017"
VAL_DIR    = DATA_DIR / "val2017"

MAX_TRAIN  = 2000
MAX_VAL    = 500


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_coco_json(path: Path) -> dict:
    print(f"Loading {path} ...")
    with open(path) as f:
        return json.load(f)


def filter_coco(coco: dict, target_cats: list[str], max_images: int) -> dict:
    # Map category name → original COCO id
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    target_ids     = {cat_name_to_id[n] for n in target_cats if n in cat_name_to_id}
    missing        = [n for n in target_cats if n not in cat_name_to_id]
    if missing:
        print(f"⚠️  Categories not found in COCO: {missing}")

    # Remap to new contiguous ids (1-indexed, 0 = background)
    old_to_new = {old: new for new, old in enumerate(sorted(target_ids), start=1)}
    new_categories = [
        {"id": old_to_new[c["id"]], "name": c["name"], "supercategory": c["supercategory"]}
        for c in coco["categories"] if c["id"] in target_ids
    ]

    # Filter annotations
    anns = [a for a in coco["annotations"] if a["category_id"] in target_ids]

    # Collect valid image ids (must have at least one annotation)
    img_ids_with_anns = list(dict.fromkeys(a["image_id"] for a in anns))  # preserve order, dedupe
    img_ids_subset    = img_ids_with_anns[:max_images]
    img_id_set        = set(img_ids_subset)

    # Filter images
    images = [img for img in coco["images"] if img["id"] in img_id_set]

    # Filter annotations to subset images + remap category ids
    anns_subset = []
    for a in anns:
        if a["image_id"] in img_id_set:
            a = dict(a)
            a["category_id"] = old_to_new[a["category_id"]]
            anns_subset.append(a)

    print(f"  Images:      {len(images)}")
    print(f"  Annotations: {len(anns_subset)}")
    print(f"  Categories:  {[c['name'] for c in new_categories]}")

    return {
        "images":      images,
        "annotations": anns_subset,
        "categories":  new_categories,
    }


def download_images(images: list[dict], img_dir: Path, split: str):
    """Download only the images we need (for train split)."""
    img_dir.mkdir(parents=True, exist_ok=True)
    to_download = [img for img in images if not (img_dir / img["file_name"]).exists()]

    if not to_download:
        print(f"  ⏭️  All {split} images already present, skipping download.")
        return

    print(f"  Downloading {len(to_download)} {split} images ...")
    base_url = f"http://images.cocodataset.org/{split}/"

    for img in tqdm(to_download, desc=f"  {split}"):
        url  = base_url + img["file_name"]
        dest = img_dir / img["file_name"]
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            dest.write_bytes(r.content)
        except Exception as e:
            print(f"  ⚠️  Failed to download {img['file_name']}: {e}")


def save_subset_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  ✅ Saved → {path}  ({path.stat().st_size / 1024:.1f} KB)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    # ── Train ──
    print("\n=== Processing Train Split ===")
    train_coco    = load_coco_json(ANN_DIR / "instances_train2017.json")
    train_subset  = filter_coco(train_coco, CATEGORIES, MAX_TRAIN)
    del train_coco  # free RAM

    print("\n  Downloading missing train images ...")
    download_images(train_subset["images"], TRAIN_DIR, "train2017")
    save_subset_json(train_subset, ANN_DIR / "subset_train2017.json")

    # ── Val ──
    print("\n=== Processing Val Split ===")
    val_coco   = load_coco_json(ANN_DIR / "instances_val2017.json")
    val_subset = filter_coco(val_coco, CATEGORIES, MAX_VAL)
    del val_coco

    # Val images should already be on disk from download_data.sh
    missing_val = [
        img for img in val_subset["images"]
        if not (VAL_DIR / img["file_name"]).exists()
    ]
    if missing_val:
        print(f"\n  ⚠️  {len(missing_val)} val images missing — downloading ...")
        download_images(val_subset["images"], VAL_DIR, "val2017")
    else:
        print("\n  ⏭️  All val images present.")

    save_subset_json(val_subset, ANN_DIR / "subset_val2017.json")

    print("\n✅ Dataset preparation complete.")
    print(f"   Train: {len(train_subset['images'])} images, {len(train_subset['annotations'])} annotations")
    print(f"   Val:   {len(val_subset['images'])} images, {len(val_subset['annotations'])} annotations")
    print(f"   Categories ({len(train_subset['categories'])}): {[c['name'] for c in train_subset['categories']]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare COCO subset for Mask R-CNN training")
    args = parser.parse_args()
    main(args)