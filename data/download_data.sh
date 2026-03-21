#!/bin/bash

# ─── COCO 2017 Subset Downloader ──────────────────────────────────────────────
# Downloads val2017 images (1GB) and a small train2017 subset
# Full train2017 is 18GB — we use the API to filter, not download everything

set -e

DATA_DIR="coco"
ANN_DIR="$DATA_DIR/annotations"

mkdir -p "$DATA_DIR/train2017"
mkdir -p "$DATA_DIR/val2017"
mkdir -p "$ANN_DIR"

echo "=== Downloading COCO 2017 Annotations ==="
if [ ! -f "$ANN_DIR/instances_train2017.json" ]; then
    curl -L "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
         -o "$DATA_DIR/annotations.zip"
    unzip -q "$DATA_DIR/annotations.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/annotations.zip"
    echo "✅ Annotations downloaded"
else
    echo "⏭️  Annotations already exist, skipping"
fi

echo ""
echo "=== Downloading COCO 2017 Val Images (~1GB) ==="
if [ ! "$(ls -A $DATA_DIR/val2017)" ]; then
    curl -L "http://images.cocodataset.org/zips/val2017.zip" \
         -o "$DATA_DIR/val2017.zip"
    unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/val2017.zip"
    echo "✅ Val images downloaded"
else
    echo "⏭️  Val images already exist, skipping"
fi

echo ""
echo "=== Downloading COCO 2017 Train Images (~18GB) ==="
echo "⚠️  This is large. If you want to skip and use val only, press Ctrl+C now."
echo "    Waiting 5 seconds..."
sleep 5

if [ ! "$(ls -A $DATA_DIR/train2017)" ]; then
    curl -L "http://images.cocodataset.org/zips/train2017.zip" \
         -o "$DATA_DIR/train2017.zip"
    unzip -q "$DATA_DIR/train2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/train2017.zip"
    echo "✅ Train images downloaded"
else
    echo "⏭️  Train images already exist, skipping"
fi

echo ""
echo "✅ COCO download complete. Structure:"
echo "   $DATA_DIR/train2017/     — training images"
echo "   $DATA_DIR/val2017/       — validation images"
echo "   $ANN_DIR/                — annotation JSONs"