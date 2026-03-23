# Project 4.2 — Instance Segmentation with Mask R-CNN

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Platform](https://img.shields.io/badge/Platform-macOS%20M1-lightgrey)
![AP50](https://img.shields.io/badge/AP50-52.3%25-brightgreen)
![mAP](https://img.shields.io/badge/mAP-25.2%25-green)

A research-quality instance segmentation system built on **Mask R-CNN**
(ResNet-50 + FPN), fine-tuned on a COCO subset across 8 categories.
Each detected object is segmented at the pixel level and rendered with
a unique color — in real time via webcam and on video files.

---

## Demo

### Video Inference
![Video Demo](demo_video.gif)


---

## Results

| Metric        | Score   |
|---------------|---------|
| AP50          | 52.3%   |
| AP75          | 22.0%   |
| mAP@[.5:.95]  | 25.2%   |
| AR@100        | 83.1%   |

### Per-Category AP50

| Category | AP50  |
|----------|-------|
| bicycle  | 82.1% |
| cat      | 71.1% |
| laptop   | 59.3% |
| dog      | 57.0% |
| person   | 55.1% |
| car      | 40.5% |
| chair    | 35.6% |
| bottle   | 29.8% |

---

## Architecture
```
Input Image
    │
    ▼
ResNet-50 + FPN Backbone   ← pretrained ImageNet weights
    │
    ▼
Region Proposal Network (RPN)
    │
    ▼
RoI Align
    │
    ├──► Box Head (FastRCNNPredictor)  → class + bounding box
    │
    └──► Mask Head (MaskRCNNPredictor) → 28×28 binary mask per instance
```

**Training strategy:**
- Phase A (epochs 1–5): frozen backbone, heads only
- Phase B (epochs 6–15): full fine-tune with reduced LR
- Dataset: COCO 2017 subset — 2000 train / 500 val images, 8 categories
- Hardware: Apple M1 MacBook Air (CPU training, ~10 hours)

---

## Setup
```bash
git clone https://github.com/tajwarchy/instance-segmentation.git
cd instance-segmentation

python3.10 -m venv instance-seg
source instance-seg/bin/activate
pip install -r requirements.txt
```

---

## Dataset Preparation
```bash
# Download COCO annotations + val images
bash data/download_data.sh

# Build filtered subset (2000 train / 500 val, 8 categories)
python data/prepare_coco.py

# Verify dataset visually
python data/verify_dataset.py
```

---

## Training
```bash
# Full training run (~10 hours on M1 CPU)
caffeinate -i python training/train.py

# Resume from checkpoint
python training/train.py --resume results/model_checkpoints/epoch_06.pth
```

---

## Evaluation & Metrics
```bash
# Plot training curves and AP metrics
python results/metrics/plot_metrics.py

# Box AP vs Mask AP comparison
python results/metrics/compare_box_vs_mask.py
```

---

## Inference

### Single image or folder
```bash
python inference/predict.py --input path/to/image.jpg
python inference/predict.py --input path/to/folder/ --threshold 0.6
```

### Video file
```bash
python inference/video_inference.py --input path/to/video.mp4
python inference/video_inference.py --input path/to/video.mp4 --skip 2
```

### Real-time webcam
```bash
python inference/webcam_segmenter.py
```

**Webcam controls:**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Screenshot |
| `T` | Toggle mask / box-only mode |
| `R` | Toggle report overlay |
| `+` / `-` | Raise / lower score threshold |

---

## Project Structure
```
instance-segmentation/
├── data/
│   ├── coco_dataset.py       # Dataset class + transforms
│   ├── prepare_coco.py       # Filter + download COCO subset
│   └── download_data.sh      # Download annotations + val images
├── models/
│   ├── mask_rcnn.py          # Model builder + checkpoint utils
│   ├── backbone_selection.py # ResNet-50 + FPN backbone
│   └── heads.py              # Box + mask head replacements
├── training/
│   ├── train.py              # Two-phase training loop
│   ├── evaluate.py           # Validation inference
│   ├── metrics.py            # Custom AP/AR implementation
│   └── visualization.py      # Instance rendering utilities
├── inference/
│   ├── predict.py            # Image / folder inference
│   ├── webcam_segmenter.py   # Real-time webcam pipeline
│   ├── video_inference.py    # Video file inference
│   └── postprocess.py        # NMS + mask postprocessing
├── configs/
│   └── mask_rcnn_config.yaml
├── results/
│   ├── metrics/              # AP curves, training report, JSON
│   └── visualizations/       # Prediction grids, GIF demos
└── weights/
    └── mask_rcnn_best.pth    # Best checkpoint (AP50=52.3%)
```

---

## Free Resources

- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [torchvision Mask R-CNN](https://pytorch.org/vision/stable/models/mask_rcnn.html)
- [COCO Dataset](https://cocodataset.org/)