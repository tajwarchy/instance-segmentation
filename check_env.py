import torch
import torchvision
import cv2
import numpy as np
import skimage
import albumentations
import yaml
import pycocotools

print("=== Environment Check ===")
print(f"PyTorch:        {torch.__version__}")
print(f"Torchvision:    {torchvision.__version__}")
print(f"OpenCV:         {cv2.__version__}")
print(f"NumPy:          {np.__version__}")
print(f"Scikit-image:   {skimage.__version__}")
print(f"Albumentations: {albumentations.__version__}")
print(f"PyYAML:         {yaml.__version__}")

print("\n=== Device Check ===")
print(f"MPS available:  {torch.backends.mps.is_available()}")
print(f"MPS built:      {torch.backends.mps.is_built()}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device:   {device}")

print("\n=== Mask R-CNN Smoke Test ===")
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
model = model.to(device)
model.eval()
dummy = [torch.rand(3, 512, 512).to(device)]
with torch.no_grad():
    out = model(dummy)
print(f"Output keys:    {list(out[0].keys())}")
print(f"Boxes shape:    {out[0]['boxes'].shape}")
print(f"Masks shape:    {out[0]['masks'].shape}")
print("\n✅ All checks passed. Phase 1 complete.")