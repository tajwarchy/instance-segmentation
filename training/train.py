import os
import sys
import json
import yaml
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coco_dataset import COCOSubsetDataset, get_train_transforms, get_val_transforms, collate_fn
from models.mask_rcnn import build_model, freeze_backbone, unfreeze_backbone, save_checkpoint, load_checkpoint, count_parameters
from training.evaluate import evaluate_one_epoch
from training.metrics import compute_coco_metrics, format_metrics
from training.visualization import save_prediction_grid


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    # MPS crashes on Mask R-CNN training pass (Metal command buffer error)
    # MPS is reserved for inference only
    return torch.device("cpu")


# ─── Training step ────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader:    DataLoader,
    device:    torch.device,
    epoch:     int,
    cpu_device: torch.device,
) -> dict:
    """
    Run one training epoch.
    NOTE: Forward pass on MPS, loss backward on CPU to avoid MPS OOM.
    Returns dict of mean losses.
    """
    model.train()
    loss_keys  = ["loss_classifier", "loss_box_reg", "loss_mask",
                  "loss_objectness", "loss_rpn_box_reg"]
    epoch_losses = {k: 0.0 for k in loss_keys}
    n_batches    = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch}", leave=False)
    for images, targets in pbar:
        # Move to device
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            loss_dict = model(images, targets)
        except RuntimeError as e:
            if "MPS" in str(e) or "metal" in str(e).lower():
                # MPS OOM — skip this batch
                print(f"\n  ⚠️  MPS error on batch, skipping. ({e})")
                continue
            raise e

        # Sum losses and backprop
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k in loss_keys:
            if k in loss_dict:
                epoch_losses[k] += loss_dict[k].item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss.item():.3f}",
            "mask": f"{loss_dict.get('loss_mask', torch.tensor(0)).item():.3f}",
        })

    if n_batches == 0:
        return epoch_losses

    return {k: v / n_batches for k, v in epoch_losses.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    cfg     = load_config(args.config)
    device  = get_device()
    cpu     = torch.device("cpu")

    print(f"\n{'='*50}")
    print(f"  Mask R-CNN Training")
    print(f"  Device:     {device}")
    print(f"  Config:     {args.config}")
    print(f"{'='*50}\n")

    # ── Datasets ──
    print("Loading datasets ...")
    train_ds = COCOSubsetDataset(
        img_dir  = cfg["dataset"]["train_images"],
        ann_file = cfg["dataset"]["train_ann"],
        transforms = get_train_transforms(cfg["model"]["min_size"]),
    )
    val_ds = COCOSubsetDataset(
        img_dir  = cfg["dataset"]["val_images"],
        ann_file = cfg["dataset"]["val_ann"],
        transforms = get_val_transforms(cfg["model"]["min_size"]),
    )
    print(f"  Train: {len(train_ds)} images")
    print(f"  Val:   {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["training"]["num_workers"],
        pin_memory  = False,
        collate_fn  = collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = cfg["training"]["num_workers"],
        pin_memory  = False,
        collate_fn  = collate_fn,
    )

    # ── Model ──
    print("\nBuilding model ...")
    model = build_model(
        num_classes = cfg["model"]["num_classes"],
        pretrained  = cfg["model"]["pretrained"],
        min_size    = cfg["model"]["min_size"],
        max_size    = cfg["model"]["max_size"],
        trainable_backbone_layers = 0,   # Phase A: frozen
    )
    model = model.to(device)

    params = count_parameters(model)
    print(f"  Trainable params: {params['trainable']:,} / {params['total']:,}")

    # ── Optimizer ──
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr           = opt_cfg["lr"],
        momentum     = opt_cfg["momentum"],
        weight_decay = opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = cfg["training"]["scheduler"]["step_size"],
        gamma     = cfg["training"]["scheduler"]["gamma"],
    )

    # ── Resume ──
    start_epoch   = 1
    best_ap50     = 0.0
    history       = []

    if args.resume:
        ckpt = load_checkpoint(model, args.resume, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_ap50   = ckpt.get("metrics", {}).get("AP50", 0.0)
        print(f"  Resuming from epoch {start_epoch}")

    # ── Training loop ──
    total_epochs  = cfg["training"]["epochs"]
    phase_a_end   = cfg["training"]["phase_a_epochs"]
    ckpt_dir      = Path(cfg["checkpointing"]["save_dir"])
    save_every    = cfg["checkpointing"]["save_every_n_epochs"]

    print(f"\nStarting training for {total_epochs} epochs ...")
    print(f"  Phase A (frozen backbone):   epochs 1–{phase_a_end}")
    print(f"  Phase B (full fine-tune):    epochs {phase_a_end+1}–{total_epochs}\n")

    for epoch in range(start_epoch, total_epochs + 1):

        # ── Phase transition ──
        if epoch == phase_a_end + 1:
            print("\n── Phase B: unfreezing backbone ──")
            unfreeze_backbone(model)
            # Reset optimizer with all params + lower LR
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr           = cfg["training"]["phase_b_lr"],
                momentum     = opt_cfg["momentum"],
                weight_decay = opt_cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size = cfg["training"]["scheduler"]["step_size"],
                gamma     = cfg["training"]["scheduler"]["gamma"],
            )

        print(f"\nEpoch {epoch}/{total_epochs}  "
              f"[{'A' if epoch <= phase_a_end else 'B'}]  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # ── Train ──
        train_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch, cpu
        )
        total = sum(train_losses.values())
        print(f"  Train loss: {total:.4f}  "
              f"(cls={train_losses['loss_classifier']:.3f}  "
              f"box={train_losses['loss_box_reg']:.3f}  "
              f"mask={train_losses['loss_mask']:.3f})")

        scheduler.step()

        # ── Evaluate every 2 epochs ──
        if epoch % 2 == 0 or epoch == total_epochs:
            print("  Evaluating ...")
            preds, targets_eval = evaluate_one_epoch(model, val_loader, device)
            metrics = compute_coco_metrics(preds, targets_eval)
            print(format_metrics(metrics))

            # Save prediction grid
            val_imgs = [val_ds[i][0] for i in range(min(4, len(val_ds)))]
            save_prediction_grid(
                val_imgs, preds[:4],
                save_path=f"results/visualizations/epoch_{epoch:02d}.jpg",
            )

            # Save best model
            if metrics["AP50"] > best_ap50:
                best_ap50 = metrics["AP50"]
                save_checkpoint(
                    model,
                    path    = f"weights/mask_rcnn_best.pth",
                    epoch   = epoch,
                    metrics = metrics,
                )
                print(f"  ⭐ New best AP50: {best_ap50:.4f}")
        else:
            metrics = {}

        # ── Periodic checkpoint ──
        if epoch % save_every == 0:
            save_checkpoint(
                model,
                path    = str(ckpt_dir / f"epoch_{epoch:02d}.pth"),
                epoch   = epoch,
                metrics = metrics,
            )

        # ── Log history ──
        history.append({
            "epoch":       epoch,
            "train_losses": train_losses,
            "metrics":     metrics,
            "lr":          optimizer.param_groups[0]["lr"],
        })
        with open("results/metrics/training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Training complete!")
    print(f"  Best AP50: {best_ap50:.4f}")
    print(f"  Weights:   weights/mask_rcnn_best.pth")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mask_rcnn_config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args)