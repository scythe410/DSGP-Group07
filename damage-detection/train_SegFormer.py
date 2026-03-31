"""
* Run this script on a machine with a GPU
    Without a GPU, training will be extremely slow or may not work at all.

Dataset: CarDD (Car Damage Detection) — 4,000 high-res images
Task:    Binary segmentation — damage (255) vs background (0)
Model:   SegFormer MiT-B2 (HuggingFace Transformers)
"""

import os
import glob
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

from PIL import Image
import albumentations as A
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm

# Edit these to match your correct configurations
CARDD_ROOT   = r"C:\path\to\CarDD"   # CarDD folder
OUTPUT_DIR   = r"C:\path\to\output"  # Where checkpoints and results are saved

EPOCHS           = 30
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 4
BASE_LR          = 6e-5
SAVE_EVERY       = 5
RESUME_CKPT      = None

# device check
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == "cpu":
    print("No GPU detected. Training will be very slow.")
    print("Make sure CUDA is installed and your GPU drivers are up to date.")
else:
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM   : {mem:.1f} GB")

# dataset path
SOD_ROOT = os.path.join(CARDD_ROOT, "CarDD_release", "CarDD_SOD")

SPLITS = {
    "train": {
        "images": os.path.join(SOD_ROOT, "CarDD-TR",  "CarDD-TR-Image"),
        "masks":  os.path.join(SOD_ROOT, "CarDD-TR",  "CarDD-TR-Mask"),
    },
    "val": {
        "images": os.path.join(SOD_ROOT, "CarDD-VAL", "CarDD-VAL-Image"),
        "masks":  os.path.join(SOD_ROOT, "CarDD-VAL", "CarDD-VAL-Mask"),
    },
    "test": {
        "images": os.path.join(SOD_ROOT, "CarDD-TE",  "CarDD-TE-Image"),
        "masks":  os.path.join(SOD_ROOT, "CarDD-TE",  "CarDD-TE-Mask"),
    },
}

# Validate paths
all_ok = True
for split, paths in SPLITS.items():
    for kind, p in paths.items():
        if not os.path.exists(p):
            print(f"Missing: [{split}][{kind}] {p}")
            all_ok = False
assert all_ok, "Fix the missing paths above before continuing."

# Build file lists
FILE_LISTS = {}
print(f"\n{'Split':<8} {'Images':>8} {'Masks':>8}  Match?")
for split, paths in SPLITS.items():
    imgs  = sorted(glob.glob(os.path.join(paths["images"], "*.jpg")) +
                   glob.glob(os.path.join(paths["images"], "*.png")))
    masks = sorted(glob.glob(os.path.join(paths["masks"],  "*.png")))
    match = "Matches" if len(imgs) == len(masks) and len(imgs) > 0 else "Doesn't Match"
    print(f"{split:<8} {len(imgs):>8} {len(masks):>8}  {match}")
    FILE_LISTS[split] = {"images": imgs, "masks": masks}

# dataset class
PROCESSOR = SegformerImageProcessor(
    do_resize=True,
    size={"height": 512, "width": 512},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
    do_reduce_labels=False,
)

TRAIN_AUG = A.Compose([
    A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomRotate90(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.3),
    A.GaussNoise(var_limit=(5, 25), p=0.2),
    A.Blur(blur_limit=3, p=0.2),
    A.CLAHE(p=0.2),
])

VAL_AUG = A.Compose([
    A.Resize(height=512, width=512),
])


class CarDDSODDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=None):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.augment    = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask  = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        if self.augment:
            aug   = self.augment(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        mask_bin = (mask == 255).astype(np.uint8)  # {0, 1}

        encoding = PROCESSOR(
            images=Image.fromarray(image),
            segmentation_maps=Image.fromarray(mask_bin),
            return_tensors="pt",
        )
        pixel_values = encoding["pixel_values"].squeeze(0)   # [3, 512, 512]
        labels       = encoding["labels"].squeeze(0).float() # [128, 128]

        return {"pixel_values": pixel_values, "labels": labels}


train_ds = CarDDSODDataset(FILE_LISTS["train"]["images"], FILE_LISTS["train"]["masks"], TRAIN_AUG)
val_ds   = CarDDSODDataset(FILE_LISTS["val"]["images"],   FILE_LISTS["val"]["masks"],   VAL_AUG)
test_ds  = CarDDSODDataset(FILE_LISTS["test"]["images"],  FILE_LISTS["test"]["masks"],  VAL_AUG)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain : {len(train_ds)} images | Val : {len(val_ds)} | Test : {len(test_ds)}")

# loss function
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        probs  = torch.sigmoid(logits_up).squeeze(1)
        inter  = (probs * targets).sum(dim=(1, 2))
        cardin = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice   = (2.0 * inter + self.smooth) / (cardin + self.smooth)
        return (1.0 - dice).mean()


class HybridBinaryLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice   = BinaryDiceLoss()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return self.bce_w * self.bce(logits_up, targets) + self.dice_w * self.dice(logits, targets)


POS_WEIGHT = torch.tensor([7.0]).to(DEVICE)
criterion  = HybridBinaryLoss(pos_weight=POS_WEIGHT)

# model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2",
    num_labels=1,
    id2label={0: "damage"},
    label2id={"damage": 0},
    ignore_mismatched_sizes=True,
)
model = model.to(DEVICE)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total/1e6:.1f}M")
print(f"Trainable params: {trainable/1e6:.1f}M")

# Quick sanity check
with torch.no_grad():
    dummy = torch.randn(1, 3, 512, 512).to(DEVICE)
    out   = model(pixel_values=dummy)
print(f"Output logits shape : {out.logits.shape}")  # [1, 1, 128, 128]

# optimizer and scheduler
backbone_params = [p for n, p in model.named_parameters() if "segformer.encoder" in n]
head_params     = [p for n, p in model.named_parameters() if "segformer.encoder" not in n]

optimizer = AdamW([
    {"params": backbone_params, "lr": BASE_LR,      "weight_decay": 0.01},
    {"params": head_params,     "lr": BASE_LR * 10, "weight_decay": 0.01},
])

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
scheduler = OneCycleLR(
    optimizer,
    max_lr=[BASE_LR, BASE_LR * 10],
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,
    anneal_strategy="cos",
)
scaler = GradScaler()

# metrics
@torch.no_grad()
def compute_binary_metrics(logits, targets, threshold=0.5):
    logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    preds = (torch.sigmoid(logits_up) >= threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    eps = 1e-7

    precision = (tp / (tp + fp + eps)).item()
    recall    = (tp / (tp + fn + eps)).item()
    beta2     = 0.3

    return {
        "iou":       (tp / (tp + fp + fn + eps)).item(),
        "dice":      (2 * tp / (2 * tp + fp + fn + eps)).item(),
        "pix_acc":   ((tp + tn) / (tp + fp + fn + tn + eps)).item(),
        "precision": precision,
        "recall":    recall,
        "fbeta":     ((1 + beta2) * precision * recall) / (beta2 * precision + recall + eps),
    }

# checkpoint utilities
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_DIR = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, metrics, is_best=False):
    state = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "metrics":         metrics,
    }
    path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt")
    torch.save(state, path)
    print(f"Checkpoint: {path}")

    if is_best:
        model.save_pretrained(BEST_DIR)
        PROCESSOR.save_pretrained(BEST_DIR)
        with open(os.path.join(BEST_DIR, "metrics.json"), "w") as f:
            json.dump({"epoch": epoch, **metrics}, f, indent=2)
        print(f"Best model: {BEST_DIR}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    print(f"Resumed from epoch {state['epoch']}: {path}")
    return state["epoch"], state["metrics"]


START_EPOCH = 0
best_iou    = 0.0

if RESUME_CKPT and os.path.exists(RESUME_CKPT):
    START_EPOCH, prev = load_checkpoint(RESUME_CKPT, model, optimizer, scheduler, scaler)
    best_iou = prev.get("val_iou", 0.0)

# training loop
print("Training SegFormer MiT-B2 (binary damage segmentation)")
print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Effective batch : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"Output dir: {OUTPUT_DIR}")

for epoch in range(START_EPOCH, EPOCHS):
    t0 = time.time()

    # train
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{EPOCHS} [train]", leave=False)):
        pv = batch["pixel_values"].to(DEVICE, non_blocking=True)
        lb = batch["labels"].to(DEVICE, non_blocking=True)

        with autocast():
            out  = model(pixel_values=pv)
            loss = criterion(out.logits, lb) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        train_loss += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    train_loss /= len(train_loader)

    # validate
    model.eval()
    val_loss   = 0.0
    metric_acc = {"iou": 0, "dice": 0, "pix_acc": 0, "precision": 0, "recall": 0, "fbeta": 0}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{EPOCHS} [val]  ", leave=False):
            pv = batch["pixel_values"].to(DEVICE, non_blocking=True)
            lb = batch["labels"].to(DEVICE, non_blocking=True)

            with autocast():
                out  = model(pixel_values=pv)
                loss = criterion(out.logits, lb)

            val_loss += loss.item()
            m = compute_binary_metrics(out.logits, lb)
            for k in metric_acc:
                metric_acc[k] += m[k]

    val_loss /= len(val_loader)
    for k in metric_acc:
        metric_acc[k] /= len(val_loader)

    elapsed = time.time() - t0
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"TrainLoss={train_loss:.4f} | "
          f"ValLoss={val_loss:.4f} | "
          f"IoU={metric_acc['iou']:.4f} | "
          f"Dice={metric_acc['dice']:.4f} | "
          f"Fβ={metric_acc['fbeta']:.4f} | "
          f"{elapsed:.0f}s")

    metrics = {"val_loss": val_loss, "train_loss": train_loss, **metric_acc}
    is_best = metric_acc["iou"] > best_iou
    if is_best:
        best_iou = metric_acc["iou"]
        print(f"New best IoU: {best_iou:.4f}")

    if (epoch + 1) % SAVE_EVERY == 0 or is_best or (epoch + 1) == EPOCHS:
        save_checkpoint(epoch + 1, model, optimizer, scheduler, scaler, metrics, is_best)

print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")

# evaluate
best_model = SegformerForSemanticSegmentation.from_pretrained(BEST_DIR).to(DEVICE)
best_model.eval()
print(f"\nLoaded best model from {BEST_DIR}")

test_metrics_acc = {"iou": 0, "dice": 0, "pix_acc": 0, "precision": 0, "recall": 0, "fbeta": 0}

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        pv = batch["pixel_values"].to(DEVICE)
        lb = batch["labels"].to(DEVICE)
        with autocast():
            out = best_model(pixel_values=pv)
        m = compute_binary_metrics(out.logits, lb)
        for k in test_metrics_acc:
            test_metrics_acc[k] += m[k]

n = len(test_loader)
for k in test_metrics_acc:
    test_metrics_acc[k] /= n

print("TEST SET RESULTS")
for k, v in test_metrics_acc.items():
    print(f"  {k:<12}: {v:.4f}")

results_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(results_path, "w") as f:
    json.dump(test_metrics_acc, f, indent=2)
print(f"Results saved: {results_path}")
