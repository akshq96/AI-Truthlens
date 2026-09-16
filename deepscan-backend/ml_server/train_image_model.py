"""
Fine-tune the DeepScan image deepfake-detection model.

Root-causes this addresses (found via calibrate_image_model.py against two
independent Kaggle real/fake face datasets, pooled n=3000):
  - Current model_v3.pth checkpoint tops out at ~58% balanced accuracy no
    matter what decision threshold is used (random guessing = 50%).
  - ~60% of genuine real photos are misclassified as fake at the shipped
    threshold.

Fix: re-initialize the same Xception architecture used in image_server.py
from ImageNet-pretrained weights (the shipped checkpoint appears to have been
trained from scratch or undertrained) and fine-tune on labeled real/fake face
data, model-selecting on a held-out validation split, then confirming
generalization on a *completely separate* dataset the model never trains on.

Usage:
  python train_image_model.py \
      --train-real calibration_data/rvf10k/train/real \
      --train-fake calibration_data/rvf10k/train/fake \
      --val-real   calibration_data/rvf10k/valid/real \
      --val-fake   calibration_data/rvf10k/valid/fake \
      --test-real  calibration_data2/real \
      --test-fake  calibration_data2/fake \
      --epochs 8 --batch-size 32 --lr 1e-4 \
      --out trained_models/image_prediction_model/model_v3_finetuned.pth
"""

import argparse
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def list_images(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(full)
    return files


class RealFakeFaceDataset(Dataset):
    """label = 1.0 for real, 0.0 for fake (matches server's positive_class='real')."""

    def __init__(self, real_dir: str, fake_dir: str, image_size: int, train: bool):
        self.samples: List[Tuple[str, float]] = []
        for p in list_images(real_dir):
            self.samples.append((p, 1.0))
        for p in list_images(fake_dir):
            self.samples.append((p, 0.0))
        random.Random(42).shuffle(self.samples)

        if train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (64, 64))
        return self.transform(img), torch.tensor([label], dtype=torch.float32)


class XceptionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.BatchNorm1d(2048)
        self.l = nn.Linear(2048, 512)
        self.b2 = nn.BatchNorm1d(512)
        self.o = nn.Linear(512, 1)

    def forward(self, x):
        x = self.b1(x)
        x = self.l(x)
        x = F.relu(x, inplace=False)
        x = self.b2(x)
        x = self.o(x)
        return x


class ImageTorchModel(nn.Module):
    """Same architecture as image_server._ImageTorchModel — state_dict-compatible."""

    def __init__(self, pretrained: bool):
        super().__init__()
        base_model = ptcv_get_model("xception", pretrained=pretrained)
        base_model.features.final_block.pool = nn.AdaptiveAvgPool2d(1)
        self.base = nn.Sequential(base_model.features)
        self.h1 = XceptionHead()

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = self.h1(x)
        return x


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_probs = []
    all_labels = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
        all_labels.append(y.numpy().flatten())
    y_prob_real = np.concatenate(all_probs)
    y_true_real = np.concatenate(all_labels)

    y_pred_real = (y_prob_real >= 0.5).astype(np.int32)
    y_true_i = y_true_real.astype(np.int32)

    tp = int(np.sum((y_true_i == 1) & (y_pred_real == 1)))
    tn = int(np.sum((y_true_i == 0) & (y_pred_real == 0)))
    fp = int(np.sum((y_true_i == 0) & (y_pred_real == 1)))
    fn = int(np.sum((y_true_i == 1) & (y_pred_real == 0)))

    tpr = tp / (tp + fn) if (tp + fn) else 0.0  # real correctly kept real
    tnr = tn / (tn + fp) if (tn + fp) else 0.0  # fake correctly kept fake
    bal_acc = (tpr + tnr) / 2.0
    acc = (tp + tn) / max(1, y_true_i.size)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "real_kept_real_rate": float(tpr),
        "fake_kept_fake_rate": float(tnr),
        "n": int(y_true_i.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-real", required=True)
    ap.add_argument("--train-fake", required=True)
    ap.add_argument("--val-real", required=True)
    ap.add_argument("--val-fake", required=True)
    ap.add_argument("--test-real", required=False)
    ap.add_argument("--test-fake", required=False)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrained-init", action="store_true", default=True)
    args = ap.parse_args()

    device = pick_device()
    print(f"Using device: {device}")

    train_ds = RealFakeFaceDataset(args.train_real, args.train_fake, args.image_size, train=True)
    val_ds = RealFakeFaceDataset(args.val_real, args.val_fake, args.image_size, train=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_loader = None
    if args.test_real and args.test_fake:
        test_ds = RealFakeFaceDataset(args.test_real, args.test_fake, args.image_size, train=False)
        print(f"Held-out OOD test samples: {len(test_ds)}")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ImageTorchModel(pretrained=args.pretrained_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_bal_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={running_loss / max(1, n_batches):.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"real_kept={val_metrics['real_kept_real_rate']:.4f} fake_kept={val_metrics['fake_kept_fake_rate']:.4f} "
            f"({dt:.1f}s)"
        )

        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break

    if best_state is None:
        raise SystemExit("Training produced no valid checkpoint")

    model.load_state_dict(best_state)
    print(f"\nBest val balanced_accuracy: {best_val_bal_acc:.4f}")

    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, device)
        print(f"Held-out OOD test metrics: {test_metrics}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(best_state, args.out)
    print(f"Saved fine-tuned checkpoint to: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
