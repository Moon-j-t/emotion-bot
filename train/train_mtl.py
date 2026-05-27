"""
Multi-Task Learning 학습 스크립트 (표정 CE + 나이 MSE).

데이터셋 CSV 형식 (헤더 포함):
  image_path,expression_id,age
  faces/001.jpg,4,32.0

expression_id: config.EXPRESSION_LABELS 인덱스 (0~6)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from models.mtl_model import FaceAttributeMTL  # noqa: E402
from models.preprocess import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def _build_transforms(input_size: int, train: bool) -> transforms.Compose:
    steps = [transforms.Resize((input_size, input_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


class FaceAttributeDataset(Dataset):
    def __init__(
        self, csv_path: Path, root: Path, input_size: int = 224, train: bool = True
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.root = root
        self.transform = _build_transforms(input_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root / str(row["image_path"])
        image = Image.open(img_path).convert("RGB")
        x = self.transform(image)
        y_expr = int(row["expression_id"])
        y_age = float(row["age"])
        return x, y_expr, y_age


def train_one_epoch(model, loader, optimizer, device, age_weight: float):
    model.train()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total_loss = 0.0

    for images, expr_ids, ages in loader:
        images = images.to(device)
        expr_ids = expr_ids.to(device)
        ages = ages.to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        expr_logits, pred_ages = model(images)
        loss_expr = ce(expr_logits, expr_ids)
        loss_age = mse(pred_ages, ages)
        loss = loss_expr + age_weight * loss_age
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, device, age_weight: float):
    model.eval()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total_loss = 0.0

    for images, expr_ids, ages in loader:
        images = images.to(device)
        expr_ids = expr_ids.to(device)
        ages = ages.to(device, dtype=torch.float32)
        expr_logits, pred_ages = model(images)
        loss = ce(expr_logits, expr_ids) + age_weight * mse(pred_ages, ages)
        total_loss += float(loss.item()) * images.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--age-weight", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--output", type=Path, default=config.MTL_WEIGHTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceAttributeMTL(
        num_expressions=config.NUM_EXPRESSIONS,
        variant=config.MOBILENET_VARIANT,
        pretrained_backbone=True,
        dropout=args.dropout,
    ).to(device)

    train_ds = FaceAttributeDataset(
        args.train_csv, args.data_root, config.INPUT_SIZE, train=True
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loader = None
    if args.val_csv:
        val_ds = FaceAttributeDataset(
            args.val_csv, args.data_root, config.INPUT_SIZE, train=False
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.age_weight)
        msg = f"epoch {epoch}/{args.epochs} train_loss={train_loss:.4f}"

        if val_loader:
            val_loss = evaluate(model, val_loader, device, args.age_weight)
            msg += f" val_loss={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), args.output)
                msg += " [saved]"
            else:
                epochs_no_improve += 1
                msg += f" (patience {epochs_no_improve}/{args.patience})"
        else:
            torch.save(model.state_dict(), args.output)

        print(msg)
        if val_loader and epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
