#!/usr/bin/env python
import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += batch_size

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def save_checkpoint(checkpoint: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        for idx in range(1, 1000):
            candidate = out_path.with_name(f"{out_path.stem}_{idx:03d}{out_path.suffix}")
            if not candidate.exists():
                out_path = candidate
                break
    try:
        with out_path.open("wb") as f:
            torch.save(checkpoint, f)
        return out_path
    except OSError:
        pass

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt", dir=out_path.parent) as tmp:
        torch.save(checkpoint, tmp)
        tmp_path = Path(tmp.name)
    tmp_path.replace(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 for traffic-light color classification.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights. May download weights on first run.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "checkpoints" / "resnet18_trafficlight.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = datasets.ImageFolder(args.data / "train", transform=train_tfms)
    val_set = datasets.ImageFolder(args.data / "val", transform=eval_tfms)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(len(train_set.classes), args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_checkpoint_path = args.out
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model": model.state_dict(),
                "classes": train_set.classes,
                "image_size": 224,
                "pretrained": args.pretrained,
                "epoch": epoch,
                "val_acc": val_acc,
            }
            epoch_out = args.out.with_name(f"{args.out.stem}_epoch{epoch:03d}{args.out.suffix}")
            best_checkpoint_path = save_checkpoint(checkpoint, epoch_out)

    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"classes": train_set.classes, "best_val_acc": best_val_acc, "checkpoint": str(best_checkpoint_path)}, indent=2),
        encoding="utf-8",
    )
    print(f"Best val_acc={best_val_acc:.3f}")
    print(f"Saved checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
