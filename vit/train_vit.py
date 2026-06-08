#!/usr/bin/env python
"""Fine-tune a Vision Transformer (ViT) for traffic-light color classification.

Exposes ``train(config)`` for programmatic use (e.g. hyperparameter search)
and a CLI for standalone training.
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME = "google/vit-base-patch16-224"


@dataclass
class TrainConfig:
    data_dir: str = ""
    epochs: int = 30
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 1e-4
    aug_strength: float = 0.3
    num_workers: int = 4
    use_amp: bool = True
    class_weights: bool = False
    checkpoint_dir: str = ""
    results_dir: str = ""
    verbose: bool = True


@dataclass
class TrainResult:
    best_val_acc: float = 0.0
    best_epoch: int = 0
    history: dict = field(default_factory=lambda: {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    })
    checkpoint_path: str = ""
    class_names: list = field(default_factory=list)
    elapsed: float = 0.0


def build_model(num_classes: int, class_names: list[str]) -> ViTForImageClassification:
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True,
    )
    model.config.id2label = {i: n for i, n in enumerate(class_names)}
    model.config.label2id = {n: i for i, n in enumerate(class_names)}
    return model


def get_processor() -> ViTImageProcessor:
    return ViTImageProcessor.from_pretrained(MODEL_NAME)


def build_transforms(processor: ViTImageProcessor, aug_strength: float):
    mean, std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    s = aug_strength

    train_tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(brightness=s, contrast=s, saturation=s, hue=min(s * 0.1, 0.5)),
        transforms.RandomRotation(int(15 * s / 0.3)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.1),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfm, eval_tfm


def run_epoch(model, loader, criterion, optimizer, device, scaler, is_train: bool) -> tuple[float, float]:
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if is_train and scaler is not None:
                with autocast():
                    outputs = model(pixel_values=images, labels=targets)
                    loss = outputs.loss if outputs.loss is not None else criterion(outputs.logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                with autocast(enabled=(not is_train)):
                    outputs = model(pixel_values=images, labels=targets)
                    loss = outputs.loss if outputs.loss is not None else criterion(outputs.logits, targets)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (outputs.logits.argmax(dim=1) == targets).sum().item()
        total_count += bs

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def plot_curves(history: dict, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="train", linewidth=1.5)
    ax1.plot(history["val_loss"], label="val", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="train", linewidth=1.5)
    ax2.plot(history["val_acc"], label="val", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(dataset: datasets.ImageFolder) -> torch.utils.data.WeightedRandomSampler:
    counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        counts[label] += 1
    sample_weights = [1.0 / counts[label] for _, label in dataset.samples]
    return torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train(config: TrainConfig) -> TrainResult:
    """Run a single training experiment. Returns TrainResult with metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = get_processor()
    train_tfm, eval_tfm = build_transforms(processor, config.aug_strength)

    data_dir = Path(config.data_dir)
    train_set = datasets.ImageFolder(data_dir / "train", transform=train_tfm)
    val_set = datasets.ImageFolder(data_dir / "val", transform=eval_tfm)
    class_names = train_set.classes

    if config.class_weights:
        cw = compute_class_weights(train_set).to(device)
        if config.verbose:
            print(f"  class_weights: {dict(zip(class_names, [f'{w:.2f}' for w in cw]))}")
        sampler = make_weighted_sampler(train_set)
        train_loader = DataLoader(
            train_set, batch_size=config.batch_size, sampler=sampler,
            num_workers=config.num_workers, pin_memory=True, drop_last=True,
        )
    else:
        cw = None
        train_loader = DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True, drop_last=True,
        )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    model = build_model(len(class_names), class_names).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    use_amp = config.use_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    result = TrainResult(class_names=class_names)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
    results_dir = Path(config.results_dir) if config.results_dir else None

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler, is_train=False)
        scheduler.step()

        result.history["train_loss"].append(train_loss)
        result.history["train_acc"].append(train_acc)
        result.history["val_loss"].append(val_loss)
        result.history["val_acc"].append(val_acc)

        if config.verbose:
            print(
                f"  epoch {epoch:03d}/{config.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if val_acc >= result.best_val_acc:
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            if checkpoint_dir:
                prev_path = result.checkpoint_path
                ckpt_path = checkpoint_dir / f"vit_trafficlight_epoch{epoch:03d}.pt"
                result.checkpoint_path = str(ckpt_path)
                torch.save({
                    "model": model.state_dict(),
                    "config": model.config.to_dict(),
                    "classes": class_names,
                    "model_name": MODEL_NAME,
                    "image_size": processor.size["height"],
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "processor": processor.to_dict(),
                }, ckpt_path)
                if prev_path and Path(prev_path).exists() and prev_path != str(ckpt_path):
                    Path(prev_path).unlink()

    result.elapsed = time.time() - t0

    if results_dir:
        csv_path = results_dir / "training_history.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            writer.writeheader()
            for i in range(config.epochs):
                writer.writerow({
                    "epoch": i + 1,
                    "train_loss": f"{result.history['train_loss'][i]:.4f}",
                    "train_acc": f"{result.history['train_acc'][i]:.4f}",
                    "val_loss": f"{result.history['val_loss'][i]:.4f}",
                    "val_acc": f"{result.history['val_acc'][i]:.4f}",
                })
        plot_curves(result.history, results_dir / "training_curves.png")

        meta_path = Path(config.checkpoint_dir) / "vit_trafficlight.json"
        meta_path.write_text(json.dumps({
            "classes": class_names,
            "model_name": MODEL_NAME,
            "best_val_acc": result.best_val_acc,
            "best_checkpoint": result.checkpoint_path,
            "epochs": config.epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
            "aug_strength": config.aug_strength,
        }, indent=2), encoding="utf-8")

    return result


def main() -> None:
    vit_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fine-tune ViT for traffic-light classification.")
    parser.add_argument("--data", type=Path, default=vit_dir / "dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aug-strength", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--class-weights", action="store_true", help="Use class weights + weighted sampler for imbalanced data.")
    parser.add_argument("--checkpoint-dir", type=Path, default=vit_dir / "checkpoints")
    parser.add_argument("--results-dir", type=Path, default=vit_dir / "results")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=str(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aug_strength=args.aug_strength,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        class_weights=args.class_weights,
        checkpoint_dir=str(args.checkpoint_dir),
        results_dir=str(args.results_dir),
    )

    result = train(cfg)
    print(f"\nBest val_acc={result.best_val_acc:.3f} (epoch {result.best_epoch})")
    print(f"Elapsed: {result.elapsed:.1f}s")
    print(f"Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
