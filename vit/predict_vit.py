#!/usr/bin/env python
"""Predict traffic-light color on YOLO boxes using a fine-tuned ViT.

Outputs:
  - predictions.csv          per-box ground truth vs prediction
  - annotated images         bounding boxes drawn on original images
  - confusion_matrix.png     classification confusion matrix
  - per_class_accuracy.png   bar chart of accuracy per class
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor


COLORS = {"red": (255, 0, 0), "yellow": (255, 210, 0), "green": (0, 220, 80)}


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, cx, cy, w, h = parts
            labels.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return labels


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def yolo_to_xyxy(label, image_w: int, image_h: int, padding: float) -> tuple[int, int, int, int]:
    _, cx, cy, w, h = label
    box_w, box_h = w * image_w, h * image_h
    center_x, center_y = cx * image_w, cy * image_h
    pad_x, pad_y = box_w * padding, box_h * padding
    left = max(0, int(round(center_x - box_w / 2 - pad_x)))
    top = max(0, int(round(center_y - box_h / 2 - pad_y)))
    right = min(image_w, int(round(center_x + box_w / 2 + pad_x)))
    bottom = min(image_h, int(round(center_y + box_h / 2 + pad_y)))
    return left, top, right, bottom


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def plot_confusion_matrix(gt_labels, pred_labels, class_names, out_path: Path) -> None:
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    cls_idx = {name: i for i, name in enumerate(class_names)}
    for gt, pred in zip(gt_labels, pred_labels):
        if gt in cls_idx and pred in cls_idx:
            cm[cls_idx[gt]][cls_idx[pred]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix (ViT)", fontsize=14)
    fig.colorbar(im, ax=ax)

    tick_marks = range(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=12)

    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix: {out_path}")


def plot_per_class_accuracy(gt_labels, pred_labels, class_names, out_path: Path) -> None:
    correct = Counter()
    total = Counter()
    for gt, pred in zip(gt_labels, pred_labels):
        total[gt] += 1
        if gt == pred:
            correct[gt] += 1

    accs = []
    counts = []
    for name in class_names:
        t = total.get(name, 0)
        c = correct.get(name, 0)
        accs.append(c / t * 100 if t > 0 else 0.0)
        counts.append(f"{c}/{t}")

    bar_colors = [COLORS.get(name, (128, 128, 128)) for name in class_names]
    bar_colors = [(r / 255, g / 255, b / 255) for r, g, b in bar_colors]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, accs, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy (ViT)", fontsize=14)
    ax.set_ylim(0, 110)

    for bar, acc, count in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{acc:.1f}%\n({count})",
                ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-class accuracy: {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    vit_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Predict traffic-light color with ViT.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=root / "test_imgs")
    parser.add_argument("--labels", type=Path, default=root / "test_imgs" / "test_imgs_label")
    parser.add_argument("--out", type=Path, default=vit_dir / "results")
    parser.add_argument("--padding", type=float, default=0.25)
    args = parser.parse_args()

    if args.checkpoint is None:
        meta_path = vit_dir / "checkpoints" / "vit_trafficlight.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        args.checkpoint = Path(meta["best_checkpoint"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    label_classes = read_classes(args.labels / "classes.txt")

    model_name = checkpoint.get("model_name", "google/vit-base-patch16-224")
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(classes),
        ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    img_size = processor.size["height"]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    gt_all = []
    pred_all = []
    correct = 0
    total = 0

    for img_path in image_paths(args.images):
        label_path = args.labels / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_path)
        if not labels:
            continue

        with Image.open(img_path) as src:
            image = src.convert("RGB")
        draw = ImageDraw.Draw(image)
        iw, ih = image.size

        for box_idx, label in enumerate(labels):
            gt_class = label_classes[label[0]] if label[0] < len(label_classes) else str(label[0])
            l, t, r, b = yolo_to_xyxy(label, iw, ih, args.padding)
            crop = image.crop((l, t, r, b))
            tensor = tfm(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(pixel_values=tensor)
                probs = outputs.logits.softmax(dim=1)[0]

            pred_id = int(probs.argmax().item())
            pred_class = classes[pred_id]
            confidence = float(probs[pred_id].item())

            is_correct = pred_class == gt_class
            correct += int(is_correct)
            total += 1
            gt_all.append(gt_class)
            pred_all.append(pred_class)

            color = COLORS.get(pred_class, (255, 255, 255))
            draw.rectangle((l, t, r, b), outline=color, width=3)
            draw.text((l, max(0, t - 18)), f"{pred_class} {confidence:.2f}", fill=color)

            rows.append({
                "image": img_path.name,
                "box": box_idx,
                "gt": gt_class,
                "pred": pred_class,
                "confidence": f"{confidence:.4f}",
                "correct": int(is_correct),
            })

        image.save(args.out / img_path.name, quality=95)

    # save CSV
    csv_path = args.out / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "box", "gt", "pred", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    # charts
    plot_confusion_matrix(gt_all, pred_all, classes, args.out / "confusion_matrix.png")
    plot_per_class_accuracy(gt_all, pred_all, classes, args.out / "per_class_accuracy.png")

    acc = correct / total if total else 0.0
    print(f"\nBoxes: {total}, correct: {correct}, accuracy: {acc:.3f}")
    print(f"CSV: {csv_path}")
    print(f"Annotated images: {args.out}")


if __name__ == "__main__":
    main()
