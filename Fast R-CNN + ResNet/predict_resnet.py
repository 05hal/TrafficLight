#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision import models, transforms


COLORS = {"red": (255, 0, 0), "yellow": (255, 210, 0), "green": (0, 220, 80)}


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, cx, cy, width, height = parts
            labels.append((int(cls), float(cx), float(cy), float(width), float(height)))
    return labels


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def yolo_to_xyxy(label: tuple[int, float, float, float, float], image_w: int, image_h: int, padding: float) -> tuple[int, int, int, int]:
    _, cx, cy, width, height = label
    box_w = width * image_w
    box_h = height * image_h
    center_x = cx * image_w
    center_y = cy * image_h
    pad_x = box_w * padding
    pad_y = box_h * padding
    return (
        max(0, int(round(center_x - box_w / 2 - pad_x))),
        max(0, int(round(center_y - box_h / 2 - pad_y))),
        min(image_w, int(round(center_x + box_w / 2 + pad_x))),
        min(image_h, int(round(center_y + box_h / 2 + pad_y))),
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict traffic-light color on YOLO boxes with a trained ResNet18.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=Path(__file__).resolve().parents[1] / "test_imgs")
    parser.add_argument("--labels", type=Path, default=Path(__file__).resolve().parents[1] / "test_imgs" / "test_imgs_label")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--padding", type=float, default=0.0)
    args = parser.parse_args()

    if args.checkpoint is None:
        meta = json.loads((Path(__file__).resolve().parent / "checkpoints" / "resnet18_trafficlight.json").read_text(encoding="utf-8"))
        args.checkpoint = Path(meta["checkpoint"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with args.checkpoint.open("rb") as f:
        checkpoint = torch.load(f, map_location=device)
    classes = checkpoint["classes"]
    label_classes = read_classes(args.labels / "classes.txt")
    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    correct = 0
    total = 0
    for image_path in image_paths(args.images):
        labels = read_yolo_labels(args.labels / f"{image_path.stem}.txt")
        if not labels:
            continue
        with Image.open(image_path) as src:
            image = src.convert("RGB")
        draw = ImageDraw.Draw(image)
        image_w, image_h = image.size
        for box_idx, label in enumerate(labels):
            gt_class = label_classes[label[0]] if label[0] < len(label_classes) else str(label[0])
            left, top, right, bottom = yolo_to_xyxy(label, image_w, image_h, args.padding)
            tensor = tfm(image.crop((left, top, right, bottom))).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = model(tensor).softmax(dim=1)[0]
            pred_id = int(probs.argmax().item())
            pred_class = classes[pred_id]
            confidence = float(probs[pred_id].item())
            correct += int(pred_class == gt_class)
            total += 1
            color = COLORS.get(pred_class, (255, 255, 255))
            draw.rectangle((left, top, right, bottom), outline=color, width=3)
            draw.text((left, max(0, top - 18)), f"{pred_class} {confidence:.2f}", fill=color)
            rows.append({"image": image_path.name, "box": box_idx, "gt": gt_class, "pred": pred_class, "confidence": f"{confidence:.4f}", "correct": int(pred_class == gt_class)})
        image.save(args.out / image_path.name, quality=95)
    csv_path = args.out / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "box", "gt", "pred", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Boxes: {total}, correct: {correct}, accuracy: {correct / total if total else 0.0:.3f}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
