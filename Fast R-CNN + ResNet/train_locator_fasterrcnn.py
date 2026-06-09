#!/usr/bin/env python
import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES = ["traffic_light"]


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


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def yolo_to_xyxy(label: tuple[int, float, float, float, float], image_w: int, image_h: int) -> list[float]:
    _, cx, cy, width, height = label
    box_w = width * image_w
    box_h = height * image_h
    center_x = cx * image_w
    center_y = cy * image_h
    return [
        max(0.0, center_x - box_w / 2),
        max(0.0, center_y - box_h / 2),
        min(float(image_w), center_x + box_w / 2),
        min(float(image_h), center_y + box_h / 2),
    ]


class TrafficLightLocatorDataset(Dataset):
    def __init__(self, images: list[Path], label_dir: Path, max_image_size: int) -> None:
        self.images = images
        self.label_dir = label_dir
        self.max_image_size = max_image_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        with Image.open(image_path) as src:
            image = src.convert("RGB")
        image_w, image_h = image.size
        boxes = []
        for label in read_yolo_labels(self.label_dir / f"{image_path.stem}.txt"):
            box = yolo_to_xyxy(label, image_w, image_h)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)

        if self.max_image_size > 0 and max(image_w, image_h) > self.max_image_size:
            scale = self.max_image_size / max(image_w, image_h)
            new_w = max(1, int(round(image_w * scale)))
            new_h = max(1, int(round(image_h * scale)))
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            boxes = [[coord * scale for coord in box] for box in boxes]

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.ones((boxes_tensor.shape[0],), dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64),
        }
        return F.to_tensor(image), target, image_path.name


def collate_fn(batch):
    images, targets, names = zip(*batch)
    return list(images), list(targets), list(names)


def build_model(pretrained: bool, min_size: int, max_size: int):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    return model


def box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-9, area_a + area_b - inter)


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for images, targets, _ in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        losses = model(images, targets)
        loss = sum(value for value in losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(1, batches)


@torch.no_grad()
def evaluate(model, loader, device, score_threshold: float, iou_threshold: float) -> dict:
    model.eval()
    tp = fp = fn = 0
    rows = []
    for images, targets, names in loader:
        outputs = model([image.to(device) for image in images])
        for name, target, output in zip(names, targets, outputs):
            gt_boxes = target["boxes"].cpu()
            matched = set()
            keep = output["scores"].cpu() >= score_threshold
            pred_boxes = output["boxes"].cpu()[keep]
            pred_scores = output["scores"].cpu()[keep]
            for box, score in zip(pred_boxes, pred_scores):
                best_iou = 0.0
                best_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched:
                        continue
                    iou = box_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gt_idx
                ok = best_iou >= iou_threshold and best_idx >= 0
                if ok:
                    tp += 1
                    matched.add(best_idx)
                else:
                    fp += 1
                rows.append({"image": name, "score": f"{float(score):.4f}", "iou": f"{best_iou:.4f}", "correct": int(ok)})
            fn += max(0, len(gt_boxes) - len(matched))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "rows": rows}


def make_splits(root: Path, data_source: str, val_ratio: float, seed: int):
    if data_source == "test":
        image_dir = root / "test_imgs"
        label_dir = image_dir / "test_imgs_label"
        images = image_paths(image_dir)
    elif data_source == "combined":
        image_dir = None
        label_dir = None
        images = image_paths(root / "train_imgs") + image_paths(root / "test_imgs")
    else:
        image_dir = root / "train_imgs"
        label_dir = image_dir / "train_imgs_label"
        images = image_paths(image_dir)

    rng = random.Random(seed)
    rng.shuffle(images)
    val_count = max(1, round(len(images) * val_ratio))
    return sorted(images[val_count:], key=lambda p: p.name), sorted(images[:val_count], key=lambda p: p.name)


def label_dir_for(image_path: Path, root: Path) -> Path:
    if image_path.parent.name == "test_imgs":
        return root / "test_imgs" / "test_imgs_label"
    return root / "train_imgs" / "train_imgs_label"


class MixedLocatorDataset(Dataset):
    def __init__(self, images: list[Path], root: Path, max_image_size: int) -> None:
        self.images = images
        self.root = root
        self.max_image_size = max_image_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        return TrafficLightLocatorDataset([image_path], label_dir_for(image_path, self.root), self.max_image_size)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN as a single-class traffic-light locator.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-source", choices=["train", "test", "combined"], default="test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-image-size", type=int, default=640)
    parser.add_argument("--model-min-size", type=int, default=480)
    parser.add_argument("--model-max-size", type=int, default=640)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "checkpoints" / "traffic_light_locator.pt")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "locator_results")
    args = parser.parse_args()

    fit_images, val_images = make_splits(args.root, args.data_source, args.val_ratio, args.seed)
    if args.max_train_samples > 0:
        fit_images = fit_images[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_images = val_images[: args.max_val_samples]
    train_set = MixedLocatorDataset(fit_images, args.root, args.max_image_size)
    val_set = MixedLocatorDataset(val_images, args.root, args.max_image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.pretrained, args.model_min_size, args.model_max_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_recall = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_path = args.out
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device, args.score_threshold, 0.5)
        print(f"epoch {epoch:03d}/{args.epochs} train_loss={loss:.4f} val_precision={metrics['precision']:.3f} val_recall={metrics['recall']:.3f}")
        if metrics["recall"] >= best_recall:
            best_recall = metrics["recall"]
            with args.out.open("wb") as f:
                torch.save({"model": model.state_dict(), "classes": CLASSES, "data_source": args.data_source}, f)
            best_path = args.out

    metrics = evaluate(model, val_loader, device, args.score_threshold, 0.5)
    args.results.mkdir(parents=True, exist_ok=True)
    csv_path = args.results / "val_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "score", "iou", "correct"])
        writer.writeheader()
        writer.writerows(metrics["rows"])
    args.out.with_suffix(".json").write_text(
        json.dumps({"classes": CLASSES, "checkpoint": str(best_path), "best_val_recall": best_recall, "val_metrics": {k: v for k, v in metrics.items() if k != "rows"}}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint: {best_path}")
    print(f"Val precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
