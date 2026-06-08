#!/usr/bin/env python
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label in {path}: {line}")
        cls, cx, cy, width, height = parts
        labels.append((int(cls), float(cx), float(cy), float(width), float(height)))
    return labels


def yolo_to_xyxy(label: tuple[int, float, float, float, float], image_w: int, image_h: int, padding: float) -> tuple[int, int, int, int]:
    _, cx, cy, width, height = label
    box_w = width * image_w
    box_h = height * image_h
    center_x = cx * image_w
    center_y = cy * image_h

    pad_x = box_w * padding
    pad_y = box_h * padding
    left = max(0, int(round(center_x - box_w / 2 - pad_x)))
    top = max(0, int(round(center_y - box_h / 2 - pad_y)))
    right = min(image_w, int(round(center_x + box_w / 2 + pad_x)))
    bottom = min(image_h, int(round(center_y + box_h / 2 + pad_y)))
    return left, top, right, bottom


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def copy_split(
    samples: list[Path],
    label_dir: Path,
    out_root: Path,
    split: str,
    class_names: list[str],
    padding: float,
) -> Counter:
    counter = Counter()
    for image_path in samples:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        labels = read_yolo_labels(label_path)
        if not labels:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_w, image_h = image.size
            for box_idx, label in enumerate(labels):
                class_id = label[0]
                class_name = class_names[class_id]
                left, top, right, bottom = yolo_to_xyxy(label, image_w, image_h, padding)
                if right <= left or bottom <= top:
                    continue
                crop = image.crop((left, top, right, bottom))
                dst_dir = out_root / split / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_name = f"{image_path.stem}_{box_idx:02d}.jpg"
                crop.save(dst_dir / dst_name, quality=95)
                counter[class_name] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop YOLO traffic-light boxes into a ResNet classification dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--padding", type=float, default=0.25, help="Extra crop padding as a fraction of bbox size.")
    args = parser.parse_args()

    root = args.root
    train_image_dir = root / "train_imgs"
    train_label_dir = train_image_dir / "train_imgs_label"
    test_image_dir = root / "test_imgs"
    test_label_dir = test_image_dir / "test_imgs_label"
    train_class_names = read_classes(train_label_dir / "classes.txt")
    test_class_names = read_classes(test_label_dir / "classes.txt")

    train_images = image_paths(train_image_dir)
    if not train_images:
        raise FileNotFoundError(f"No training images found in {train_image_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(train_images)
    val_count = max(1, round(len(train_images) * args.val_ratio))
    val_images = sorted(train_images[:val_count], key=lambda path: path.name)
    fit_images = sorted(train_images[val_count:], key=lambda path: path.name)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    train_counter = copy_split(fit_images, train_label_dir, args.out, "train", train_class_names, args.padding)
    val_counter = copy_split(val_images, train_label_dir, args.out, "val", train_class_names, args.padding)
    test_counter = copy_split(image_paths(test_image_dir), test_label_dir, args.out, "test", test_class_names, args.padding)

    print(f"Train classes: {train_class_names}")
    print(f"Test classes: {test_class_names}")
    print(f"Output: {args.out}")
    print(f"Train crops: {sum(train_counter.values())}, {dict(train_counter)}")
    print(f"Val crops: {sum(val_counter.values())}, {dict(val_counter)}")
    print(f"Test crops: {sum(test_counter.values())}, {dict(test_counter)}")


if __name__ == "__main__":
    main()
