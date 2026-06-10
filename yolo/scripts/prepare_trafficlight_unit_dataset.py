#!/usr/bin/env python
import random
import shutil
from collections import Counter
from pathlib import Path


SEED = 20260605
VAL_RATIO = 0.2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_classes(classes_path: Path) -> list[str]:
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_label_classes(label_path: Path) -> Counter:
    counts = Counter()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        counts[int(line.split()[0])] += 1
    return counts


def copy_samples(image_paths: list[Path], src_label_dir: Path, dst_root: Path, split: str) -> Counter:
    dst_image_dir = dst_root / "images" / split
    dst_label_dir = dst_root / "labels" / split
    ensure_dir(dst_image_dir)
    ensure_dir(dst_label_dir)

    class_counter = Counter()
    for image_path in image_paths:
        shutil.copy2(image_path, dst_image_dir / image_path.name)
        label_path = src_label_dir / f"{image_path.stem}.txt"
        shutil.copy2(label_path, dst_label_dir / label_path.name)
        class_counter.update(read_label_classes(label_path))
    return class_counter


def write_yaml(yaml_path: Path, dataset_root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {dataset_root.name}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # Only use the test_imgs split because its boxes match the "whole traffic-light unit" annotation style.
    src_image_dir = root / "test_imgs"
    src_label_dir = src_image_dir / "test_imgs_label"
    dataset_root = root / "trafficlight_unit_dataset_v1"
    yaml_path = root / "trafficlight_unit_dataset_v1.yaml"

    image_paths = sorted(src_image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {src_image_dir}")

    rng = random.Random(SEED)
    rng.shuffle(image_paths)

    val_count = max(1, round(len(image_paths) * VAL_RATIO))
    val_images = sorted(image_paths[:val_count], key=lambda p: p.name)
    train_images = sorted(image_paths[val_count:], key=lambda p: p.name)

    class_names = read_classes(src_label_dir / "classes.txt")

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    ensure_dir(dataset_root)

    train_counter = copy_samples(train_images, src_label_dir, dataset_root, "train")
    val_counter = copy_samples(val_images, src_label_dir, dataset_root, "val")
    write_yaml(yaml_path, dataset_root, class_names)

    print("Traffic-light unit YOLO11 dataset prepared successfully.")
    print(f"Source split: {src_image_dir.name}")
    print(f"Train images: {len(train_images)}, labels: {sum(train_counter.values())}, class_counts: {dict(sorted(train_counter.items()))}")
    print(f"Val images: {len(val_images)}, labels: {sum(val_counter.values())}, class_counts: {dict(sorted(val_counter.items()))}")
    print(f"Dataset directory: {dataset_root}")
    print(f"Dataset yaml: {yaml_path}")


if __name__ == "__main__":
    main()
