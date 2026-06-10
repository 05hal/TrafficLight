#!/usr/bin/env python
import shutil
from collections import Counter
from pathlib import Path


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


def copy_split(
    image_dir: Path,
    label_dir: Path,
    dataset_root: Path,
    split: str,
    prefix: str,
) -> tuple[int, Counter]:
    dst_image_dir = dataset_root / "images" / split
    dst_label_dir = dataset_root / "labels" / split
    ensure_dir(dst_image_dir)
    ensure_dir(dst_label_dir)

    image_count = 0
    class_counter: Counter = Counter()

    for image_path in sorted(image_dir.glob(f"{prefix}*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        shutil.copy2(image_path, dst_image_dir / image_path.name)
        shutil.copy2(label_path, dst_label_dir / label_path.name)
        image_count += 1
        class_counter.update(read_label_classes(label_path))

    return image_count, class_counter


def write_yaml(yaml_path: Path, dataset_root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {dataset_root.as_posix()}",
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
    root = Path(__file__).resolve().parent

    train_image_dir = root / "TrafficLight-main" / "train_imgs"
    train_label_dir = root / "train_imgs_label"
    val_image_dir = root / "test_imgs" / "test_imgs"
    val_label_dir = val_image_dir / "test_imgs_label"

    dataset_root = root / "groupmate_unified_dataset_v1"
    yaml_path = root / "groupmate_unified_dataset_v1.yaml"

    class_names = read_classes(train_label_dir / "classes.txt")
    val_class_names = read_classes(val_label_dir / "classes.txt")
    if class_names != val_class_names:
        raise ValueError(f"Train/val classes mismatch: {class_names} vs {val_class_names}")

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    ensure_dir(dataset_root)

    train_images, train_counter = copy_split(train_image_dir, train_label_dir, dataset_root, "train", "train")
    val_images, val_counter = copy_split(val_image_dir, val_label_dir, dataset_root, "val", "test")

    write_yaml(yaml_path, dataset_root, class_names)

    print("Groupmate unified dataset prepared successfully.")
    print(
        f"Train images: {train_images}, objects: {sum(train_counter.values())}, "
        f"class_counts: {dict(sorted(train_counter.items()))}"
    )
    print(
        f"Val images: {val_images}, objects: {sum(val_counter.values())}, "
        f"class_counts: {dict(sorted(val_counter.items()))}"
    )
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset yaml: {yaml_path}")


if __name__ == "__main__":
    main()
