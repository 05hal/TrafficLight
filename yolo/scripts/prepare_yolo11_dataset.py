#!/usr/bin/env python
import shutil
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_split(src_img_dir: Path, src_label_dir: Path, dst_root: Path, split: str) -> tuple[int, int]:
    dst_img_dir = dst_root / "images" / split
    dst_label_dir = dst_root / "labels" / split
    ensure_dir(dst_img_dir)
    ensure_dir(dst_label_dir)

    image_count = 0
    label_count = 0

    for img_path in sorted(src_img_dir.glob("*.jpg")):
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        image_count += 1

        label_path = src_label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, dst_label_dir / label_path.name)
            label_count += 1
        else:
            (dst_label_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")

    return image_count, label_count


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
    root = Path(__file__).resolve().parents[2]
    dataset_root = root / "yolo11_dataset"

    train_img_dir = root / "train_imgs"
    train_label_dir = train_img_dir / "train_imgs_label"
    val_img_dir = root / "test_imgs"
    val_label_dir = val_img_dir / "test_imgs_label"

    class_names = [
        line.strip()
        for line in (train_label_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    train_images, train_labels = copy_split(train_img_dir, train_label_dir, dataset_root, "train")
    val_images, val_labels = copy_split(val_img_dir, val_label_dir, dataset_root, "val")

    yaml_path = root / "dataset.yaml"
    write_yaml(yaml_path, dataset_root, class_names)

    print("YOLO11 dataset prepared successfully.")
    print(f"Classes: {class_names}")
    print(f"Train images: {train_images}, train labels: {train_labels}")
    print(f"Val images: {val_images}, val labels: {val_labels}")
    print(f"Dataset directory: {dataset_root}")
    print(f"Dataset yaml: {yaml_path}")


if __name__ == "__main__":
    main()
