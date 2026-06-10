#!/usr/bin/env python
"""Crop YOLO-annotated traffic-light boxes into a classification dataset.

Supports multiple data sources for future expansion. Each source is a dict
with ``images`` (image folder) and ``labels`` (YOLO label folder) keys.
"""

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
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        labels.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return labels


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
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def crop_source(
    images: list[Path],
    label_dir: Path,
    out_root: Path,
    split: str,
    class_names: list[str],
    padding: float,
) -> Counter:
    counter = Counter()
    id_to_name = {i: name for i, name in enumerate(class_names)}
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_path)
        if not labels:
            continue
        with Image.open(img_path) as src:
            src = src.convert("RGB")
            iw, ih = src.size
        for box_idx, label in enumerate(labels):
            cls_id = label[0]
            if cls_id not in id_to_name:
                continue
            cls_name = id_to_name[cls_id]
            l, t, r, b = yolo_to_xyxy(label, iw, ih, padding)
            if r <= l or b <= t:
                continue
            with Image.open(img_path) as src:
                crop = src.convert("RGB").crop((l, t, r, b))
            dst_dir = out_root / split / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            crop.save(dst_dir / f"{img_path.stem}_{box_idx:02d}.jpg", quality=95)
            counter[cls_name] += 1
    return counter


def stratified_split(
    images: list[Path], label_dir: Path, val_ratio: float, rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    by_class: dict[int, list[Path]] = {}
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_path)
        cls_id = labels[0][0] if labels else -1
        by_class.setdefault(cls_id, []).append(img_path)

    fit_imgs, val_imgs = [], []
    for cls_id, group in sorted(by_class.items()):
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))
        val_imgs.extend(group[:n_val])
        fit_imgs.extend(group[n_val:])
    return sorted(fit_imgs), sorted(val_imgs)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Prepare cropped classification dataset from YOLO labels.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        dest="sources",
        metavar=("NAME", "IMAGES", "LABELS"),
        help="Extra data source: name images_dir labels_dir (can repeat).",
    )
    parser.add_argument(
        "--swap", action="store_true",
        help="Swap roles: use test_imgs for train+val, train_imgs for test. "
             "Useful when test_imgs have more consistent annotation style.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.swap:
        fit_img_dir, fit_lbl_dir = root / "test_imgs", root / "test_imgs" / "test_imgs_label"
        eval_img_dir, eval_lbl_dir = root / "train_imgs", root / "train_imgs" / "train_imgs_label"
    else:
        fit_img_dir, fit_lbl_dir = root / "train_imgs", root / "train_imgs" / "train_imgs_label"
        eval_img_dir, eval_lbl_dir = root / "test_imgs", root / "test_imgs" / "test_imgs_label"

    fit_classes = read_classes(fit_lbl_dir / "classes.txt")
    eval_classes = read_classes(eval_lbl_dir / "classes.txt")
    all_fit = image_paths(fit_img_dir)

    fit_imgs, val_imgs = stratified_split(all_fit, fit_lbl_dir, args.val_ratio, rng)

    print(f"Classes (fit):  {fit_classes}")
    print(f"Classes (eval): {eval_classes}")
    c1 = crop_source(fit_imgs, fit_lbl_dir, args.out, "train", fit_classes, args.padding)
    c2 = crop_source(val_imgs, fit_lbl_dir, args.out, "val", fit_classes, args.padding)
    print(f"  train crops: {sum(c1.values())} {dict(c1)}")
    print(f"  val crops:   {sum(c2.values())} {dict(c2)}")

    c3 = crop_source(image_paths(eval_img_dir), eval_lbl_dir, args.out, "test", eval_classes, args.padding)
    print(f"  test crops:  {sum(c3.values())} {dict(c3)}")

    extra = []
    if args.sources:
        for name, img_dir, lbl_dir in args.sources:
            extra.append((name, Path(img_dir), Path(lbl_dir)))
    for name, img_dir, lbl_dir in extra:
        classes = read_classes(lbl_dir / "classes.txt")
        print(f"Classes ({name}): {classes}")
        ce = crop_source(image_paths(img_dir), lbl_dir, args.out, "test", classes, args.padding)
        print(f"  {name} crops: {sum(ce.values())} {dict(ce)}")

    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
