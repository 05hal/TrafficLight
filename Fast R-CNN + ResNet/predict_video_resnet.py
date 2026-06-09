#!/usr/bin/env python
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


COLORS = {"red": (0, 0, 255), "yellow": (0, 210, 255), "green": (0, 220, 80)}


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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


def read_label_timeline(labels_dir: Path) -> list[list[tuple[int, float, float, float, float]]]:
    def label_key(path: Path) -> tuple[str, int]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return path.stem.rstrip(digits), int(digits or 0)

    files = sorted((p for p in labels_dir.glob("*.txt") if p.name != "classes.txt"), key=label_key)
    return [read_yolo_labels(path) for path in files]


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


def timeline_boxes(timeline, frame_idx: int, total_frames: int, image_w: int, image_h: int, padding: float):
    if not timeline:
        return []
    timeline_idx = round(frame_idx * (len(timeline) - 1) / max(1, total_frames - 1))
    return [yolo_to_xyxy(label, image_w, image_h, padding) for label in timeline[timeline_idx]]


def save_case(case_dir: Path, frame: np.ndarray, prefix: str, frame_idx: int, limit: int) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    if len(list(case_dir.glob(f"{prefix}_*.jpg"))) >= limit:
        return
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if ok:
        encoded.tofile(str(case_dir / f"{prefix}_frame{frame_idx:06d}.jpg"))


def draw_label(frame: np.ndarray, box, label: str, confidence: float) -> None:
    x1, y1, x2, y2 = box
    color = COLORS.get(label, (255, 255, 255))
    text = f"{label} {confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
    cv2.putText(frame, text, (x1 + 4, y_text + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ResNet traffic-light color predictions on test.mp4.")
    parser.add_argument("--video", type=Path, default=Path(__file__).resolve().parents[1] / "test.mp4")
    parser.add_argument("--labels", type=Path, default=Path(__file__).resolve().parents[1] / "test_imgs" / "test_imgs_label")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "video_results")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--label-padding", type=float, default=0.0)
    parser.add_argument("--case-limit", type=int, default=8)
    args = parser.parse_args()

    if args.checkpoint is None:
        meta = json.loads((Path(__file__).resolve().parent / "checkpoints" / "resnet18_trafficlight.json").read_text(encoding="utf-8"))
        args.checkpoint = Path(meta["checkpoint"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with args.checkpoint.open("rb") as f:
        checkpoint = torch.load(f, map_location=device)
    classes = checkpoint["classes"]
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

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timeline = read_label_timeline(args.labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in (args.out_dir / "success_cases", args.out_dir / "failure_cases"):
        if case_dir.exists():
            for old_case in case_dir.glob("*.jpg"):
                old_case.unlink()
    out_video = args.out_dir / "test_predicted.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rows = []
    recent_labels: deque[str] = deque(maxlen=5)
    detected_frames = 0
    low_conf_frames = 0
    switch_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = frame.copy()
        boxes = timeline_boxes(timeline, frame_idx, total_frames, width, height, args.label_padding)
        if boxes:
            detected_frames += 1
        frame_preds = []
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            tensor = tfm(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = model(tensor).softmax(dim=1)[0]
            pred_id = int(probs.argmax().item())
            label = classes[pred_id]
            confidence = float(probs[pred_id].item())
            frame_preds.append((label, confidence, box))
            draw_label(annotated, box, label, confidence)
            rows.append(
                {
                    "frame": frame_idx,
                    "time_sec": f"{frame_idx / fps:.3f}",
                    "box": box_idx,
                    "pred": label,
                    "confidence": f"{confidence:.4f}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        if frame_preds:
            best = max(frame_preds, key=lambda item: item[1])
            recent_labels.append(best[0])
            low_conf = best[1] < args.confidence_threshold
            unstable = len(recent_labels) >= 3 and len(set(recent_labels)) > 1
            low_conf_frames += int(low_conf)
            switch_frames += int(unstable)
            if low_conf or unstable:
                save_case(args.out_dir / "failure_cases", annotated, "failure", frame_idx, args.case_limit)
            else:
                save_case(args.out_dir / "success_cases", annotated, "success", frame_idx, args.case_limit)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    csv_path = args.out_dir / "frame_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "time_sec", "box", "pred", "confidence", "x1", "y1", "x2", "y2"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    (args.out_dir / "case_analysis.md").write_text(
        f"# test.mp4 ResNet 预测结果分析\n\n"
        f"- 视频总帧数：{frame_idx}\n"
        f"- 成功获得交通灯候选框的帧数：{detected_frames}\n"
        f"- 低置信度帧数：{low_conf_frames}\n"
        f"- 状态跳变/不稳定帧数：{switch_frames}\n\n"
        f"本版本使用 `test_imgs/test_imgs_label` 中由 `test.mp4` 抽帧得到的标注框作为交通灯位置先验，再用 ResNet 对框内区域进行红/黄/绿分类。\n",
        encoding="utf-8",
    )
    print(f"Frames: {frame_idx}")
    print(f"Detected frames: {detected_frames}")
    print(f"Low-confidence frames: {low_conf_frames}")
    print(f"State-switch frames: {switch_frames}")
    print(f"Video: {out_video}")
    print(f"CSV: {csv_path}")
    print(f"Analysis: {args.out_dir / 'case_analysis.md'}")


if __name__ == "__main__":
    main()
