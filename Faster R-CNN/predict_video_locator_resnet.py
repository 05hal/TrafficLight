#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F


COLORS = {"red": (0, 0, 255), "yellow": (0, 210, 255), "green": (0, 220, 80)}


def build_resnet(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_locator(checkpoint_arg: Path | None, model_min_size: int, model_max_size: int):
    if checkpoint_arg is None:
        meta_path = Path(__file__).resolve().parent / "checkpoints" / "traffic_light_locator.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checkpoint_arg = Path(meta["checkpoint"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with checkpoint_arg.open("rb") as f:
        checkpoint = torch.load(f, map_location=device)
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=len(checkpoint["classes"]) + 1,
        min_size=model_min_size,
        max_size=model_max_size,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, device


def load_resnet(checkpoint_arg: Path | None, device: torch.device):
    if checkpoint_arg is None:
        meta_path = Path(__file__).resolve().parents[1] / "resnet" / "checkpoints" / "resnet18_trafficlight.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checkpoint_arg = Path(meta["checkpoint"])
    with checkpoint_arg.open("rb") as f:
        checkpoint = torch.load(f, map_location=device)
    classes = checkpoint["classes"]
    model = build_resnet(len(classes)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, classes, tfm


def classify_crop(model, classes, tfm, device, crop_bgr: np.ndarray):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = tfm(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(tensor).softmax(dim=1)[0]
    pred_id = int(probs.argmax().item())
    return classes[pred_id], float(probs[pred_id].item())


def draw_label(frame: np.ndarray, box, label: str, locator_score: float, cls_score: float) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    color = COLORS.get(label, (255, 255, 255))
    text = f"{label} cls:{cls_score:.2f} loc:{locator_score:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
    cv2.putText(frame, text, (x1 + 4, y_text + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def valid_box(box, frame_w: int, frame_h: int, min_area_ratio: float, max_area_ratio: float, min_aspect: float, max_aspect: float) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1 or bh <= 1:
        return False
    area_ratio = (bw * bh) / max(1.0, frame_w * frame_h)
    aspect = bw / bh
    return min_area_ratio <= area_ratio <= max_area_ratio and min_aspect <= aspect <= max_aspect


def main() -> None:
    parser = argparse.ArgumentParser(description="Use Faster R-CNN to locate traffic lights, then ResNet to classify the light state.")
    parser.add_argument("--video", type=Path, default=Path(__file__).resolve().parents[1] / "test.mp4")
    parser.add_argument("--locator-checkpoint", type=Path, default=None)
    parser.add_argument("--resnet-checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "locator_video_results")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--model-min-size", type=int, default=480)
    parser.add_argument("--model-max-size", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-boxes", type=int, default=1)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.08)
    parser.add_argument("--min-aspect", type=float, default=0.08)
    parser.add_argument("--max-aspect", type=float, default=1.2)
    args = parser.parse_args()

    locator, device = load_locator(args.locator_checkpoint, args.model_min_size, args.model_max_size)
    classifier, classes, tfm = load_resnet(args.resnet_checkpoint, device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_video = args.out_dir / "test_predicted_locator_resnet.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    rows = []
    frame_idx = 0
    detected_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        annotated = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(frame_rgb).to(device)
        with torch.no_grad():
            output = locator([tensor])[0]
        keep = output["scores"].detach().cpu() >= args.score_threshold
        boxes = output["boxes"].detach().cpu()[keep]
        scores = output["scores"].detach().cpu()[keep]
        candidates = [
            (box, score)
            for box, score in zip(boxes, scores)
            if valid_box(box, width, height, args.min_area_ratio, args.max_area_ratio, args.min_aspect, args.max_aspect)
        ]
        candidates.sort(key=lambda item: float(item[1]), reverse=True)
        candidates = candidates[: args.max_boxes]

        if candidates:
            detected_frames += 1
        for box_idx, (box, score) in enumerate(candidates):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            label, cls_score = classify_crop(classifier, classes, tfm, device, frame[y1:y2, x1:x2])
            draw_label(annotated, (x1, y1, x2, y2), label, float(score), cls_score)
            rows.append(
                {
                    "frame": frame_idx,
                    "time_sec": f"{frame_idx / fps:.3f}",
                    "box": box_idx,
                    "pred": label,
                    "class_confidence": f"{cls_score:.4f}",
                    "locator_confidence": f"{float(score):.4f}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    csv_path = args.out_dir / "frame_predictions_locator_resnet.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "time_sec", "box", "pred", "class_confidence", "locator_confidence", "x1", "y1", "x2", "y2"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"Frames: {frame_idx}")
    print(f"Detected frames: {detected_frames}")
    print(f"Video: {out_video}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
