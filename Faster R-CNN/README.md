# Faster R-CNN traffic-light locator

This folder adds a single-class Faster R-CNN locator for small-sample traffic-light videos.

The recommended small-sample pipeline is:

```text
full video frame -> Faster R-CNN locates traffic_light -> crop box -> ResNet classifies red/yellow/green
```

Faster R-CNN only learns one class, `traffic_light`, so red, yellow, and green samples are merged for localization. This is more stable than asking a small dataset to train a three-class detector directly.

## Install

Run from the project root:

```powershell
pip install -r '.\Faster R-CNN\requirements_fasterrcnn.txt'
```

## Train Locator

For the current `test.mp4` visualization, use the same-style extracted frames:

```powershell
python '.\Faster R-CNN\train_locator_fasterrcnn.py' --data-source test --epochs 20 --batch-size 1 --score-threshold 0.2
```

For a stricter setup using only `train_imgs`:

```powershell
python '.\Faster R-CNN\train_locator_fasterrcnn.py' --data-source train --epochs 30 --batch-size 1 --score-threshold 0.2
```

Outputs:

```text
Faster R-CNN/checkpoints/traffic_light_locator.pt
Faster R-CNN/checkpoints/traffic_light_locator.json
Faster R-CNN/locator_results/val_predictions.csv
```

## Predict test.mp4

After training the locator and ResNet classifier:

```powershell
python '.\Faster R-CNN\predict_video_locator_resnet.py' --score-threshold 0.2
```

Output video:

```text
Faster R-CNN/locator_video_results/test_predicted_locator_resnet.mp4
```

If detection is still sparse, try lowering the locator threshold:

```powershell
python '.\Faster R-CNN\predict_video_locator_resnet.py' --score-threshold 0.05
```

## Quick Smoke Test

To verify the pipeline on CPU without waiting for full training:

```powershell
python '.\Faster R-CNN\train_locator_fasterrcnn.py' --data-source test --epochs 1 --batch-size 1 --max-image-size 320 --model-min-size 320 --model-max-size 320 --max-train-samples 4 --max-val-samples 2 --score-threshold 0.05
python '.\Faster R-CNN\predict_video_locator_resnet.py' --model-min-size 320 --model-max-size 320 --max-frames 10 --score-threshold 0.05
```
