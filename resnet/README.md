# ResNet traffic-light prediction

This folder trains and evaluates a ResNet18 traffic-light color classifier from the existing YOLO-format annotations.

ResNet is used as a classifier. It does not locate traffic lights by itself. The YOLO label files provide bounding boxes, and `prepare_resnet_dataset.py` crops those boxes into class folders before training.

## Run

Install dependencies:

```powershell
pip install -r .\resnet\requirements_resnet.txt
```

Prepare cropped classification data:

```powershell
python .\resnet\prepare_resnet_dataset.py
```

Train ResNet18:

```powershell
python .\resnet\train_resnet.py --epochs 30 --batch-size 16
```

Predict on test images:

```powershell
python .\resnet\predict_resnet.py
```

## Current Result

The latest local run produced:

```text
Boxes: 100
Correct: 86
Accuracy: 0.860
```

Detailed per-box predictions are saved in:

```text
resnet/results/predictions.csv
```

Model checkpoints, cropped datasets, and rendered result images are ignored by Git because they are generated artifacts.
