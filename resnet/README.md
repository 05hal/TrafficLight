# ResNet traffic-light prediction

This folder trains a ResNet18 traffic-light color classifier and renders predictions on `test.mp4`.

ResNet is used as a classifier. It does not locate traffic lights by itself. The video script uses the existing `test_imgs/test_imgs_label` boxes as the traffic-light position prior, then classifies each cropped region as `red`, `yellow`, or `green`.

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

Render predictions on `test.mp4`:

```powershell
python .\resnet\predict_video_resnet.py
```

The video result is written to:

```text
resnet/video_results/test_predicted.mp4
```

The same folder also contains `frame_predictions.csv`, `case_analysis.md`, `success_cases/`, and `failure_cases/`.

The current prediction scripts use no extra bbox padding by default. This keeps the crop focused on the annotated traffic-light region and improves red-light recognition on `test.mp4`.
