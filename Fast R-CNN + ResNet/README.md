# Fast R-CNN + ResNet 两阶段交通信号灯识别

本文件夹汇总了两阶段交通信号灯识别所需的全部代码。

整体流程：

```text
test.mp4 视频帧
-> Faster R-CNN 定位交通信号灯 traffic_light
-> 裁剪检测框区域
-> ResNet 判断 red / yellow / green
-> 生成带预测框和类别的视频
```

## 文件说明

```text
Fast R-CNN + ResNet/
  requirements.txt                  统一依赖文件
  prepare_resnet_dataset.py          根据标注框裁剪 ResNet 分类数据集
  train_resnet.py                    训练 ResNet18 灯色分类模型
  predict_resnet.py                  对测试图片标注框进行 ResNet 分类
  predict_video_resnet.py            ResNet 单模型基线视频预测
  train_locator_fasterrcnn.py        训练 Faster R-CNN 单类定位模型
  predict_video_locator_resnet.py    Faster R-CNN 定位 + ResNet 分类两阶段视频预测
```

原始数据仍保留在项目根目录：

```text
train_imgs/
test_imgs/
test.mp4
```

## 运行流程

以下命令均在项目根目录 `TrafficLight` 下执行。

### 1. 安装依赖

```powershell
pip install -r '.\Fast R-CNN + ResNet\requirements.txt'
```

### 2. 训练 ResNet 分类模型

先根据已有标注框裁剪分类数据：

```powershell
python '.\Fast R-CNN + ResNet\prepare_resnet_dataset.py'
```

训练 ResNet18：

```powershell
python '.\Fast R-CNN + ResNet\train_resnet.py' --epochs 30 --batch-size 16
```

### 3. 训练 Faster R-CNN 定位模型

小样本情况下，Faster R-CNN 只学习一个类别：

```text
traffic_light
```

推荐先使用 `test.mp4` 抽帧得到的同风格样本训练定位模型：

```powershell
python '.\Fast R-CNN + ResNet\train_locator_fasterrcnn.py' --data-source test --epochs 20 --batch-size 1 --score-threshold 0.2
```

如果要做更严格的泛化测试，可以只使用 `train_imgs`：

```powershell
python '.\Fast R-CNN + ResNet\train_locator_fasterrcnn.py' --data-source train --epochs 30 --batch-size 1 --score-threshold 0.2
```

### 4. 生成两阶段预测视频

训练完 Faster R-CNN 定位模型和 ResNet 分类模型后，运行：

```powershell
python '.\Fast R-CNN + ResNet\predict_video_locator_resnet.py' --score-threshold 0.2
```

输出视频：

```text
Fast R-CNN + ResNet/locator_video_results/test_predicted_locator_resnet.mp4
```

如果检测框太少，可以降低定位阈值：

```powershell
python '.\Fast R-CNN + ResNet\predict_video_locator_resnet.py' --score-threshold 0.05
```

## ResNet 单模型基线

该版本不自动定位交通信号灯，而是使用 `test_imgs/test_imgs_label` 中的人工标注框作为位置先验：

```powershell
python '.\Fast R-CNN + ResNet\predict_video_resnet.py'
```

输出视频：

```text
Fast R-CNN + ResNet/video_results/test_predicted.mp4
```

## 快速测试

如果只是确认代码能跑通，可以使用很小的数据量做 smoke test：

```powershell
python '.\Fast R-CNN + ResNet\train_locator_fasterrcnn.py' --data-source test --epochs 1 --batch-size 1 --max-image-size 320 --model-min-size 320 --model-max-size 320 --max-train-samples 4 --max-val-samples 2 --score-threshold 0.05
python '.\Fast R-CNN + ResNet\predict_video_locator_resnet.py' --model-min-size 320 --model-max-size 320 --max-frames 10 --score-threshold 0.05
```

注意：快速测试只验证训练、保存模型和生成视频流程是否正常，不代表最终模型效果。

## 方法说明

在样本量较少时，如果直接让 Faster R-CNN 同时学习 `red/yellow/green` 三类检测，每一类可用样本都会变少，模型容易漏检或混淆类别。

因此这里采用两阶段方案：

```text
Faster R-CNN：只负责定位 traffic_light
ResNet：只负责判断红灯、黄灯、绿灯
```

这样可以降低每个模型的学习难度，更适合当前的小样本数据。
