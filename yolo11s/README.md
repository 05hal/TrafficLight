# YOLO11s 红绿灯检测实验

## 1. 项目概述

本文件夹用于归档本次基于统一组员标签数据集完成的 `YOLO11s` 实验结果。

本次任务为三分类交通灯检测，类别定义如下：

- `0: green`
- `1: yellow`
- `2: red`

训练集使用原始 `train_imgs` 图像及更新后的 `train_imgs_label` 标签。
验证集使用组员新上传的 `test_imgs/test_imgs` 图像及其更新后的 `test_imgs_label` 标签。

本归档文件夹主要保存以下内容：

- 数据集整理脚本
- 数据集配置文件
- 训练输出结果与模型权重
- 少量代表性的带框带标签预测图片
- 带框带标签的预测视频

## 2. 环境配置

### 2.1 硬件环境

- GPU：`NVIDIA GeForce RTX 4070 Laptop GPU`
- 操作系统：`Windows`

### 2.2 Python 环境

- Python 可执行文件：`D:\ai-envs\tl-gpu-cu124\python.exe`
- YOLO 可执行文件：`D:\ai-envs\tl-gpu-cu124\Scripts\yolo.exe`

### 2.3 核心依赖

- `torch 2.5.1`
- `ultralytics 8.4.60`
- 训练使用 CUDA 设备：`device=0`

## 3. 数据集整理

### 3.1 数据来源

- 训练图片：`TrafficLight-main/train_imgs`
- 训练标签：`train_imgs_label`
- 验证图片：`test_imgs/test_imgs`
- 验证标签：`test_imgs/test_imgs/test_imgs_label`

### 3.2 整理脚本

本次数据集重组使用的脚本为：

- `scripts/prepare_groupmate_unified_dataset.py`

该脚本会生成标准 YOLO 目录结构：

- `images/train`
- `images/val`
- `labels/train`
- `labels/val`

训练时实际使用的数据集配置文件为：

- `configs/groupmate_unified_dataset_v1.yaml`

## 4. 模型训练

### 4.1 模型选择

最终使用模型为 `yolo11s.pt`。

考虑到此前实验中出现过 CUDA 不稳定问题，本次采用了较为保守的训练参数，以优先保证训练过程能够稳定完成。

### 4.2 训练命令

```powershell
D:\ai-envs\tl-gpu-cu124\Scripts\yolo.exe detect train `
  model=yolo11s.pt `
  data="D:/mine/bjtu/交通模型预测/homework/groupmate_unified_dataset_v1/groupmate_unified_dataset_v1.yaml" `
  imgsz=640 `
  epochs=100 `
  batch=2 `
  device=0 `
  workers=0 `
  amp=False `
  cache=False `
  patience=30 `
  cos_lr=False `
  close_mosaic=0 `
  degrees=0.0 `
  shear=0.0 `
  perspective=0.0 `
  translate=0.05 `
  scale=0.15 `
  fliplr=0.5 `
  flipud=0.0 `
  hsv_h=0.010 `
  hsv_s=0.40 `
  hsv_v=0.25 `
  mosaic=0.0 `
  mixup=0.0 `
  erasing=0.0 `
  project="D:/mine/bjtu/交通模型预测/homework/groupmate_unified_dataset_v1/runs" `
  name="train_yolo11s_groupmate_v1_safe" `
  exist_ok=True
```

### 4.3 训练产物

本次训练的全部输出已归档到：

- `training/train_yolo11s_groupmate_v1_safe`

其中保留的关键文件包括：

- `weights/best.pt`
- `results.csv`
- `results.png`
- `BoxPR_curve.png`
- `BoxF1_curve.png`
- `confusion_matrix_normalized.png`
- `labels.jpg`
- `train_batch0.jpg`
- `val_batch0_labels.jpg`
- `val_batch0_pred.jpg`

## 5. 实验结果

### 5.1 关键指标

根据 `training/train_yolo11s_groupmate_v1_safe/results.csv` 统计：

- 最佳 `mAP50`：`0.995`，出现在第 `4` 轮
- 对应 `Precision`：`0.88194`
- 对应 `Recall`：`0.97905`
- 最佳 `mAP50-95`：`0.60396`，出现在第 `30` 轮
- 最终轮次 `Precision`：`0.99698`
- 最终轮次 `Recall`：`1.00000`
- 最终轮次 `mAP50`：`0.99500`
- 最终轮次 `mAP50-95`：`0.57902`

### 5.2 结果分析

从整体结果看，模型在这套重新整理后的数据集上收敛较好。

最终 `mAP50` 保持在较高水平，说明更新后的训练集与验证集标签口径已经明显比之前跨场景失败实验更一致。

同时，`mAP50-95` 也明显优于之前几次失败实验，说明模型不仅分类效果较好，边界框回归质量也比较稳定。

## 6. 预测结果

### 6.1 验证集图片预测

带框、带标签的验证集预测样例图片保存在：

- `prediction/images/samples`

### 6.2 视频预测结果

带框、带标签、带置信度的视频预测结果保存在：

- `prediction/video/video_with_boxes_yolo11s/f595b632d8a4323366a5ea323fadf168.avi`

### 6.3 预测命令

图片预测命令：

```powershell
D:\ai-envs\tl-gpu-cu124\Scripts\yolo.exe detect predict `
  model="D:/mine/bjtu/交通模型预测/homework/groupmate_unified_dataset_v1/runs/train_yolo11s_groupmate_v1_safe/weights/best.pt" `
  source="D:/mine/bjtu/交通模型预测/homework/groupmate_unified_dataset_v1/images/val" `
  conf=0.25 `
  save=True `
  show_labels=True `
  show_conf=True `
  project="D:/mine/bjtu/交通模型预测/homework/groupmate_unified_dataset_v1/predicts" `
  name="val_images_with_boxes_yolo11s" `
  exist_ok=True
```

视频预测命令：

```powershell
D:\ai-envs\tl-gpu-cu124\Scripts\yolo.exe detect predict `
  model="D:/mine/bjtu/交通模型预测/homework/groupmate_unified_dataset_v1/runs/train_yolo11s_groupmate_v1_safe/weights/best.pt" `
  source="D:/mine/bjtu/交通模型预测/homework/f595b632d8a4323366a5ea323fadf168.mp4" `
  conf=0.25 `
  save=True `
  show_labels=True `
  show_conf=True `
  project="D:/mine/bjtu/交通模型预测/homework/groupmate_unified_dataset_v1/predicts" `
  name="video_with_boxes_yolo11s" `
  exist_ok=True
```

## 7. 文件夹结构

```text
yolo11s/
├─ README.md
├─ scripts/
│  └─ prepare_groupmate_unified_dataset.py
├─ configs/
│  └─ groupmate_unified_dataset_v1.yaml
├─ training/
│  └─ train_yolo11s_groupmate_v1_safe/
├─ prediction/
│  ├─ images/
│  │  └─ samples/
│  └─ video/
│     └─ video_with_boxes_yolo11s/
└─ dataset_preview/
   └─ train/
      └─ train01.jpg
```

## 8. 备注

- 本归档文件夹仅保留本次最终成功的 `YOLO11s` 实验结果。
- 此处未完整重复存放全部原始数据集，也未保留全部验证集预测图片，只保留了配置文件、脚本、关键结果文件、少量样例图和最终视频，便于实验汇报和 GitHub 上传。
- 如有需要，可直接复用归档中的 `best.pt` 继续进行图片或视频推理。
