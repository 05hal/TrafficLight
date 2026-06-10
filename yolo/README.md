# YOLO11 Traffic Light Experiments

## 1. 说明

本文件夹用于整理本项目中与 YOLO11 训练相关的内容，包括：

- 训练用脚本
- 数据集配置文件
- 一次失败训练的结果快照
- 一次成功训练的结果快照
- 对失败原因和成功原因的分析

本次任务的目标是对交通灯进行目标检测与分类，类别为：

- `red`
- `yellow`
- `green`

本整理文件夹对应的核心结论是：

- 原始 `train_imgs + test_imgs` 直接组合训练时，效果很差
- 按老师示例的标注风格统一为“框整个交通灯单元”后，YOLO11s 训练效果显著提升

## 2. 文件结构

```text
yolo/
├─ README.md
├─ scripts/
│  ├─ extract_adaptive_frame.py
│  ├─ prepare_yolo11_dataset.py
│  └─ prepare_trafficlight_unit_dataset.py
├─ configs/
│  ├─ dataset.yaml
│  └─ trafficlight_unit_dataset_v1.yaml
└─ results/
   ├─ failed/
   │  ├─ args.yaml
   │  ├─ results.csv
   │  ├─ results.png
   │  ├─ confusion_matrix.png
   │  ├─ val_batch0_labels.jpg
   │  └─ val_batch0_pred.jpg
   └─ successful/
      ├─ args.yaml
      ├─ results.csv
      ├─ results.png
      ├─ confusion_matrix.png
      ├─ val_batch0_labels.jpg
      └─ val_batch0_pred.jpg
```

## 3. 脚本作用

### `scripts/extract_adaptive_frame.py`

作用：

- 从视频中自适应抽帧
- 根据相邻时间段画面变化幅度决定每秒抽取 5 到 10 帧
- 用于从原始交通灯视频中生成训练图片

典型命令：

```bash
python yolo/scripts/extract_adaptive_frame.py
```

说明：

- 脚本默认处理 `train.mp4` 和 `test.mp4`
- 输出目录默认为 `train_imgs/` 和 `test_imgs/`

### `scripts/prepare_yolo11_dataset.py`

作用：

- 将原始 `train_imgs/` 和 `test_imgs/` 整理为 YOLO 目录结构
- 生成第一次实验使用的配置文件
- 对应的是“失败训练”的数据准备方式

典型命令：

```bash
python yolo/scripts/prepare_yolo11_dataset.py
```

输出：

- 数据目录：`yolo11_dataset/`
- 配置文件：`dataset.yaml`

### `scripts/prepare_trafficlight_unit_dataset.py`

作用：

- 只使用与老师示例标注风格一致的样本
- 按“框整个交通灯单元”的口径重新划分 `train/val`
- 生成第二次实验使用的配置文件
- 对应的是“成功训练”的数据准备方式

典型命令：

```bash
python yolo/scripts/prepare_trafficlight_unit_dataset.py
```

输出：

- 数据目录：`trafficlight_unit_dataset_v1/`
- 配置文件：`trafficlight_unit_dataset_v1.yaml`

## 4. 环境准备

建议在项目根目录 `TrafficLight-main/` 下执行下列命令：

```bash
conda activate D:\ai-envs\tl-gpu-cu124
cd D:\mine\bjtu\交通模型预测\homework\TrafficLight-main
```

如果尚未安装 Ultralytics：

```bash
python -m pip install ultralytics
```

### 本机运行配置

以下配置为本次训练结果对应的实际本机环境，可用于实验报告中的“实验环境”部分：

- 操作系统：Windows
- Python 环境：`conda`，环境路径为 `D:\ai-envs\tl-gpu-cu124`
- Python 版本：`3.12.13`
- PyTorch 版本：`2.5.1`
- Torchvision 版本：`0.20.1`
- Ultralytics 版本：`8.4.60`
- CUDA 可用性：`torch.cuda.is_available() = True`
- PyTorch 使用的 CUDA 版本：`12.4`
- 显卡：`NVIDIA GeForce RTX 4070 Laptop GPU`
- 显存：`8 GB`
- NVIDIA 驱动版本：`591.74`
- `nvidia-smi` 显示驱动支持 CUDA 版本：`13.1`

对应的环境检查命令如下：

```bash
python -c "import torch, torchvision; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'); print('torchvision', torchvision.__version__)"
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
nvidia-smi
```

## 5. 第一次训练：失败实验

### 数据准备命令

```bash
python yolo/scripts/prepare_yolo11_dataset.py
```

### 训练命令

```bash
yolo detect train model=yolo11s.pt data=yolo/configs/dataset.yaml imgsz=640 batch=4 device=0 workers=0 amp=False cache=False
```

### 历史结果目录

- 原始结果目录：`runs/detect/train/`
- 本文件夹内快照：`yolo/results/failed/`

### 最终结果

根据 `results.csv` 最后一轮结果：

- `Precision = 0`
- `Recall = 0`
- `mAP50 = 0`
- `mAP50-95 = 0`

### 失败现象

- 训练损失虽然下降，但验证指标几乎始终为 0
- 混淆矩阵中大量真值目标被判为背景
- 预测框与真实框风格不匹配

### 失败原因

- 原始 `train_imgs` 与 `test_imgs` 标注口径不一致
- `train_imgs` 更像框亮灯区域或较小目标
- `test_imgs` 更像框整个交通灯单元或较大目标
- 训练集与验证集目标尺度差异过大
- 训练集与验证集的目标数量、构图和场景分布差异明显
- 模型训练时学到的是一种任务，验证时却按另一种任务评估，导致 IoU 和 mAP 极低

## 6. 第二次训练：成功实验

### 数据准备命令

```bash
python yolo/scripts/prepare_trafficlight_unit_dataset.py
```

### 训练命令

```bash
yolo detect train model=yolo11s.pt data=yolo/configs/trafficlight_unit_dataset_v1.yaml imgsz=640 batch=4 device=0 workers=0 amp=False cache=False
```

### 历史结果目录

- 原始结果目录：`runs/detect/train-2/`
- 本文件夹内快照：`yolo/results/successful/`

### 最终结果

根据 `results.csv` 最后一轮结果：

- `Precision = 0.99532`
- `Recall = 1.00000`
- `mAP50 = 0.99500`
- `mAP50-95 = 0.80213`

### 成功现象

- 训练与验证损失整体下降，收敛稳定
- `precision`、`recall`、`mAP50` 很早就达到较高水平
- 验证图中的预测框和标注框基本一致
- 红、黄、绿三类分类结果较稳定

### 成功原因

- 重新统一了标注口径，改为与老师示例一致的“框整个交通灯单元”
- 训练集和验证集来自同一类标注风格，任务定义一致
- 新的数据划分减少了风格冲突，提升了数据分布一致性
- YOLO11s 在统一风格数据上能够快速收敛并学到稳定特征

## 7. 两次实验的对比结论

第一次失败并不是因为 YOLO11s 不适合该任务，而是因为原始数据集存在明显的数据质量问题：

- 标注风格不一致
- 目标尺度不一致
- 场景分布不一致

第二次成功说明：

- 对目标检测任务来说，数据集的标注一致性比单纯增加图片数量更重要
- 当训练集和验证集遵循同一种框选标准时，即使样本数量不大，模型也能取得较好效果

## 8. 建议

- 后续如果继续扩展真实视频数据，应统一采用“框整个交通灯单元”的标注方式
- 新增样本时应尽量保持 train/val 同分布
- 如果需要进一步验证泛化能力，建议在不同天气、不同光照、不同距离的真实视频上继续测试
