# 豆类检测数据集处理工具

一个用于处理豆类检测数据集的工具，支持数据预处理、数据增强和数据集分割。

## 快速开始

### 1. 安装依赖
```bash
pip install albumentations opencv-python numpy
```

### 2. 准备数据
1. 将所有图片和标注文件放在同一个文件夹中
2. 运行数据预处理脚本进行文件整理：
```bash
python divide_dataset.py
```

### 3. 运行数据增强和分割
```bash
python data_augmentation.py
```

## 目录结构

```
project/
├── data/
│   ├── images/        # 原始图像
│   ├── labels/        # 原始标注
│   └── augmented/     # 增强后的数据
│       ├── train/     # 训练集
│       ├── val/       # 验证集
│       └── test/      # 测试集
├── data_augmentation.py
├── divide_dataset.py
└── README.md
```

## 功能特性

### 数据预处理
- 自动整理图片(.jpg)和标注文件(.json)
- 创建标准化的目录结构

### 数据增强
支持多种增强方式的组合：

1. 几何变换
   - 缩放: 0.8-1.2
   - 平移: ±20%
   - 旋转: ±30°

2. 颜色变换
   - 亮度/对比度: ±20%
   - 色调/饱和度/明度
   - RGB通道偏移

3. 噪声和模糊
   - 高斯噪声
   - 高斯模糊
   - 运动模糊

4. 质量压缩
   - JPEG压缩质量: 80%

### 数据集分割
- 训练集: 80%
- 验证集: 10%
- 测试集: 10%

## 数据要求

### 图片要求
- 格式: JPG
- 分辨率: 建议保持一致

### 标注要求
- 格式: LabelMe JSON
- 标注类型: 矩形框
- 标签名称: "bean"

## 配置说明

### 数据增强配置
```python
# data_augmentation.py
num_augmentations=25  # 每张图片增强数量
```

### 数据集分割配置
```python
# data_augmentation.py
train_ratio=0.8  # 训练集比例
val_ratio=0.1    # 验证集比例
```

## 输出格式

### 图片文件
- 格式: `原文件名_aug_序号.jpg`
- 位置: `data/augmented/[train|val|test]/images/`

### 标注文件
- 格式: `原文件名_aug_序号.txt`
- 位置: `data/augmented/[train|val|test]/labels/`
- YOLO格式: `class_id center_x center_y width height`
  - class_id: 0 (bean)
  - 所有坐标已归一化到[0,1]

## 注意事项

1. 确保磁盘空间充足（建议预留原始数据集30倍空间）
2. 程序会自动跳过处理失败的图片
3. 建议使用Python 3.6+版本

## 错误处理

- 所有错误和警告都会被记录并打印
- 单个文件处理失败不会影响其他文件
- 可以查看控制台输出了解处理进度和错误信息

## 系统要求

- Python 3.6+
- Windows/Linux/MacOS
- 足够的磁盘空间