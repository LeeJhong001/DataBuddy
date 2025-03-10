# 数据集处理工具说明

本仓库包含两个Python脚本：`data_augmentation.py` 和 `divide_dataset.py`，用于自动化数据增强、标注格式转换及数据集管理。以下是详细功能介绍和使用说明。

---

## 文件目录结构
运行前需确保原始数据按以下结构组织：

data/
├── images/ # 存放原始图像（.jpg）
└── labels/ # 存放LabelMe格式标注文件（.json）

---

## 脚本功能说明

###  `divide_dataset.py`
#### **功能**
- 将用户提供的原始数据（图片和JSON标注文件）按类别自动整理到标准化目录中。
- 原始数据中的 `.jpg` 文件会被移动到 `data/images`，`.json` 文件会被移动到 `data/labels`。

#### **使用场景**
- 当原始数据混合存放时，用于快速初始化标准化的数据集目录结构。

#### **参数配置**
- `source_folder`: 原始数据存放路径（需用户自定义，如 `C:\Users\leezh\Desktop\子实体`）。
- `output_folder`: 标准化输出目录（默认为 `data`）。

#### **运行方式**
```python
python divide_dataset.py
```

###  `data_augmentation.py`

#### **核心功能**

1. **数据增强**
   - 支持多种增强操作，包括：
     - **颜色变换**：亮度/对比度调整、CLAHE直方图均衡化、色调/饱和度调整。
     - **噪声与模糊**：高斯模糊、运动模糊、中值模糊、高斯噪声。
   - 支持多边形标注的同步变换（如目标检测任务中的物体轮廓）。
2. **数据集分割**
   - 将增强后的数据按比例分割为训练集、验证集、测试集。
   - 自动生成 `train/`、`val/`、`test/` 子目录。
3. **标注格式转换**
   - 将LabelMe格式（JSON）的标注转换为YOLO格式（.txt），支持多边形坐标归一化。

#### **类与方法说明**

- **`DatasetAugmentation` 类**
  - `augment_dataset()`: 对数据集进行增强，生成多组增强图像及标注。
    - 参数：
      - `data_dir`: 输入数据路径（如 `data/images`）。
      - `output_dir`: 增强数据输出路径（如 `data/augmented`）。
      - `num_augmentations`: 每张图像的增强次数（默认3次）。
  - `split_dataset()`: 分割数据集。
    - 参数：
      - `dataset_dir`: 增强后的数据集路径。
      - `train_ratio` 和 `val_ratio`: 训练集与验证集比例（测试集自动计算剩余部分）。

#### **运行方式**

```python
python data_augmentation.py
```

##  依赖安装

确保安装以下Python库：

```python
pip install albumentations opencv-python numpy jsonlib-python3 shutil random pathlib
```

## 使用流程示例

1. **整理原始数据**
   运行 `divide_dataset.py`，将原始数据移动到 `data/images` 和 `data/labels`。
2. **执行数据增强**
   运行 `data_augmentation.py`，生成增强数据到 `data/augmented`。
3. **查看结果**
   - 增强后的数据保存在 `augmented/images` 和 `augmented/labels`。
   - 分割后的数据集保存在 `augmented/train`、`augmented/val`、`augmented/test`。

------

## 注意事项

- **路径配置**：需根据实际环境修改 `divide_dataset.py` 中的 `source_folder`。
- **标注兼容性**：当前支持LabelMe的多边形标注（`shape_type: 'polygon'`）。
- **增强失败处理**：若某次增强出错（如坐标越界），脚本会自动跳过并继续处理下一张图像。

------

## 扩展性

- **自定义增强策略**：可通过修改 `data_augmentation.py` 中的 `transform` 组合调整增强流程。
- **多类别支持**：若需支持多类别，需修改 `convert_labelme_to_yolo()` 中的类别ID映射逻辑。