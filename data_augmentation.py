import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
import random

class DatasetAugmentation:
    def __init__(self):
        # 定义基础增强流程
        self.transform = A.Compose([
            # 1. 几何变换
            # A.OneOf([
            #     A.HorizontalFlip(p=0.5),
            #     A.VerticalFlip(p=0.3),
            #     A.RandomRotate90(p=0.5),
            #     A.Affine(
            #         scale=(0.7, 1.3),
            #         translate_percent=(-0.2, 0.2),
            #         rotate=(-30, 30),
            #         p=0.7
            #     )
            # ], p=0.8),
            
            # 2. 颜色变换
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=25,
                    val_shift_limit=25,
                    p=0.5
                )
            ], p=0.8),
            
            # 3. 图像质量和噪声
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussNoise(
                    mean=0,
                    var=20,
                    p=0.3
                )
            ], p=0.4),
        ], 
        # 添加关键点参数以支持多边形标注
        keypoint_params=A.KeypointParams(
            format='xy',  # 坐标格式
            label_fields=['class_labels'],  # 标签字段
            remove_invisible=False,  # 保留不可见点
            angle_in_degrees=True  # 角度使用度数
        ))

    def process_labelme_annotation(self, label_data, image_shape):
        """处理 LabelMe 格式的标注数据"""
        polygons = []
        labels = []
        
        for shape in label_data['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'])
                polygons.append(points)
                labels.append(shape['label'])
                
        return polygons, labels

    def augment_dataset(self, data_dir, output_dir, num_augmentations=3):
        """
        对数据集进行增强
        :param data_dir: 原始数据集路径
        :param output_dir: 输出路径
        :param num_augmentations: 每张图像增强的次数
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        
        # 创建输出目录
        output_images_dir = output_dir / 'images'
        output_labels_dir = output_dir / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        # 复制原始数据
        print("复制原始数据...")
        for img_path in data_dir.glob('*.jpg'):
            shutil.copy(str(img_path), str(output_images_dir / img_path.name))
            json_path = data_dir.parent / 'labels' / (img_path.stem + '.json')
            if json_path.exists():
                shutil.copy(str(json_path), str(output_labels_dir / json_path.name))

        # 对每张图像进行增强
        total_images = len(list(data_dir.glob('*.jpg')))
        for idx, img_path in enumerate(data_dir.glob('*.jpg'), 1):
            print(f"\n处理图片 [{idx}/{total_images}]: {img_path.name}")
            
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"警告: 无法读取图片 {img_path}")
                continue
            
            # 读取标注文件
            json_path = data_dir.parent / 'labels' / (img_path.stem + '.json')
            if not json_path.exists():
                print(f"警告: 找不到标注文件 {img_path}")
                continue

            try:
                with open(json_path) as f:
                    label_data = json.load(f)

                # 处理每个增强实例
                for i in range(num_augmentations):
                    try:
                        # 准备关键点数据
                        keypoints = []
                        class_labels = []
                        shape_lengths = []  # 记录每个多边形的点数
                        
                        # 收集所有多边形的点
                        for shape in label_data['shapes']:
                            if shape['shape_type'] == 'polygon':
                                points = shape['points']
                                keypoints.extend(points)
                                class_labels.extend([shape['label']] * len(points))
                                shape_lengths.append(len(points))

                        # 应用增强
                        transformed = self.transform(
                            image=image,
                            keypoints=keypoints,
                            class_labels=class_labels
                        )

                        aug_image = transformed['image']
                        aug_keypoints = transformed['keypoints']
                        
                        # 重建标注数据
                        aug_label_data = label_data.copy()
                        aug_label_data['imagePath'] = f"{img_path.stem}_aug_{i+1}.jpg"
                        aug_label_data['imageHeight'] = aug_image.shape[0]
                        aug_label_data['imageWidth'] = aug_image.shape[1]

                        # 更新多边形点
                        start_idx = 0
                        for j, shape in enumerate(aug_label_data['shapes']):
                            if shape['shape_type'] == 'polygon':
                                end_idx = start_idx + shape_lengths[j]
                                shape['points'] = aug_keypoints[start_idx:end_idx]
                                start_idx = end_idx

                        # 保存增强后的图像和标注
                        aug_img_name = f"{img_path.stem}_aug_{i+1}.jpg"
                        aug_json_name = f"{img_path.stem}_aug_{i+1}.json"
                        
                        cv2.imwrite(str(output_images_dir / aug_img_name), aug_image)
                        with open(output_labels_dir / aug_json_name, 'w', encoding='utf-8') as f:
                            json.dump(aug_label_data, f, indent=2, ensure_ascii=False)
                            
                        print(f"  - 已生成增强图像 {aug_img_name}")
                        
                    except Exception as e:
                        print(f"  - 警告: 处理增强 {i+1} 时出错: {str(e)}")
                        continue

            except Exception as e:
                print(f"错误: 处理图片 {img_path.name} 时出错: {str(e)}")
                continue

        print("\n数据增强完成!")

    def convert_labelme_to_yolo(self, json_path, output_path, image_width, image_height):
        """将 LabelMe 格式的标注转换为 YOLO 格式"""
        with open(json_path, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for shape in labelme_data['shapes']:
                if shape['shape_type'] != 'polygon':
                    continue
                    
                # 获取类别ID (这里假设只有一个类别 'mg')
                class_id = 0
                
                # 获取多边形顶点
                points = np.array(shape['points'])
                
                # 归一化坐标
                points[:, 0] = points[:, 0] / image_width
                points[:, 1] = points[:, 1] / image_height
                
                # 确保坐标在[0,1]范围内
                points = np.clip(points, 0, 1)
                
                # 写入YOLO格式
                f.write(f"{class_id}")
                for point in points:
                    f.write(f" {point[0]:.6f} {point[1]:.6f}")
                f.write("\n")

    def split_dataset(self, dataset_dir, train_ratio=0.7, val_ratio=0.2):
        """将数据集分割为训练集、验证集和测试集"""
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'

        # 获取所有图像文件
        all_images = list(images_dir.glob('*.jpg'))
        random.shuffle(all_images)

        # 计算分割点
        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # 分割数据集
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]

        print(f"\n开始数据集分割...")
        print(f"总图像数: {n_total}")
        print(f"训练集: {len(train_images)}")
        print(f"验证集: {len(val_images)}")
        print(f"测试集: {len(test_images)}")

        # 创建并移动文件
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        for split_name, split_images in splits.items():
            split_dir = dataset_dir / split_name
            split_images_dir = split_dir / 'images'
            split_labels_dir = split_dir / 'labels'
            
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n处理{split_name}集...")
            for img_path in split_images:
                try:
                    # 移动图像
                    shutil.move(str(img_path), str(split_images_dir / img_path.name))
                    
                    # 处理标签文件
                    json_path = labels_dir / (img_path.stem + '.json')
                    txt_path = split_labels_dir / (img_path.stem + '.txt')
                    
                    if json_path.exists():
                        # 读取图像尺寸
                        img = cv2.imread(str(split_images_dir / img_path.name))
                        if img is None:
                            print(f"警告: 无法读取图片 {img_path}")
                            continue
                            
                        height, width = img.shape[:2]
                        
                        # 转换并保存YOLO格式标签
                        self.convert_labelme_to_yolo(json_path, txt_path, width, height)
                        print(f"  - 已处理: {img_path.name}")
                    
                except Exception as e:
                    print(f"错误: 处理文件 {img_path.name} 时出错: {str(e)}")
                    continue

        print("\n数据集分割完成!")

if __name__ == "__main__":
    augmentor = DatasetAugmentation()
    
    # 数据增强
    print("开始数据增强...")
    augmentor.augment_dataset(
        data_dir='data/images',
        output_dir='data/augmented',
        num_augmentations=25
    )
    
    # 数据集分割
    print("\n开始数据集分割...")
    augmentor.split_dataset(
        dataset_dir='data/augmented',
        train_ratio=0.8,
        val_ratio=0.1
    ) 