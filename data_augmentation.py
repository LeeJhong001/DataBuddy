import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
import random
import os

class DatasetAugmentation:
    def __init__(self):
        # 定义基础增强流程
        self.augmentation_methods = [
            # 1. 几何变换
            lambda img: A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.2, 0.2),
                rotate=(-30, 30),
                p=1.0
            )(image=img)['image'],
            
            # 2. 颜色变换
            lambda img: A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1.0
                )
            ], p=1.0)(image=img)['image'],
            
            # 3. 噪声和模糊
            lambda img: A.OneOf([
                A.GaussNoise(p=1.0),  # 使用默认参数
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0)
            ], p=1.0)(image=img)['image'],
            
            # 4. 质量压缩
            lambda img: A.ImageCompression(
                quality=80,  # 设置JPEG压缩质量
                p=1.0
            )(image=img)['image']
        ]

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

    def check_disk_space(self, path, required_space_mb):
        """检查磁盘空间是否足够"""
        try:
            total, used, free = shutil.disk_usage(path)
            free_mb = free / (1024 * 1024)  # 转换为MB
            if free_mb < required_space_mb:
                print(f"警告: 磁盘空间不足! 可用: {free_mb:.1f}MB, 需要: {required_space_mb:.1f}MB")
                return False
            return True
        except Exception as e:
            print(f"警告: 检查磁盘空间时出错: {str(e)}")
            return False

    def estimate_required_space(self, data_dir, num_augmentations):
        """估算所需磁盘空间"""
        try:
            total_size = 0
            for img_path in Path(data_dir).glob('*.jpg'):
                total_size += os.path.getsize(img_path)
            
            # 估算增强后需要的空间 (原始数据 + 增强数据 + 额外缓冲)
            estimated_space = total_size * (num_augmentations + 1) * 1.2
            return estimated_space / (1024 * 1024)  # 转换为MB
        except Exception as e:
            print(f"警告: 估算空间需求时出错: {str(e)}")
            return 1000  # 默认返回1GB作为安全值

    def augment_dataset(self, data_dir, output_dir, num_augmentations=3):
        """数据增强主函数"""
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        
        # 估算所需空间并检查
        required_space = self.estimate_required_space(data_dir, num_augmentations)
        if not self.check_disk_space(output_dir.parent, required_space):
            print("错误: 磁盘空间不足，无法继续处理")
            return False

        # 创建输出目录
        output_images_dir = output_dir / 'images'
        output_labels_dir = output_dir / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        # 处理所有图像
        image_files = list(data_dir.glob('*.jpg'))
        total_images = len(image_files)
        
        print(f"开始处理 {total_images} 张图像...")
        for idx, img_path in enumerate(image_files, 1):
            print(f"处理图片 [{idx}/{total_images}]: {img_path.name}")
            
            # 检查剩余空间
            if not self.check_disk_space(output_dir, required_space / total_images):
                print("错误: 磁盘空间不足，停止处理")
                return False

            try:
                # 读取原始图像
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"警告: 无法读取图片 {img_path}")
                    continue

                # 处理原始图像的标注
                json_path = data_dir.parent / 'labels' / f"{img_path.stem}.json"
                if not json_path.exists():
                    print(f"警告: 找不到标注文件 {json_path}")
                    continue

                # 保存原始图像（使用JPEG压缩以节省空间）
                orig_img_path = output_images_dir / img_path.name
                cv2.imwrite(str(orig_img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # 转换并保存原始标注
                orig_txt_path = output_labels_dir / f"{img_path.stem}.txt"
                if not self.convert_labelme_to_yolo(json_path, orig_txt_path, image.shape[1], image.shape[0]):
                    print(f"警告: 转换标注失败 {json_path}")
                    continue

                # 生成增强图像
                for i in range(num_augmentations):
                    try:
                        # 应用随机增强
                        aug_image = image.copy()
                        # 随机选择1-3个增强方法
                        selected_methods = random.sample(
                            self.augmentation_methods, 
                            random.randint(1, min(3, len(self.augmentation_methods)))
                        )
                        
                        # 依次应用选中的增强方法
                        for method in selected_methods:
                            aug_image = method(aug_image)

                        # 保存增强后的图像和对应的标注
                        aug_name = f"{img_path.stem}_aug_{i+1}"
                        aug_img_path = output_images_dir / f"{aug_name}.jpg"
                        aug_txt_path = output_labels_dir / f"{aug_name}.txt"
                        
                        # 保存增强后的图像
                        cv2.imwrite(str(aug_img_path), aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        # 复制原始标注（因为我们的增强不改变标注位置）
                        shutil.copy(str(orig_txt_path), str(aug_txt_path))
                        
                        print(f"  - 已生成增强图像: {aug_name}")
                        
                    except Exception as e:
                        print(f"警告: 处理增强 {i+1} 时出错: {str(e)}")
                        continue

            except Exception as e:
                print(f"错误: 处理图片 {img_path.name} 时出错: {str(e)}")
                continue

        print("数据增强完成!")
        return True

    def convert_labelme_to_yolo(self, json_path, output_path, image_width, image_height):
        """将 LabelMe 格式的标注转换为 YOLO 格式"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            
            yolo_lines = []
            for shape in labelme_data['shapes']:
                if shape['shape_type'] != 'rectangle':
                    continue
                    
                # 获取类别ID (这里假设标签为'bean'的为0类)
                label = shape['label'].lower().strip()
                class_id = 0 if label == 'bean' else -1
                
                if class_id == -1:
                    continue
                
                points = shape['points']
                if len(points) != 4:  # 矩形框必须是4个点
                    continue
                    
                # LabelMe的矩形框点的顺序：左上、右上、右下、左下
                # 我们只需要左上角和右下角的点来计算中心点和宽高
                x1, y1 = points[0]  # 左上角
                x2, y2 = points[2]  # 右下角
                
                # 计算中心点坐标和宽高
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                # 归一化坐标
                center_x = center_x / image_width
                center_y = center_y / image_height
                width = width / image_width
                height = height / image_height
                
                # 确保坐标在[0,1]范围内
                center_x = np.clip(center_x, 0, 1)
                center_y = np.clip(center_y, 0, 1)
                width = np.clip(width, 0, 1)
                height = np.clip(height, 0, 1)
                
                # 写入YOLO格式: class_id center_x center_y width height
                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            if yolo_lines:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
                return True
            return False
            
        except Exception as e:
            print(f"错误: 转换文件 {json_path} 时出错: {str(e)}")
            return False

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
                    
                    # 移动对应的标签文件（.txt文件）
                    txt_path = labels_dir / f"{img_path.stem}.txt"
                    if txt_path.exists():
                        shutil.move(str(txt_path), str(split_labels_dir / txt_path.name))
                        print(f"  - 已处理: {img_path.name} 及其标签文件")
                    else:
                        print(f"警告: 找不到标签文件 {txt_path}")
                    
                except Exception as e:
                    print(f"错误: 处理文件 {img_path.name} 时出错: {str(e)}")
                    continue

        print("\n数据集分割完成!")

if __name__ == "__main__":
    augmentor = DatasetAugmentation()
    
    # 数据增强
    print("开始数据增强...")
    if augmentor.augment_dataset(
        data_dir='data/images',
        output_dir='data/augmented',
        num_augmentations=25
    ):
        # 只有在数据增强成功时才进行数据集分割
        print("\n开始数据集分割...")
        augmentor.split_dataset(
            dataset_dir='data/augmented',
            train_ratio=0.8,
            val_ratio=0.1
        )
    else:
        print("数据增强失败，程序终止")