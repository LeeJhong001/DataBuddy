import os
import shutil

# 定义源文件夹路径和目标文件夹路径
source_folder = r"C:\Users\leezh\Desktop\DataBuddy\bean"  # 替换为你的文件夹路径
output_folder = "data"  # 替换为目标文件夹路径

# 创建目标文件夹
os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)

    if os.path.isfile(file_path):
        # 根据文件后缀名判断是图片还是 JSON 文件
        if filename.endswith(".jpg"):
            # 移动 JPG 文件到 images 文件夹
            shutil.move(file_path, os.path.join(output_folder, "images", filename))
        elif filename.endswith(".json"):
            # 移动 JSON 文件到 labels 文件夹
            shutil.move(file_path, os.path.join(output_folder, "labels", filename))

print("文件处理完成！")