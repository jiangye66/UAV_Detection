import cv2
import json
import os
import numpy as np
from shutil import rmtree  # 用于清空文件夹

# 文件路径
video_path = 'E:/datasets/anti-uav/test-dev/20190925_101846_1_1/RGB.mp4'
json_path = 'E:/datasets/anti-uav/test-dev/20190925_101846_1_1/RGB_label.json'

# 设置随机种子
np.random.seed(42)

# 提取视频所在目录和文件名
video_dir = os.path.dirname(os.path.abspath(video_path))
video_name = os.path.splitext(os.path.basename(video_path))[0]

# 主目录路径
main_dir = video_dir

# 创建训练和验证集的目录
train_images_dir = os.path.join(main_dir, 'train', 'images')
train_labels_dir = os.path.join(main_dir, 'train', 'labels')
val_images_dir = os.path.join(main_dir, 'val', 'images')
val_labels_dir = os.path.join(main_dir, 'val', 'labels')

# 如果目录已存在，先清空它们
for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    if os.path.exists(dir_path):
        rmtree(dir_path)
    os.makedirs(dir_path)

# 读取标注文件
with open(json_path, 'r') as json_file:
    data = json.load(json_file)

exist_data = data["exist"]
gt_rect_data = data["gt_rect"]
frame_count = len(exist_data)

# 生成随机索引
indices = np.arange(frame_count)
np.random.shuffle(indices)

# 划分训练集和验证集
train_size = int(frame_count * 0.8)
train_indices = set(indices[:train_size])
val_indices = set(indices[train_size:])

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("无法打开视频文件！")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频帧大小: {frame_width}x{frame_height}")
print(f"总帧数: {frame_count}")
print(f"训练集帧数: {len(train_indices)}")
print(f"验证集帧数: {len(val_indices)}")

# 处理每一帧
for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        print(f"警告：在读取第 {frame_idx} 帧时失败")
        break

    # 根据frame_idx决定是训练集还是验证集
    if frame_idx in train_indices:
        images_dir = train_images_dir
        labels_dir = train_labels_dir
    else:
        images_dir = val_images_dir
        labels_dir = val_labels_dir

    # 保存图片
    image_path = os.path.join(images_dir, f"{video_name}_{frame_idx:04d}.png")
    cv2.imwrite(image_path, frame)

    # 处理标注数据
    exist_value = exist_data[frame_idx]
    rect_values = gt_rect_data[frame_idx] if gt_rect_data[frame_idx] else [0, 0, 0, 0]

    x_min, y_min, width, height = rect_values

    # 转换为YOLO格式
    center_x = (x_min + width / 2) / frame_width
    center_y = (y_min + height / 2) / frame_height
    width_ratio = width / frame_width
    height_ratio = height / frame_height

    # 保存标注文件
    label_path = os.path.join(labels_dir, f"{video_name}_{frame_idx:04d}.txt")
    with open(label_path, 'w') as f:
        f.write(f"{exist_value} {center_x:.6f} {center_y:.6f} {width_ratio:.6f} {height_ratio:.6f}\n")

    if (frame_idx + 1) % 50 == 0:
        print(f"已处理 {frame_idx + 1}/{frame_count} 帧...")

# 释放资源
cap.release()

# 验证文件数量
train_images = len(os.listdir(train_images_dir))
val_images = len(os.listdir(val_images_dir))
print(f"\n最终统计:")
print(f"训练集图片数量: {train_images}")
print(f"验证集图片数量: {val_images}")
print("处理完成！数据集已按8:2的比例随机分配为训练集和验证集。")