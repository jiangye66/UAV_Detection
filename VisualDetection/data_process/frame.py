import cv2
import json
import os
import numpy as np
from shutil import rmtree


def process_video(video_path, json_path, output_base_dir, video_name):
    """处理单个视频文件"""
    # 设置随机种子
    np.random.seed(42)

    # 创建输出目录
    train_images_dir = os.path.join(output_base_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_base_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_base_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_base_dir, 'val', 'labels')

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

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
        print(f"警告：无法打开视频文件 {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"处理视频: {video_path}")
    print(f"视频帧大小: {frame_width}x{frame_height}")
    print(f"总帧数: {frame_count}")

    # 处理每一帧
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"警告：在读取第 {frame_idx} 帧时失败")
            break

        # 确定保存路径
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

    cap.release()


def process_dataset(base_dir):
    """处理整个数据集"""
    # 创建输出目录
    output_ir_dir = os.path.join(base_dir, 'IR')
    output_rgb_dir = os.path.join(base_dir, 'RGB')

    # 清空并创建输出目录
    for dir_path in [output_ir_dir, output_rgb_dir]:
        if os.path.exists(dir_path):
            rmtree(dir_path)
        os.makedirs(dir_path)
        os.makedirs(os.path.join(dir_path, 'train', 'images'))
        os.makedirs(os.path.join(dir_path, 'train', 'labels'))
        os.makedirs(os.path.join(dir_path, 'val', 'images'))
        os.makedirs(os.path.join(dir_path, 'val', 'labels'))

    # 遍历所有数据集目录
    test_dir = os.path.join(base_dir, 'test')
    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        if os.path.isdir(dataset_path):
            # 处理IR视频
            ir_video_path = os.path.join(dataset_path, 'IR.mp4')
            ir_json_path = os.path.join(dataset_path, 'IR_label.json')
            if os.path.exists(ir_video_path) and os.path.exists(ir_json_path):
                process_video(ir_video_path, ir_json_path, output_ir_dir, f"{dataset_name}_IR")

            # 处理RGB视频
            rgb_video_path = os.path.join(dataset_path, 'RGB.mp4')
            rgb_json_path = os.path.join(dataset_path, 'RGB_label.json')
            if os.path.exists(rgb_video_path) and os.path.exists(rgb_json_path):
                process_video(rgb_video_path, rgb_json_path, output_rgb_dir, f"{dataset_name}_RGB")


if __name__ == "__main__":
    # 指定数据集根目录
    base_directory = "E:/datasets/test"  # 请修改为您的实际路径
    process_dataset(base_directory)
    print("所有数据集处理完成！")