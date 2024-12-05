import json
import os
from PIL import Image
import shutil


def process_annotations(json_file_path, dataset_type, output_base):
    """处理单个json文件及其对应的图像文件夹"""

    # 读取JSON文件
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # 提取数据
    exist_data = data["exist"]
    gt_rect_data = data["gt_rect"]

    # 确保数据长度一致
    assert len(exist_data) == len(gt_rect_data), "数据长度不一致！"

    # 获取JSON文件的目录和同名文件夹
    json_dir = os.path.dirname(json_file_path)
    image_folder_name = os.path.splitext(os.path.basename(json_file_path))[0]
    image_folder_path = os.path.join(json_dir, image_folder_name)

    # 创建输出目录结构
    images_folder = os.path.join(output_base, dataset_type, "images")
    labels_folder = os.path.join(output_base, dataset_type, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # 获取图像文件列表
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 获取序列名称（父文件夹名称）
    sequence_name = os.path.basename(os.path.dirname(json_file_path))
    modality = image_folder_name  # infrared 或 visible

    # 处理每隔 5 张图像
    for i, image_filename in enumerate(sorted(image_files)):
        if i % 5 != 0:  # 只处理每隔 5 张图像
            continue

        if i >= len(exist_data):
            break

        # 构造图像完整路径
        image_path = os.path.join(image_folder_path, image_filename)

        exist_value = exist_data[i]
        gt_rect = gt_rect_data[i]

        # 读取图像并复制
        new_image_name = f"{sequence_name}_{modality}_{image_filename}"
        new_image_path = os.path.join(images_folder, new_image_name)
        shutil.copy2(image_path, new_image_path)

        # 处理标签文件
        label_filename = os.path.splitext(new_image_name)[0] + '.txt'
        label_filepath = os.path.join(labels_folder, label_filename)

        if exist_value == 1 and gt_rect:  # 无人机存在，保存目标框
            try:
                with Image.open(image_path) as img:
                    image_width, image_height = img.size

                xmin, ymin, width, height = gt_rect
                x_center = xmin + width / 2
                y_center = ymin + height / 2

                # 归一化坐标
                x_center /= image_width
                y_center /= image_height
                width /= image_width
                height /= image_height

                with open(label_filepath, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
            except Exception as e:
                print(f"处理图像 {image_filename} 时发生错误: {e}")

        else:  # 无人机不存在，创建空标签文件
            open(label_filepath, 'w').close()


def process_dataset():
    """处理整个数据集"""
    # 数据集根目录
    dataset_root = "E:/datasets/Anti-UAV-RGBT"
    # 输出目录
    output_base = "E:/datasets/Anti-UAV-RGBT/yolo_dataset"

    # 创建主输出目录
    os.makedirs(output_base, exist_ok=True)
    print(f"Created output directory: {output_base}")

    # 处理每个数据集分割（train, val, test）
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_root, split)
        if not os.path.exists(split_path):
            continue

        print(f"Processing {split} split...")

        # 处理每个序列文件夹
        for sequence in os.listdir(split_path):
            sequence_path = os.path.join(split_path, sequence)
            if not os.path.isdir(sequence_path):
                continue

            # 处理红外和可见光数据
            for modality in ['infrared', 'visible']:
                json_file = os.path.join(sequence_path, f"{modality}.json")
                if os.path.exists(json_file):
                    print(f"Processing {split}/{sequence}/{modality}")
                    process_annotations(json_file, split, output_base)


if __name__ == "__main__":
    process_dataset()
