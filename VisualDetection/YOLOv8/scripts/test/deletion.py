import os
import shutil


def process_yolo_dataset(images_dir, labels_dir):
    # 获取images和labels目录下的所有文件
    images = sorted(os.listdir(images_dir))
    labels = sorted(os.listdir(labels_dir))

    # 确保images和labels数量相同
    if len(images) != len(labels):
        print("警告：images和labels的文件数量不一致！")
        return

    # 每隔10个文件保留一个
    for i in range(len(images)):
        if i % 10 != 0:
            # 删除images中的文件
            os.remove(os.path.join(images_dir, images[i]))
            # 删除labels中的文件
            os.remove(os.path.join(labels_dir, labels[i]))

    print("处理完成！保留了每隔10个的文件。")


# 调用函数，替换路径为你的实际路径
images_directory = "D:/detection/datasets/yolo_dataset/val/images"
labels_directory = "D:/detection/datasets/yolo_dataset/val/labels"
process_yolo_dataset(images_directory, labels_directory)
