import os

def find_empty_txt_files(directory):
    empty_files = []
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            # 检查文件大小
            if os.path.getsize(file_path) == 0:
                empty_files.append(file_path)
                # 插入 YOLO 标签内容
                # with open(file_path, 'a') as file:
                #     file.write("1 0.5 0.5 0.5 0.5")  # 插入标签内
    return empty_files

# 替换为你的文件夹路径
directory_path = "D:/detection/datasets/yolo_dataset/train/labels"
empty_text_files = find_empty_txt_files(directory_path)

# 输出空文件列表
if empty_text_files:
    print("找到以下空的 TXT 文件并已插入内容:")
    for empty_file in empty_text_files:
        print(empty_file)
else:
    print("没有找到空的 TXT 文件。")
