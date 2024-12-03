import pandas as pd
import os
import shutil


def copy_wav_files(csv_path, source_dir, target_dir):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 创建目标文件夹中fold1-fold10的子文件夹
    for i in range(1, 11):
        os.makedirs(os.path.join(target_dir, f'fold{i}'), exist_ok=True)

    # 遍历CSV中的每一行
    for _, row in df.iterrows():
        file_name = row['slice_file_name']
        fold = row['fold']

        # 找到源目录中的对应文件
        for root, _, files in os.walk(source_dir):
            if file_name in files:
                source_path = os.path.join(root, file_name)
                target_path = os.path.join(target_dir, f'fold{fold}', file_name)

                # 复制文件到对应的fold文件夹
                shutil.copy2(source_path, target_path)
                print(f'Copied {file_name} to fold{fold}')
                break


# 参数设置
csv_path = "E:/datasets/UAS/labels.csv"  # CSV 文件路径
source_dir = "E:/datasets/UAS"  # 音频文件所在的源文件夹路径
target_dir = "E:/datasets/UrbanSound8K/audio"  # 目标文件夹路径

# 调用函数
copy_wav_files(csv_path, source_dir, target_dir)
