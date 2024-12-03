import os
import random
import librosa
import pandas as pd
from tqdm import tqdm


def create_labels(input_path, output_csv, max_duration=10):
    """
    为10秒以下的音频文件创建标签文件

    参数:
    input_path: 输入文件夹路径
    output_csv: 输出CSV文件路径
    max_duration: 最大音频长度(秒)
    """

    # 存储所有数据的列表
    data = []

    # 获取所有.wav文件
    wav_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    # 计数器
    processed_count = 0
    skipped_count = 0

    # 处理每个音频文件
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        try:
            # 读取音频文件获取持续时间
            duration = librosa.get_duration(filename=wav_file)

            # 只处理10秒以下的文件
            if duration <= max_duration:
                # 获取文件名
                file_name = os.path.basename(wav_file)

                # 创建数据条目
                data_entry = {
                    'slice_file_name': file_name,
                    'start': 0,
                    'end': duration,
                    'fold': random.randint(1, 10),  # 随机分配1-10的fold
                    'classID': 1,
                    'class': 'drone'
                }

                data.append(data_entry)
                processed_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")

    # 创建DataFrame并保存为CSV
    if data:  # 确保有数据再创建DataFrame
        df = pd.DataFrame(data)

        # 确保列的顺序与示例一致
        df = df[['slice_file_name', 'start', 'end', 'fold', 'classID', 'class']]

        # 保存到CSV文件
        df.to_csv(output_csv, index=False)

        print(f"\n处理完成:")
        print(f"- 已处理 {processed_count} 个文件")
        print(f"- 跳过 {skipped_count} 个超过{max_duration}秒的文件")
        print(f"- 标签文件已保存到: {output_csv}")
    else:
        print(f"\n未找到符合条件的音频文件（低于{max_duration}秒）")


# 使用示例
if __name__ == "__main__":
    input_folder = "E:/datasets/UAS"  # 替换为你的输入文件夹路径
    output_csv = "E:/datasets/UAS/labels.csv"  # 输出的CSV文件名

    create_labels(
        input_path=input_folder,
        output_csv=output_csv,
        max_duration=10  # 设置最大时长为10秒
    )
