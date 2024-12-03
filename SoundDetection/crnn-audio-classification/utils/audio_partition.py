import os
import random
import librosa
import soundfile as sf
from tqdm import tqdm

def split_audio(input_path, min_duration=2, max_duration=4):
    """
    将音频文件切割成多个不重叠的短音频片段

    参数:
    input_path: 输入文件夹路径
    min_duration: 最小音频长度(秒)
    max_duration: 最大音频长度(秒)
    """

    # 获取所有.wav文件
    wav_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    # 处理每个音频文件
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        try:
            # 读取音频文件
            audio, sr = librosa.load(wav_file, sr=None)

            # 获取音频总长度(秒)
            total_duration = librosa.get_duration(y=audio, sr=sr)

            if total_duration < min_duration:
                continue

            # 计算可以切割的段数
            current_position = 0
            file_count = 1

            while current_position + min_duration <= total_duration:
                # 随机确定当前片段的长度
                segment_duration = random.uniform(min_duration, min(max_duration, total_duration - current_position))

                # 计算采样点位置
                start_sample = int(current_position * sr)
                end_sample = int((current_position + segment_duration) * sr)

                # 提取音频片段
                segment = audio[start_sample:end_sample]

                # 生成输出文件名
                original_filename = os.path.splitext(os.path.basename(wav_file))[0]
                output_filename = f"{original_filename}_segment_{file_count}_{segment_duration:.2f}s.wav"
                output_filepath = os.path.join(os.path.dirname(wav_file), output_filename)

                # 保存音频片段
                sf.write(output_filepath, segment, sr)

                # 更新位置和计数器
                current_position += segment_duration
                file_count += 1

        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")

# 使用示例
if __name__ == "__main__":
    input_folder = "E:/datasets/UAS/Protocol1/Recording1"  # 替换为你的输入文件夹路径

    split_audio(
        input_path=input_folder,
        min_duration=2,
        max_duration=4
    )