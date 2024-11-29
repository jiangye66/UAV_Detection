import cv2
import json
import os

# 文件路径
video_path = 'E:/datasets/anti-uav/test-dev/20190925_101846_1_1/RGB.mp4'
json_path = 'E:/datasets/anti-uav/test-dev/20190925_101846_1_1/RGB_label.json'

# 提取视频所在目录和文件名
video_dir = os.path.dirname(os.path.abspath(video_path))  # 视频所在目录
video_name = os.path.splitext(os.path.basename(video_path))[0]  # 视频文件名（无后缀）

# 输出文件夹路径
output_dir = os.path.join(video_dir, video_name)  # 输出文件夹以视频名命名

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 读取标注文件
with open(json_path, 'r') as json_file:
    data = json.load(json_file)

exist_data = data["exist"]
gt_rect_data = data["gt_rect"]
frame_count = len(exist_data)

# 打开视频文件
video_capture = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not video_capture.isOpened():
    print("无法打开视频文件！")
    exit()

# 获取视频的宽度和高度
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频帧大小: 宽度={frame_width}, 高度={frame_height}")

# 按帧读取视频
for frame_idx in range(frame_count):
    # 读取一帧
    success, frame = video_capture.read()
    if not success:
        print(f"读取第 {frame_idx} 帧失败！")
        break

    # 保存帧图片
    frame_filename = os.path.join(output_dir, f"{video_name}_{frame_idx + 1:04d}.png")
    cv2.imwrite(frame_filename, frame)

    # 写入对应的 TXT 文件
    exist_value = exist_data[frame_idx]
    rect_values = gt_rect_data[frame_idx] if gt_rect_data[frame_idx] else [0, 0, 0, 0]

    # 提取 x_min, y_min, width, height
    x_min, y_min, bbox_width, bbox_height = rect_values

    # 转换为比值
    center_x_ratio = (x_min + bbox_width / 2) / frame_width
    center_y_ratio = (y_min + bbox_height / 2) / frame_height
    width_ratio = bbox_width / frame_width
    height_ratio = bbox_height / frame_height

    # 生成 TXT 文件路径
    txt_filename = os.path.join(output_dir, f"{video_name}_{frame_idx + 1:04d}.txt")

    # 写入新的标注
    with open(txt_filename, 'w') as txt_file:
        txt_line = f"{exist_value} {center_x_ratio:.6f} {center_y_ratio:.6f} {width_ratio:.6f} {height_ratio:.6f}\n"
        txt_file.write(txt_line)

    if frame_idx % 50 == 0:
        print(f"已处理 {frame_idx + 1} 帧...")

# 释放视频捕获对象
video_capture.release()
print(f"处理完成，所有图片和对应的 TXT 文件保存在 '{output_dir}' 文件夹中。")
