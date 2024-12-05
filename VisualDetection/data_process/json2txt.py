import json

# 读取 JSON 文件
with open('E:/datasets/Anti-UAV-RGBT/sample/train/20190925_101846_1_1/infrared.json', 'r') as json_file:
    data = json.load(json_file)

# 提取数据
exist_data = data["exist"]
gt_rect_data = data["gt_rect"]

# 确保 gt_rect 数据长度与 exist 数据长度一致
assert len(exist_data) == len(gt_rect_data), "数据长度不一致！"

# 转换为指定格式并写入 TXT 文件
import json

# 读取原始 JSON 文件
with open('E:/datasets/Anti-UAV-RGBT/sample/train/20190925_101846_1_1/infrared.json', 'r') as json_file:
    data = json.load(json_file)

# 提取数据
exist_data = data["exist"]
gt_rect_data = data["gt_rect"]

# 确保 gt_rect 数据长度与 exist 数据长度一致
assert len(exist_data) == len(gt_rect_data), "数据长度不一致！"

# 获取图像的宽度和高度（假设为某个值）
image_width = 640  # 根据实际情况修改
image_height = 480  # 根据实际情况修改

# 转换为 YOLO 格式并写入 TXT 文件
with open('E:/datasets/Anti-UAV-RGBT/sample/train/20190925_101846_1_1/infrared_yolo.txt', 'w') as txt_file:
    for i in range(len(exist_data)):
        exist_value = exist_data[i]  # 类别索引
        if gt_rect_data[i]:
            xmin, ymin, xmax, ymax = gt_rect_data[i]
            # 计算 YOLO 格式需要的值
            box_width = xmax - xmin
            box_height = ymax - ymin
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # 将坐标归一化到[0, 1]
            x_center /= image_width
            y_center /= image_height
            box_width /= image_width
            box_height /= image_height

            line = f"{exist_value} {x_center} {y_center} {box_width} {box_height}\n"
            txt_file.write(line)

print("转换完成，结果已写入 infrared_yolo.txt 文件。")
