import json

# 读取 JSON 文件
with open('E:/datasets/anti-uav/test-dev/20190925_101846_1_1/RGB_label.json', 'r') as json_file:
    data = json.load(json_file)

# 提取数据
exist_data = data["exist"]
gt_rect_data = data["gt_rect"]

# 确保 gt_rect 数据长度与 exist 数据长度一致
assert len(exist_data) == len(gt_rect_data), "数据长度不一致！"

# 转换为指定格式并写入 TXT 文件
with open('E:/datasets/anti-uav/test-dev/20190925_101846_1_1/RGB_label.txt', 'w') as txt_file:
    for i in range(len(exist_data)):
        exist_value = exist_data[i]
        rect_values = gt_rect_data[i] if gt_rect_data[i] else [0, 0, 0, 0]
        line = f"{exist_value} {rect_values[0]} {rect_values[1]} {rect_values[2]} {rect_values[3]}\n"
        txt_file.write(line)

print("转换完成，结果已写入 output.txt 文件。")
