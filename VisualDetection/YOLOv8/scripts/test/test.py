from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('D:/detection/UAV_Detection/VisualDetection/YOLOv8/scripts/train/runs/detect/train10/weights/best.pt')  # 这里用你训练完以后保存的模型文件

# 推理单张图片
results = model('D:/detection/datasets/yolo_dataset/test/images/00001.jpg')

# 显示推理结果并输出检测结果
for result in results:
    # 输出检测的边界框和置信度
    for box in result.boxes:
        cls = box.cls.item()          # 转换为标量
        conf = box.conf.item()        # 转换为标量
        xyxy = box.xyxy.cpu().numpy()  # 转换为 NumPy 数组
        print(f'分类: {cls}, 置信度: {conf:.2f}, 边界框: {xyxy}')  # 打印检测结果中的类别、置信度和边界框
    result.show()  # 显示检测结果