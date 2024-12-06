from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('D:/code/UAV_Detection/VisualDetection/YOLOv8/runs/detect/best.pt') # 这里用你训练完以后保存的模型文件

# 推理单张图片
results = model('D:/code/UAV_Detection/VisualDetection/YOLOv8/runs/detect/20190925_101846_1_1_IR_0000.png')

# 显示推理结果
for result in results:
    result.show()
