from ultralytics import YOLO

if __name__ == '__main__':
    # 初始训练
    model = YOLO("yolov8.yaml").load("yolov8n.pt")# 加载预训练模型，如果本地没有会自动下载
    results = model.train(data="data.yaml", epochs=300, imgsz=640, batch=32, workers=16)
