#coding: utf-8
from ultralytics import YOLO
import matplotlib
matplotlib.use( "TkAgg")

if __name__ == '__main__':
    #加载训练好的模型
    model = YOLO('D:/code/UAV_Detection/VisualDetection/YOLOv8/script/train/runs/best.pt')
    # 对验证集进行评估
    results = model.val(data = 'D:/code/UAV_Detection/VisualDetection/YOLOv8/script/train/data.yaml', split='test')

# 打印主要评估指标
print(f"mAP50: {results.metrics['map50']:.4f}")  # 平均精度 (IoU=0.5)
print(f"mAP50-95: {results.metrics['map']:.4f}")  # 平均精度 (IoU=0.5:0.95)
print(f"Precision: {results.metrics['precision']:.4f}")  # 精确率
print(f"Recall: {results.metrics['recall']:.4f}")  # 召回率

# 如果你想打印所有指标
print("\n详细评估指标:")
for key, value in results.metrics.items():
    print(f"{key}: {value:.4f}")