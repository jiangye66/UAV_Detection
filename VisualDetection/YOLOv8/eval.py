#coding: utf-8
from ultralytics import YOLO
import matplotlib
matplotlib.use( "TkAgg")

if __name__ == '__main__':
    #加载训练好的模型
    model = YOLO('D:/detection/UAV_Detection/VisualDetection/YOLOv8/runs/detect/train2/weights/best.pt')
    # 对验证集进行评估
    metrics = model.val(data = 'D:/detection/UAV_Detection/VisualDetection/YOLOv8/data.yaml')
