import os
import unittest
from ultralytics import YOLO

class TestYOLOModel(unittest.TestCase):

    def setUp(self):
        # Step 1: 加载训练好的模型
        self.model = YOLO('D:/detection/UAV_Detection/VisualDetection/YOLOv8/scripts/train/runs/detect/train7/weights/best.pt')
        # Step 2: 配置目标数据集的路径
        self.data_yaml_path = 'D:/detection/UAV_Detection/VisualDetection/YOLOv8/scripts/test/data.yaml'


    def test_model_evaluation(self):
        # Step 4: 评估数据集
        results = self.model.val(
            data=self.data_yaml_path,
            conf=0.25,

            plots=True,  # 生成评估图表
            save=True,   # 保存预测结果
            save_txt=True,  # 保存预测框的坐标
            save_conf=True, # 保存置信度
            verbose=True    # 显示详细信息
        )
        # 数据保存在“C:\Users\Administrator\anaconda3\envs\detection\Lib\site-packages\tests\tmp\runs\detect\val”

        # Step 5: 输出评估结果
        print("\n评估结果：")

        # 使用 results.box.mean_results 获取平均指标
        mean_results = results.box.mean_results()
        print(f"Precision: {mean_results[0]:.4f}")
        print(f"Recall: {mean_results[1]:.4f}")
        print(f"mAP@0.5: {mean_results[2]:.4f}")
        print(f"mAP@0.5:0.95: {mean_results[3]:.4f}")

        # 推理时间
        print(f"\n推理时间 (每张图片): {results.speed['inference']:.2f} ms")



if __name__ == '__main__':
    unittest.main(verbosity=2)