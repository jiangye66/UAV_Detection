# 导入必要的库
import torch
from utils.util import load_image, load_audio  # 从工具模块加载加载图像和音频的函数
from PIL import ImageDraw, ImageFont  # 用于图像绘制和字体设置


class ImageInference:
    """
    用于图像推理的类
    """

    def __init__(self, model, transforms):
        # 初始化推理模型和转换操作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为CUDA或CPU
        self.model = model.to(self.device)  # 将模型移动到设备上
        self.model.eval()  # 设置模型为评估模式

        self.transforms = transforms  # 存储图像转换

    def infer(self, path):
        """
        对输入图像进行推理
        :param path: 图像路径
        :return: 标签和置信度
        """
        image = load_image(path)  # 加载图像

        # 应用图像转换
        image_t, _ = self.transforms.apply(image, None)
        # 执行模型预测
        label, conf = self.model.predict(image_t.to(self.device))

        return label, conf  # 返回预测的标签和置信度

    def draw(self, path, label, conf):
        """
        在图像上绘制预测结果
        :param path: 图像路径
        :param label: 预测标签
        :param conf: 置信度
        """
        image = load_image(path)  # 加载图像
        draw = ImageDraw.Draw(image)  # 创建绘图对象        
        font = ImageFont.truetype('utils/Verdana.ttf', 15)  # 加载字体
        # 绘制标签和置信度
        draw.text((0, 0), "%s (%.1f%%)" % (label, 100 * conf), (255, 0, 255), font)
        # 保存带有预测结果的图像
        image.save(path.split('.')[0] + '_pred.png')


import os
from net import MelspectrogramStretch  # 导入梅尔频谱图类
from utils import plot_heatmap  # 导入热图绘制函数


class AudioInference:
    """
    用于音频推理的类
    """

    def __init__(self, model, transforms):
        # 初始化推理模型和转换操作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为CUDA或CPU
        self.model = model.to(self.device)  # 将模型移动到设备上
        self.model.eval()  # 设置模型为评估模式
        self.transforms = transforms  # 存储音频转换

        self.mel = MelspectrogramStretch(norm='db')  # 初始化梅尔频谱图生成器
        self.mel.eval()  # 设置梅尔频谱图生成器为评估模式

    def infer(self, path):
        """
        对输入音频进行推理
        :param path: 音频文件路径
        :return: 标签和置信度
        """
        data = load_audio(path)  # 加载音频数据
        sig_t, sr, _ = self.transforms.apply(data, None)  # 应用音频转换

        length = torch.tensor(sig_t.size(0))  # 记录信号长度
        sr = torch.tensor(sr)  # 转换采样率为张量
        data = [d.unsqueeze(0).to(self.device) for d in [sig_t, length, sr]]  # 将输入数据移动到设备上

        # 执行模型预测
        # label, conf = self.model.predict(data)
        label, conf = self.model.predict(data,"dog_bark")   #输入模型预测的标签
        return label, conf  # 返回预测的标签和置信度

    def draw(self, path, label, conf):
        """
        绘制音频的梅尔频谱图并保存
        :param path: 音频文件路径
        :param label: 预测标签
        :param conf: 置信度
        """
        sig, sr = load_audio(path)  # 加载音频
        sig = torch.tensor(sig).mean(dim=1).view(1, 1, -1).float()  # 处理音频信号为合适尺寸
        spec = self.mel(sig)[0]  # 生成梅尔频谱图

        # 创建输出路径的文件夹
        output_dir = 'result'  # 输出文件夹为当前目录下的 result
        os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

        out_path = os.path.join(output_dir, os.path.basename(path).split('.')[0] + '_pred.png')  # 设置输出路径
        pred_txt = "%s (%.1f%%)" % (label, 100 * conf)  # 生成预测文本
        plot_heatmap(spec.cpu().numpy(), out_path, pred=pred_txt)  # 绘制热图并保存
        print("Result directory:", out_path)

