import numpy as np
import torch
import torch.nn as nn

from torchaudio.transforms import TimeStretch, AmplitudeToDB
from torch.distributions import Uniform

from torchaudio.transforms import Spectrogram, MelSpectrogram


def _num_stft_bins(lengths, fft_length, hop_length, pad):
    # 计算每个音频的STFT频谱的长度
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length


class MelspectrogramStretch(MelSpectrogram):
    """
    扩展的梅尔频谱图类，增加了时间拉伸的数据增强功能
    """

    def __init__(self, hop_length=None,
                 sample_rate=44100,
                 num_mels=128,
                 fft_length=2048,
                 norm='whiten',
                 stretch_param=[0.4, 0.4]):
        """
        初始化梅尔频谱图拉伸的参数
        :param hop_length: 每帧之间的跳跃长度
        :param sample_rate: 音频采样率
        :param num_mels: 梅尔滤波器的数量
        :param fft_length: FFT（快速傅里叶变换）长度
        :param norm: 归一化类型（'whiten' 或 'db'）
        :param stretch_param: 拉伸参数，stretch_param[0]为拉伸的概率，stretch_param[1]为拉伸幅度
        """
        super(MelspectrogramStretch, self).__init__(sample_rate=sample_rate,
                                                    n_fft=fft_length,
                                                    hop_length=hop_length,
                                                    n_mels=num_mels)

        # STFT 变换
        self.stft = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                hop_length=self.hop_length, pad=self.pad,
                                power=None, normalized=False)

        # 数据增强：时间拉伸的概率和幅度
        self.prob = stretch_param[0]
        self.random_stretch = RandomTimeStretch(stretch_param[1],
                                                self.hop_length,
                                                self.n_fft // 2 + 1,
                                                fixed_rate=None)

        # 替代复数归一化操作
        self.complex_norm = lambda x: torch.abs(x).pow(2.0)
        # 归一化方法
        self.norm = SpecNormalization(norm)

    def forward(self, x, lengths=None):
        """
        前向传播，计算梅尔频谱图，并根据概率进行时间拉伸
        :param x: 输入的音频信号
        :param lengths: 音频的长度（用于变长序列处理）
        :return: 归一化后的梅尔频谱图和对应的长度
        """
        # 计算STFT频谱
        x = self.stft(x)

        if lengths is not None:
            # 计算STFT频谱的长度
            lengths = _num_stft_bins(lengths, self.n_fft, self.hop_length, self.n_fft // 2)
            lengths = lengths.long()

        # 在训练时，以一定的概率对频谱进行时间拉伸（数据增强）
        if torch.rand(1)[0] <= self.prob and self.training:
            # 使用Phase Vocoder进行时间拉伸
            x, rate = self.random_stretch(x)
            # 修改长度（根据拉伸因子进行调整）
            lengths = (lengths.float() / rate).long() + 1

        # 计算复数频谱的模长
        x = self.complex_norm(x)
        # 计算梅尔频谱
        x = self.mel_scale(x)

        # 归一化梅尔频谱
        x = self.norm(x)

        if lengths is not None:
            return x, lengths
        return x

    def __repr__(self):
        # 返回类的字符串表示
        return self.__class__.__name__ + '()'


class RandomTimeStretch(TimeStretch):
    """
    用于对音频信号进行随机时间拉伸的类
    """

    def __init__(self, max_perc, hop_length=None, n_freq=201, fixed_rate=None):
        """
        初始化随机时间拉伸的参数
        :param max_perc: 最大的拉伸百分比
        :param hop_length: 每帧之间的跳跃长度
        :param n_freq: 频率维度的大小
        :param fixed_rate: 固定的拉伸因子（如果有的话）
        """
        super(RandomTimeStretch, self).__init__(hop_length, n_freq, fixed_rate)
        # 使用均匀分布生成随机拉伸因子
        self._dist = Uniform(1. - max_perc, 1 + max_perc)

    def forward(self, x):
        """
        对输入信号进行时间拉伸
        :param x: 输入的频谱信号
        :return: 拉伸后的频谱和拉伸因子
        """
        # 从均匀分布中采样拉伸因子
        rate = self._dist.sample().item()
        # 调用父类的forward方法进行时间拉伸
        return super(RandomTimeStretch, self).forward(x, rate), rate


class SpecNormalization(nn.Module):
    """
    频谱归一化类，根据不同的归一化类型进行处理
    """

    def __init__(self, norm_type, top_db=80.0):
        """
        初始化归一化操作
        :param norm_type: 归一化类型（'db' 或 'whiten'）
        :param top_db: 用于db归一化的阈值
        """
        super(SpecNormalization, self).__init__()

        if 'db' == norm_type:
            # 使用分贝归一化
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            # 使用零均值标准化
            self._norm = lambda x: self.z_transform(x)
        else:
            # 默认情况下不进行归一化
            self._norm = lambda x: x

    def z_transform(self, x):
        """
        对频谱进行零均值标准化
        :param x: 输入的频谱
        :return: 标准化后的频谱
        """
        # 计算每个批次的均值和标准差
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        # 标准化处理
        x = (x - mean) / std
        return x

    def forward(self, x):
        """
        执行归一化操作
        :param x: 输入的频谱
        :return: 归一化后的频谱
        """
        return self._norm(x)
