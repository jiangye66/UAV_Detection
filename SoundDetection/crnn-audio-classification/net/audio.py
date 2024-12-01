import numpy as np
import torch
import torch.nn as nn

from torchaudio.transforms import Spectrogram, MelSpectrogram
from torchaudio.transforms import TimeStretch, AmplitudeToDB
from torch.distributions import Uniform


def _num_stft_bins(lengths, fft_length, hop_length, pad):
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length


class ComplexNorm(nn.Module):
    """Calculate the norm of a complex tensor.

    Args:
        power (float, optional): Power of the norm. (Default: 1.0)
    """

    def __init__(self, power=1.0):
        super().__init__()
        self.power = power

    def forward(self, complex_tensor):
        """
        Args:
            complex_tensor (Tensor): Complex tensor
        Returns:
            Tensor: norm of the complex tensor
        """
        # 处理不同的复数表示格式
        if torch.is_complex(complex_tensor):
            # 如果输入是复数类型
            norm = torch.abs(complex_tensor)
        else:
            # 如果输入是实数tensor，尝试将其作为实部和虚部分离的格式处理
            if complex_tensor.size(-1) == 2:
                real = complex_tensor[..., 0]
                imag = complex_tensor[..., 1]
                norm = torch.sqrt(real ** 2 + imag ** 2 + 1e-9)
            else:
                # 如果是实数tensor，直接取绝对值
                norm = torch.abs(complex_tensor)

        # 应用power
        if self.power != 1.0:
            norm = norm ** self.power

        return norm

    def __repr__(self):
        return self.__class__.__name__ + f'(power={self.power})'


class MelspectrogramStretch(MelSpectrogram):

    def __init__(self, hop_length=None,
                 sample_rate=44100,
                 num_mels=128,
                 fft_length=2048,
                 norm='whiten',
                 stretch_param=[0.4, 0.4]):

        super(MelspectrogramStretch, self).__init__(sample_rate=sample_rate,
                                                    n_fft=fft_length,
                                                    hop_length=hop_length,
                                                    n_mels=num_mels)

        self.stft = Spectrogram(n_fft=self.n_fft,
                                win_length=self.win_length,
                                hop_length=self.hop_length,
                                pad=self.pad,
                                power=None,  # 设置为None以获取复数输出
                                normalized=False,
                                return_complex=True)  # 返回复数张量

        # Augmentation
        self.prob = stretch_param[0]
        self.random_stretch = RandomTimeStretch(stretch_param[1],
                                                self.hop_length,
                                                self.n_fft // 2 + 1,
                                                fixed_rate=None)

        # Normalization (pot spec processing)
        self.complex_norm = ComplexNorm(power=2.)
        self.norm = SpecNormalization(norm)

    def forward(self, x, lengths=None):
        x = self.stft(x)

        if lengths is not None:
            lengths = _num_stft_bins(lengths, self.n_fft, self.hop_length, self.n_fft // 2)
            lengths = lengths.long()

        if torch.rand(1)[0] <= self.prob and self.training:
            x, rate = self.random_stretch(x)
            lengths = (lengths.float() / rate).long() + 1

        x = self.complex_norm(x)
        x = self.mel_scale(x)
        x = self.norm(x)

        if lengths is not None:
            # 确保 lengths 在 CPU 上
            lengths = lengths.cpu()
            return x, lengths
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTimeStretch(TimeStretch):

    def __init__(self, max_perc, hop_length=None, n_freq=201, fixed_rate=None):
        super(RandomTimeStretch, self).__init__(hop_length, n_freq, fixed_rate)
        self._dist = Uniform(1. - max_perc, 1 + max_perc)

    def forward(self, x):
        rate = self._dist.sample().item()
        return super(RandomTimeStretch, self).forward(x, rate), rate


class SpecNormalization(nn.Module):

    def __init__(self, norm_type, top_db=80.0):

        super(SpecNormalization, self).__init__()

        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x

    def z_transform(self, x):
        # Independent mean, std per batch
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x):
        return self._norm(x)
