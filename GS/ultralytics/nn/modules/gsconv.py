import torch
import torch.nn as nn
from .conv import autopad  # 使用原始的 autopad 保持一致

class GSConv(nn.Module):
    """
    GSConv: Grouped Spatial Convolution
    替代标准 Conv，用于轻量化结构设计，兼容 YOLOv8 的调用方式。
    结构: Group Conv -> Pointwise Conv -> BN -> Activation
    """

    default_act = nn.SiLU()  # 与原 Conv 保持一致

    def __init__(self, c1, c2, k=1, s=1, p=None, g=2, d=1, act=True):
        """
        初始化 GSConv。

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int): padding（自动计算）
            g (int): 分组数（默认为2）
            d (int): 膨胀率
            act (bool|Module): 激活函数
        """
        g = 1 if c1 % g != 0 else g  # 如果 in_channels 不能整除，就退回 group=1
        self.group_conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.point_conv = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.group_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        return self.act(x)

    def forward_fuse(self, x):
        return self.act(self.point_conv(self.group_conv(x)))
