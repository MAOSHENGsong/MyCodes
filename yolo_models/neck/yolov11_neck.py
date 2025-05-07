import torch.nn as nn
from ultralytics.utils.ops import make_divisible

from ..backbone.common import Conv, Concat, C3k2, C2PSA

class YOLO11Neck(nn.Module):
    """
    YOLO11改进型颈部网络，集成C3k2和注意力模块
    结构特性：
    - 基于C3k2模块构建跨阶段特征融合
    - 引入C2PSA注意力机制增强特征表达能力
    - 自适应深度/宽度缩放系数
    """

    def __init__(self, cfg):
        super(YOLO11Neck, self).__init__()
        self.gd = cfg.Model.depth_multiple  # 深度缩放系数
        self.gw = cfg.Model.width_multiple  # 宽度缩放系数

        # 通道数配置
        input_p3, input_p4, input_p5 = cfg.Model.Neck.in_channels
        output_p3, output_p4, output_p5 = cfg.Model.Neck.out_channels

        # 宽度自适应调整
        self.input_p3 = make_divisible(input_p3 * self.gw, 8)
        self.input_p4 = make_divisible(input_p4 * self.gw, 8)
        self.input_p5 = make_divisible(input_p5 * self.gw, 8)
        self.output_p3 = make_divisible(output_p3 * self.gw, 8)
        self.output_p4 = make_divisible(output_p4 * self.gw, 8)
        self.output_p5 = make_divisible(output_p5 * self.gw, 8)

        # 激活函数配置
        act = 'silu' if cfg.Model.Neck.activation == 'SiLU' else 'hard_swish'

        # 上采样路径
        self.conv_p5 = Conv(self.input_p5, self.input_p5 // 2, 1, act=act)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3k2_1 = C3k2(self.input_p5 // 2 + self.input_p4, self.input_p4,
                           n=self.get_depth(3), e=0.5, act=act)

        self.conv_p4 = Conv(self.input_p4, self.input_p3, 1, act=act)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2psa_1 = C2PSA(self.input_p3 * 2, self.output_p3,
                             n=self.get_depth(2), act=act)

        # 下采样路径
        self.conv_down1 = Conv(self.output_p3, self.output_p3, 3, 2, act=act)
        self.c3k2_2 = C3k2(self.output_p3 + self.input_p3, self.output_p4,
                           n=self.get_depth(3), e=0.5, act=act)

        self.conv_down2 = Conv(self.output_p4, self.output_p4, 3, 2, act=act)
        self.c3k2_3 = C3k2(self.output_p4 + self.input_p5 // 2, self.output_p5,
                           n=self.get_depth(3), e=0.5, act=act)

        self.concat = Concat()

    def get_depth(self, n):
        """深度动态缩放"""
        return max(round(n * self.gd), 1)

    def forward(self, inputs):
        # 输入特征分解
        p3, p4, p5 = inputs

        # 上采样分支
        x = self.conv_p5(p5)
        x = self.upsample1(x)
        x = self.concat([x, p4])
        x = self.c3k2_1(x)

        x = self.conv_p4(x)
        x = self.upsample2(x)
        x = self.concat([x, p3])
        p3_out = self.c2psa_1(x)  # 注意力增强输出

        # 下采样分支
        x = self.conv_down1(p3_out)
        x = self.concat([x, self.conv_p4.output])
        p4_out = self.c3k2_2(x)

        x = self.conv_down2(p4_out)
        x = self.concat([x, self.conv_p5.output])
        p5_out = self.c3k2_3(x)

        return p3_out, p4_out, p5_out