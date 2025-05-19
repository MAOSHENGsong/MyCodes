import logging
import sys
from pathlib import Path

from torch import nn
from ultralytics.utils.ops import make_divisible

from yolo_models.backbone.common import Conv, C3, SPPF, C3k2

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[1]          # 获取项目根目录（上两级目录）
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将根目录添加到模块搜索路径

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

import torch.nn as nn


class YOLOv11Backbone(nn.Module):
    """YOLOv11主干网络实现
    基于yolo11.yaml配置文件构建，包含特征提取核心组件

    特性:
        - 多尺度特征提取(P3/8, P4/16, P5/32)
        - 动态深度/宽度缩放
        - 集成注意力机制模块
        - 优化SPPF金字塔池化结构

    参数说明:
        cfg (dict): 配置文件对象，需包含depth/gw系数及模块参数
        in_chans (int): 输入通道数，默认3(RGB)
    """

    def __init__(self, cfg, in_chans=3):
        super().__init__()
        # ----------------- 配置解析 -----------------
        self.gd = cfg.Model.depth_multiple  # 深度缩放系数
        self.gw = cfg.Model.width_multiple  # 宽度缩放系数

        # ----------------- 基础通道定义 -----------------
        base_channels = {
            'stage1': 64,  # 初始卷积输出
            'stage2': 128,  # P2/4阶段
            'stage3': 256,  # P3/8阶段
            'stage4': 512,  # P4/16阶段
            'stage5': 1024,  # P5/32阶段
            'spp': 1024,  # SPPF输出
            'c2psa': 1024  # 注意力模块输出
        }
        self._adjust_channels(base_channels)  # 动态调整通道数

        # ----------------- 网络结构定义 -----------------
        self.layers = nn.ModuleDict(self._build_layers(in_chans))

        # ----------------- 输出特征维度 -----------------
        self.out_channels = {
            'C3': base_channels['stage3'],
            'C4': base_channels['stage4'],
            'C5': base_channels['c2psa']
        }

    def _build_layers(self, in_chans):
        """根据yaml配置构建层序列"""
        layers = nn.ModuleDict()

        # ----------------- Stage 1 (P1/2) -----------------
        layers['stage1'] = Conv(
            in_chans, self.base_channels['stage1'],
            k=3, s=2, p=1, act='silu'
        )

        # ----------------- Stage 2 (P2/4) -----------------
        layers['stage2_conv'] = Conv(
            self.base_channels['stage1'], self.base_channels['stage2'],
            k=3, s=2, act='silu'
        )
        layers['stage2_c3k2'] = C3k2(
            self.base_channels['stage2'], self.base_channels['stage2'],
            n=self._get_depth(2), shortcut=False, e=0.25
        )

        # ----------------- Stage 3 (P3/8) -----------------
        layers['stage3_conv'] = Conv(
            self.base_channels['stage2'], self.base_channels['stage3'],
            k=3, s=2, act='silu'
        )
        layers['stage3_c3k2'] = C3k2(
            self.base_channels['stage3'], self.base_channels['stage3'],
            n=self._get_depth(2), shortcut=False, e=0.25
        )

        # ----------------- Stage 4 (P4/16) -----------------
        layers['stage4_conv'] = Conv(
            self.base_channels['stage3'], self.base_channels['stage4'],
            k=3, s=2, act='silu'
        )
        layers['stage4_c3k2'] = C3k2(
            self.base_channels['stage4'], self.base_channels['stage4'],
            n=self._get_depth(2), shortcut=True
        )

        # ----------------- Stage 5 (P5/32) -----------------
        layers['stage5_conv'] = Conv(
            self.base_channels['stage4'], self.base_channels['stage5'],
            k=3, s=2, act='silu'
        )
        layers['stage5_c3k2'] = C3k2(
            self.base_channels['stage5'], self.base_channels['stage5'],
            n=self._get_depth(2), shortcut=True
        )

        # ----------------- 特征增强模块 -----------------
        layers['sppf'] = SPPF(
            self.base_channels['stage5'], self.base_channels['spp'], k=5
        )
        layers['c2psa'] = C2PSA(
            self.base_channels['spp'], self.base_channels['c2psa'],
            n=self._get_depth(2)
        )

        return layers

    def forward(self, x):
        """前向传播逻辑"""
        # P1/2
        x = self.layers['stage1'](x)

        # P2/4
        x = self.layers['stage2_conv'](x)
        x = self.layers['stage2_c3k2'](x)

        # P3/8
        x = self.layers['stage3_conv'](x)
        c3 = self.layers['stage3_c3k2'](x)

        # P4/16
        x = self.layers['stage4_conv'](c3)
        c4 = self.layers['stage4_c3k2'](x)

        # P5/32
        x = self.layers['stage5_conv'](c4)
        x = self.layers['stage5_c3k2'](x)

        # 特征增强
        x = self.layers['sppf'](x)
        c5 = self.layers['c2psa'](x)

        return c3, c4, c5

    def _get_depth(self, n):
        """动态调整模块深度"""
        return max(round(n * self.gd), 1)

    def _adjust_channels(self, channels_dict):
        """动态调整通道数"""
        for k in channels_dict:
            channels_dict[k] = self._get_width(channels_dict[k])
        self.base_channels = channels_dict

    def _get_width(self, n):
        """8的整数倍通道调整"""
        return make_divisible(n * self.gw, 8)