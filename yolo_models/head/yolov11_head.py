import torch
import torch.nn as nn
import math

from ultralytics.nn.modules import DFL
from utils.general import check_version

class Detect(nn.Module):
    """YOLO Detect head modified with config-based initialization and keypoint support."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, cfg):
        super(Detect, self).__init__()
        # 从配置动态获取参数
        self.nc = cfg.Dataset.nc  # 类别数
        self.num_keypoints = getattr(cfg.Dataset, 'np', 0)  # 关键点数（兼容无关键点情况）
        self.cur_imgsize = [cfg.Dataset.img_size, cfg.Dataset.img_size]
        anchors = cfg.Model.anchors
        ch = [int(out_c * cfg.Model.width_multiple) for out_c in cfg.Model.Neck.out_channels]

        # 核心参数计算
        self.nl = len(anchors)  # 检测层数
        self.na = len(anchors[0]) // 2  # 每层锚框数
        self.no = self.nc + self.num_keypoints * 3 + 5  # 输出维度（xywh+conf+cls+kpts）
        self.reg_max = 16  # 保留DFL参数

        # 注册锚框缓冲区
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl

        # 输出卷积（兼容DFL）
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 初始化参数
        self.stride = cfg.Model.Head.strides
        self._initialize_biases()

    def _initialize_biases(self):
        """兼容YOLOv5的偏置初始化策略"""
        for mi, s in zip(self.m, self.stride):
            b = mi.bias.view(self.na, -1)
            # 目标置信度偏置
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 基于640px图像假设
            # 分类偏置（保留关键点位置）
            cls_start = 5 + self.num_keypoints * 3
            b.data[:, cls_start:] += math.log(0.6 / (self.nc - 0.99))

    def forward(self, x):
        z = []  # 推理输出缓存
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape

            # 导出模式处理
            if self.export:
                x[i] = x[i].view(bs, self.na, self.no, -1).permute(0, 1, 3, 2)
                z.append(x[i])
                continue

            # 常规维度调整
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 推理时解码
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                # 坐标解码（保留动态锚点特性）
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        """动态网格生成（兼容不同特征图尺寸）"""
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand \
            ((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def post_process_v2(self, x):
        """优化版后处理（兼容关键点）"""
        z = []
        for i in range(self.nl):
            ny, nx = int(self.cur_imgsize[0 ] /self.stride[i]), int(self.cur_imgsize[1 ] /self.stride[i])
            bs, _, _, _ = x[i].shape
            x[i] = x[i].view(bs, self.na, ny, nx, self.no).contiguous()
            y = x[i].sigmoid()

            # 分离各部分预测
            xy = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
            kpts = y[..., 5: 5 +self.num_keypoints *3]  # (x,y,vis)格式
            conf = y[..., 4:5]
            cls = y[..., 5+ self.num_keypoints * 3:]

            # 拼接最终结果
            y = torch.cat((xy, wh, conf, cls, kpts), -1)
            z.append(y.view(bs, -1, self.no))
        return (torch.cat(z, 1), x)
