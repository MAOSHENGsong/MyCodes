import copy
import torch
import torch.nn as nn
import math

from utils.general import dist2bbox, make_anchors
from yolo_models.backbone import Conv
from yolo_models.backbone.common import DWConv, DFL

class Detect(nn.Module):
    """YOLO 检测头，用于检测模型。"""

    # 类属性定义
    dynamic = False  # 强制重建网格
    export = False  # 导出模式
    format = None  # 导出格式
    end2end = False  # 端到端模式
    max_det = 300  # 最大检测数
    shape = None  # 输入形状
    anchors = torch.empty(0)  # 初始化锚点
    strides = torch.empty(0)  # 步长
    legacy = False  # 兼容v3/v5/v8/v9模型的标志

    def __init__(self, nc=80, ch=()):
        """使用指定类别数和通道数初始化YOLO检测层。"""
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16  # DFL通道数
        self.no = nc + self.reg_max * 4  # 每个锚点的输出数
        self.stride = torch.zeros(self.nl)  # 构建时计算的步长

        # 通道数计算
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        # 构建回归分支
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )

        # 构建分类分支
        self.cv3 = (
            nn.ModuleList(nn.Sequential(DWConv(x, c3, 3), DWConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        )

        # 分布焦点损失模块
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 端到端模式特殊处理
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """拼接并返回预测边界框和类别概率。"""
        if self.end2end:
            return self.forward_end2end(x)

        # 处理每个检测层
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:  # 训练路径
            return x
        y = self._inference(x)  # 推理路径
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """端到端前向传播。"""
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1)
            for i in range(self.nl)
        ]

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:  # 训练路径
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """基于多级特征图解码预测边界框和类别概率。"""
        # 特征图拼接
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        # 动态生成锚点
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = make_anchors(x, self.stride, 0.5)
            self.shape = shape

        # 分割框和类别预测
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # 不同导出格式的特殊处理
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """初始化检测头偏置。"""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # 框回归偏置初始化
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s)  **  2)  # 分类偏置初始化

            if self.end2end:
                for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                    a[-1].bias.data[:] = 1.0
                    b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s)  **  2)

                def decode_bboxes(self, bboxes, anchors, xywh=True):
                    """解码边界框坐标。"""
                    return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

                @staticmethod
                def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
                    """后处理预测结果。"""
                    batch_size, anchors, _ = preds.shape
                    boxes, scores = preds.split([4, nc], dim=-1)
                    index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
                    boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
                    scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
                    scores, index = scores.flatten(1).topk(min(max_det, anchors))
                    i = torch.arange(batch_size)[..., None]
                    return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()],
                                     dim=-1)
