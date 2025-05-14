import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

def fitness(x):
    """
    计算模型综合性能得分，加权结合多个评估指标。
    权重w按顺序对应[精确率P, 召回率R, mAP@0.5, mAP@0.5:0.95]，重点优化mAP宽阈值指标。
    注: 前两指标权重为0，实际仅用mAP@0.5(10%)和mAP@0.5:0.95(90%)计算得分。
    """
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """构建(nc+1)x(nc+1)混淆矩阵，记录目标检测中分类与定位的综合匹配情况"""
        self.matrix = np.zeros((nc + 1, nc + 1))  # 末行/列用于背景误检(FN/FP)
        self.nc = nc  # 类别数
        self.conf = conf  # 检测框置信度阈值
        self.iou_thres = iou_thres  # IoU匹配阈值

    def process_batch(self, detections, labels):
        """
        处理单批次预测与真实框，更新混淆矩阵。
        核心逻辑：通过IoU阈值匹配预测框与真实框，区分正确检测/背景误检。
        - 检测框需满足置信度阈值conf
        - 使用匈牙利算法实现最优IoU匹配
        - 矩阵对角线记录正确分类，末列记录FP，末行记录FN
        """
        detections = detections[detections[:, 4] > self.conf]  # 置信度过滤
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])  # 计算IoU矩阵

        # 获取IoU超过阈值的匹配对
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            # 构建匹配矩阵[真实框索引, 预测框索引, IoU值]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            # 双重排序去重确保1:1匹配
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 保留预测框最优匹配
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 保留真实框最优匹配
        else:
            matches = np.zeros((0, 3))

        # 更新混淆矩阵
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # 正确匹配
            else:
                self.matrix[self.nc, gc] += 1  # 背景误检(FP)

        if n:
            # 未匹配的预测框记为背景漏检(FN)
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1

    def matrix(self):
        """返回当前混淆矩阵"""
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        """
        可视化混淆矩阵，生成归一化热力图。
        使用seaborn绘制，背景FP/FN单独显示，小数值隐藏优化可读性。
        图像保存为PNG格式，分辨率250dpi
        """
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # 按列归一化
            array[array < 0.005] = np.nan  # 过滤微小值

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # 动态调整字体大小
            labels = (0 < len(names) < 99) and len(names) == self.nc  # 是否使用类别标签
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        """打印原始混淆矩阵数值"""
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算边界框间多种IoU变体，支持标准IoU/GIoU/DIoU/CIoU。
    核心公式实现参考：
    - GIoU: https://arxiv.org/pdf/1902.09630.pdf
    - DIoU/CIoU: https://arxiv.org/abs/1911.08287v1

    参数说明：
    box1: 单个边界框[4,]或[n,4]
    box2: 多个边界框[n,4]，自动转置为[4,n]处理
    x1y1x2y2: 输入格式是否为角点坐标(True)或中心+宽高(False)
    GIoU/DIoU/CIoU: 切换不同计算模式，优先级CIoU > DIoU > GIoU > IoU
    eps: 极小值防止除零错误
    """
    # 坐标转换处理
    box2 = box2.T  # 统一转置为[4,n]维度适配广播计算

    # 获取坐标表示(角点模式直接取值，中心宽高模式转换计算)
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # 中心坐标转角点坐标计算
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # 计算交叠区域面积(利用广播机制批量计算)
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 计算并集面积(面积求和-交叠区域)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps  # 处理零宽度/高度
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps  # 加eps防止零除

    # 基础IoU计算
    iou = inter / union

    # 扩展IoU变体计算
    if GIoU or DIoU or CIoU:
        # 计算最小包围盒(convex box)的宽高
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 包围盒宽度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 包围盒高度

        if CIoU or DIoU:
            # DIoU/CIoU计算：中心点距离惩罚项
            c2 = cw  **  2 + ch  **  2 + eps  # 包围盒对角线平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)  **  2 +
                                                    (b2_y1 + b2_y2 - b1_y1 - b1_y2)  **  2) / 4  # 中心点欧式距离平方

            if DIoU:
                return iou - rho2 / c2  # DIoU = IoU - 中心距离惩罚

            elif CIoU:  # 完整CIoU计算(长宽比一致性惩罚)
                v = (4 / math.pi  **  2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))  # 自适应权重系数
                return iou - (rho2 / c2 + v * alpha)  # CIoU = IoU - 中心距离惩罚 - 长宽比惩罚
            return None #将return None设为显式

        else:  # GIoU计算：包围盒面积惩罚项
            c_area = cw * ch + eps  # 包围盒总面积
            return iou - (c_area - union) / c_area  # GIoU = IoU - 面积差异比

    else:
        return iou  # 标准IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    计算两组边界框间的交并比(IoU)，基于PyTorch向量化实现。
    输入框格式为(x1,y1,x2,y2)，返回NxM矩阵，N/M分别为box1/box2的框数量

    实现要点：
    - 利用广播机制批量计算所有框对组合，避免循环
    - 通过维度扩展实现张量间逐元素运算
    - 数值稳定处理：clamp(0)确保交叠区域非负

    参数：
    box1 (Tensor[N,4]): 第一组边界框，N为框数量
    box2 (Tensor[M,4]): 第二组边界框，M为框数量
    返回：
    iou (Tensor[N,M]): 所有框对组合的IoU值矩阵
    """

    def box_area(box):
        # 计算单组框面积，输入为4xn格式张量
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)  # 计算box1每个框面积，形状[N,]
    area2 = box_area(box2.T)  # 计算box2每个框面积，形状[M,]

    # 交集计算：通过维度扩展实现广播机制
    # box1[:, None, 2:]形状[N,1,2]，box2[:, 2:]形状[M,2]
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -  # 右边界最小值
             torch.max(box1[:, None, :2], box2[:, :2])  # 左边界最大值
             ).clamp(0).prod(2)  # 裁剪负值后求宽高乘积，形状[N,M]

    # IoU = 交集 / (面积1 + 面积2 - 交集)
    return inter / (area1[:, None] + area2 - inter)  # 广播后形状[N,M]

def bbox_ioa(box1, box2, eps=1E-7):
    """计算box1与box2各元素的交叠面积占box2面积之比(IoA)，适用于目标覆盖密度分析。
    输入要求：box1为单个边界框[x1,y1,x2,y2]，box2为多个边界框nx4矩阵
    特性：
    - 输出值范围[0,1]，1表示box1完全覆盖box2对应框
    - 通过eps防止除零错误，处理零面积框
    典型应用：评估单个预测框对多个真实框的覆盖程度，常用于密集场景遮挡分析
    """
    box2 = box2.transpose()  # 转置为4xn维度适配广播计算

    # 坐标分解(自动广播为向量运算)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # 交叠区域计算(支持向量化批量处理)
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2面积计算(添加极小值eps防零除)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # 返回IoA比率(交叠面积/box2面积)
    return inter_area / box2_area


class AverageMeter:
    """滑动平均值计算器，维护当前值(val)与滑动平均(avg)

    核心功能：
    - 累加更新：支持单次或批量更新(val*n)
    - 自动计算均值：通过总累加值(sum)与计数(count)的比值确定
    - 重置功能：快速清除历史记录

    典型应用：深度学习训练过程中loss/accuracy等指标的滑动平均追踪
    """

    def __init__(self):
        """初始化时自动调用reset，创建初始统计量"""
        self.reset()

    def reset(self):
        """重置所有统计量归零"""
        self.val = 0  # 当前最新值
        self.avg = 0  # 滑动平均值
        self.sum = 0  # 总值=Σ(val*n)
        self.count = 0  # 总样本数=Σn

    def update(self, val, n=1):
        """更新统计量，n表示当前val对应的样本数
        计算逻辑：
        sum累加当前值乘以样本数 -> sum += val*n
        样本数计数器累加 -> count +=n
        重新计算滑动平均 -> avg = sum/count
        """
        self.val = val  # 更新最新观测值
        self.sum += val * n  # 加权累加(支持批量更新)
        self.count += n  # 累计样本数量
        self.avg = self.sum / self.count  # 计算均值(自动浮点)

class MetricMeter(object):
    """指标集合管理器，支持多指标滑动平均计算与格式化输出。

    核心功能：
    - 内部使用AverageMeter自动维护各指标的当前值(val)和滑动平均(avg)
    - 支持字典形式批量更新指标
    - 提供字符串输出和平均值获取接口

    典型应用：深度学习训练过程中loss/accuracy等指标的追踪管理
    源码参考：https://github.com/KaiyangZhou/Dassl.pytorch
    """

    def __init__(self, delimiter='\t'):
        """初始化指标容器，delimiter指定字符串输出时的分隔符"""
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        """批量更新指标值，输入需为{指标名: 数值}字典。自动处理PyTorch张量类型"""
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        """格式化输出各指标，格式为'指标名 当前值 (滑动平均)'"""
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{} {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(output_str)

    def get_avg(self):
        """获取所有指标的滑动平均值列表，顺序与添加顺序一致"""
        res = []
        for name, meter in self.meters.items():
            res.append(meter.avg)
        return res

