import math

import torch


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # 转换box2维度 (n,4) -> (4,n) 以便广播计算
    box2 = box2.T

    # 解析坐标格式 --------------------------------------------------------
    # 如果输入是xyxy格式直接提取坐标，否则将xywh转换为xyxy格式
    if x1y1x2y2:  # 输入为左上右下坐标格式
        # 拆分box1坐标 (x1,y1,x2,y2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        # 拆分box2坐标 (x1,y1,x2,y2) -> (n个框的x1列表, ...)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # 输入为中心宽高格式，需要转换为xyxy
        # box1转换：中心坐标 -> 左上右下坐标
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        # box2转换：中心坐标 -> 左上右下坐标（向量化操作，处理多个框）
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # 计算交集区域 --------------------------------------------------------
    # 计算交集框的左上和右下坐标
    inter_x1 = torch.max(b1_x1, b2_x1)  # 交集左边界（所有框的x1最大值）
    inter_y1 = torch.max(b1_y1, b2_y1)  # 交集上边界
    inter_x2 = torch.min(b1_x2, b2_x2)  # 交集右边界（所有框的x2最小值）
    inter_y2 = torch.min(b1_y2, b2_y2)  # 交集下边界
    # 计算交集面积（使用clamp处理无交集情况，保证面积不小于0）
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # 计算并集区域 --------------------------------------------------------
    # 计算两个框各自的面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # box1面积
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # box2面积（向量化计算）
    # 并集面积 = 面积之和 - 交集面积 + 极小值（防止除以零）
    union_area = b1_area + b2_area - inter_area + eps

    # 基础IoU计算 --------------------------------------------------------
    iou = inter_area / union_area  # 交并比公式

    # 计算IoU变体（GIoU/DIoU/CIoU）---------------------------------------
    if GIoU or DIoU or CIoU:
        # 计算最小闭包框（能够包含两个框的最小矩形）
        c_x1 = torch.min(b1_x1, b2_x1)  # 闭包框左边界
        c_y1 = torch.min(b1_y1, b2_y1)  # 闭包框上边界
        c_x2 = torch.max(b1_x2, b2_x2)  # 闭包框右边界
        c_y2 = torch.max(b1_y2, b2_y2)  # 闭包框下边界
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + eps  # 闭包区域面积

        if DIoU or CIoU:  # 距离IoU或完整IoU计算
            # 计算中心点欧氏距离平方（分母为闭包框对角线平方）
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 / 4 +  # 水平中心距平方
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 / 4)  # 垂直中心距平方
            c_diag2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps  # 闭包框对角线平方

            if DIoU:  # 距离IoU（考虑中心点距离惩罚）
                return iou - rho2 / c_diag2  # DIoU公式

            elif CIoU:  # 完整IoU（附加宽高比一致性惩罚）
                # 计算宽高比的arctan差异（角度差异）
                w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps  # box1宽高
                w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps  # box2宽高
                arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)  # 宽高比角度差
                # 计算宽高比惩罚项（4/(π^2) * 角度差平方）
                v = (4 / (math.pi ** 2)) * torch.pow(arctan, 2)
                # 动态平衡参数alpha（分离计算图防止梯度传播）
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                # CIoU公式：综合中心距惩罚和宽高比惩罚
                return iou - (rho2 / c_diag2 + v * alpha)
        else:  # GIoU计算（考虑闭包区域惩罚）
            return iou - (c_area - union_area) / c_area  # GIoU公式

    return iou  # 返回基础IoU值