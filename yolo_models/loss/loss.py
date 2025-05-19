import math

import torch
from torch import nn
from torch.nn import functional as F
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    """Focal Loss 实现，用于解决类别不平衡问题

    功能特性：
    - 包装现有二分类损失函数（需为BCEWithLogitsLoss）
    - 实现标准Focal Loss公式，支持alpha和gamma调节
    - 自动处理输出维度和归约方式
    - 包含数值稳定性优化

    参数说明：
    loss_fcn : 基础损失函数（必须为nn.BCEWithLogitsLoss）
    gamma    : 调节因子，>0时增加难样本权重（默认1.5）
    alpha    : 类别平衡因子，建议正样本比例较低时设为0.25（默认0.25）
    """

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        # 参数校验
        assert isinstance(loss_fcn, nn.BCEWithLogitsLoss), "必须使用BCEWithLogitsLoss作为基损失"

        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction  # 保存原始归约方式
        self.loss_fcn.reduction = 'none'  # 需逐个样本计算损失

    def forward(self, pred, true):
        """前向计算

        参数：
        pred : 模型预测输出（logits），形状[N, *]
        true : 真实标签（0/1），形状[N, *]

        返回：
        计算后的损失值，根据原始reduction方式归约
        """
        # 计算基础交叉熵损失（自动应用sigmoid）
        loss = self.loss_fcn(pred, true)  # 形状与pred相同

        # 计算概率并防止数值溢出
        pred_prob = torch.sigmoid(pred)  # 获取概率值
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # 正确类别的概率

        # 计算调节因子
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 类别权重
        modulating_factor = (1.0 - p_t) ** self.gamma  # 难易样本调节

        # 应用Focal Loss公式
        loss *= alpha_factor * modulating_factor  # 元素级相乘

        # 处理梯度极端值（防止NaN）
        loss = torch.clamp(loss, min=1e-6, max=1e6)  # 限制损失范围

        # 按原始归约方式处理
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class IOUloss(nn.Module):
    """用于计算多种IoU变体的损失函数，支持iou/giou/diou/ciou/siou等类型

    特点：
    - 同时支持xyxy(左上右下)和xywh(中心+宽高)两种框格式
    - 通过iou_type参数切换不同IoU计算方式
    - 支持none/sum/mean三种损失归约方式

    参数说明：
    reduction: 损失归约方式，"none"返回逐元素损失，"mean"取平均，"sum"求和
    iou_type: IoU类型，可选iou/giou/diou/ciou/siou
    xyxy: 输入框是否为xyxy格式(False表示使用xywh格式)
    """

    def __init__(self, reduction="none", iou_type="iou", xyxy=False):
        super(IOUloss, self).__init__()
        self.reduction = reduction  # 控制损失归约方式
        self.iou_type = iou_type    # 指定IoU计算类型
        self.xyxy = xyxy            # 输入框格式标识

    def forward(self, pred, target):
        """计算IoU损失前向传播

        参数：
        pred   : 预测框，形状[N,4]
        target : 真实框，形状[N,4]

        返回：
        根据reduction参数归约后的损失值
        """
        # 形状校验和格式转换
        assert pred.shape[0] == target.shape[0]
        pred = pred.view(-1, 4).float()    # 确保浮点计算
        target = target.view(-1, 4).float()

        # 根据输入格式计算交集的左上/右下坐标
        if self.xyxy:  # xyxy格式直接计算
            tl = torch.max(pred[:, :2], target[:, :2])  # 交集的左上点
            br = torch.min(pred[:, 2:], target[:, 2:])  # 交集的右下点
            area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1)  # 预测框面积
            area_g = torch.prod(target[:, 2:] - target[:, :2], 1)  # 真实框面积
        else:  # xywh格式需先转换坐标
            # 将中心坐标转为左上/右下坐标
            tl = torch.max(
                (pred[:, :2] - pred[:, 2:] / 2),
                (target[:, :2] - target[:, 2:] / 2)
            )
            br = torch.min(
                (pred[:, :2] + pred[:, 2:] / 2),
                (target[:, :2] + target[:, 2:] / 2)
            )
            area_p = torch.prod(pred[:, 2:], 1)  # 宽高直接相乘
            area_g = torch.prod(target[:, 2:], 1)

        # 计算交集和IoU
        hw = (br - tl).clamp(min=0)  # 确保宽高非负
        area_i = torch.prod(hw, 1)    # 交集面积
        iou = area_i / (area_p + area_g - area_i + 1e-16)  # 防止除以零

        # 根据不同类型计算损失
        if self.iou_type == "iou":
            loss = 1 - iou ** 2  # 基础IoU损失
        elif self.iou_type == "giou":
            # 计算最小闭包区域
            if self.xyxy:
                c_tl = torch.min(pred[:, :2], target[:, :2])
                c_br = torch.max(pred[:, 2:], target[:, 2:])
            else:
                c_tl = torch.min(pred[:, :2] - pred[:, 2: ] /2,
                                 target[:, :2] - target[:, 2: ] /2)
                c_br = torch.max(pred[:, :2] + pred[:, 2: ] /2,
                                 target[:, :2] + target[:, 2: ] /2)
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)  # 限制GIoU范围
        elif self.iou_type == "diou":
            # 计算中心点距离和最小闭包对角线
            if self.xyxy:
                c_tl = torch.min(pred[:, :2], target[:, :2])
                c_br = torch.max(pred[:, 2:], target[:, 2:])
            else:
                c_tl = torch.min(
                    (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)  # 包围框的左上点
                )
                c_br = torch.max(
                    (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)  # 包围框的右下点
                )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            diou = iou - (center_dis / convex_dis)
            loss = 1 - diou.clamp(min=-1.0, max=1.0)
        elif self.iou_type == "ciou":
            # 在DIoU基础上增加宽高比惩罚项
            if self.xyxy:
                c_tl = torch.min(pred[:, :2], target[:, :2])
                c_br = torch.max(pred[:, 2:], target[:, 2:])
            else:
                c_tl = torch.min(
                    (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
                )
                c_br = torch.max(
                    (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
                )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared
            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            v = (4 / math.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) -
                                               torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-7)), 2)
            # 使用arctan计算宽高比差异
            v = (4/math.pi**2) * torch.pow(
                torch.atan(target[: ,2 ] /target[: ,3].clamp(1e-7)) -
                torch.atan(pred[: ,2 ] /pred[: ,3].clamp(1e-7)), 2)
            alpha = v / ((1 + 1e-7) - iou + v)  # 自动确定惩罚权重

        elif self.iou_type == 'siou':
            # 包含角度成本、距离成本、形状成本
            box1 = pred.T
            box2 = target.T
            if self.xyxy:
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            else:
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7

            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + 1e-7
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + 1e-7
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
            loss = 1.0 - iou.clamp(min=-1.0, max=1.0)


        # 损失归约处理
        if self.reduction == "mean":
            loss = loss.mean()  # 批平均
        elif self.reduction == "sum":
            loss = loss.sum()   # 批求和

        return loss


def wing_loss(pred, target, omega=10.0, epsilon=2.0):
    """Wing Loss 实现，用于关键点检测任务

    参数说明：
    pred    : 预测值，形状为(batch_size, 关键点数, 坐标维度)
              (例：[N, 17, 2] 表示batch_size=N，17个人体关键点，每个点xy坐标)
    target  : 目标值，形状需与pred一致
    omega   : 分界阈值，控制对数区域与线性区域的交界（默认10.0）
    epsilon : 对数区域曲率调节因子（默认2.0）

    返回：
    各样本各关键点的损失值，形状与输入相同
    """

    # 计算连续性常数C，确保分界点处函数连续
    # 公式推导：当diff=ω时，左右两边相等 ω*ln(1+ω/ε) = ω - C → C=ω*(1 - ln(1+ω/ε))
    C = omega * (1.0 - math.log(1.0 + omega / epsilon))  # 预计算常数项

    # 计算绝对误差（保持形状不变）
    diff = torch.abs(pred - target)  # 形状[N, K, D]

    # 分段计算损失值 ------------------------------------------------------
    # 当误差小于ω时使用对数区域，增强小误差的梯度
    # 当误差大于等于ω时使用线性区域，避免梯度爆炸
    losses = torch.where(
        diff < omega,
        omega * torch.log(1.0 + diff / epsilon),  # 对数区域：ω * ln(1 + d/ε)
        diff - C  # 线性区域：d - C
    )

    """
    Wing Loss曲线特性：
    - 当d趋近0时，梯度约为 ω/(ε*(1 + d/ε)) → ω/ε，放大微小误差的梯度
    - 当d≥ω时，梯度恒为1，与L1 Loss一致
    - 分界点d=ω处函数值连续但不可导（需通过其他方法平滑处理）
    """

    return losses  # 返回逐点损失值，形状[N, K, D]

class WingLoss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', form='obb', loss_weight=1.0):
        """
        Wing Loss 实现（当前版本参数存在未使用情况）

        参数说明：
        beta       : 控制损失曲线过渡区的参数（当前实现未使用）
        reduction  : 损失归约方式，支持'mean'或'sum'（实际强制使用mean归一化）
        form       : 输入格式标识（当前未使用，保留供后续扩展）
        loss_weight: 损失权重系数（当前未生效，需在forward中应用）
        """
        super(WingLoss, self).__init__()
        self.reduction = reduction  # 参数未实际生效（强制使用mean归一化）
        self.loss_weight = loss_weight  # 未在计算中使用
        self.form = form  # 未使用参数

    def forward(self, pred, target):
        """前向计算（存在潜在问题：beta未生效，loss_weight未应用）

        参数：
        pred   : 预测值，形状任意（需与target相同）
        target : 目标值，>0的位置视为有效样本

        返回：
        归一化后的损失值（实际为sum(masked_loss)/sum(mask)）
        """
        # 生成掩码（仅计算target>0位置的损失）
        mask = target > 0  # 假设无效/缺失标注用0填充

        # 计算基础损失（需确保wing_loss函数正确处理mask）
        losses = wing_loss(pred * mask, target.float() * mask)  # wing_loss实现未展示

        # 强制mean归一化（分母添加极小值防止除零）
        return losses.sum() / (torch.sum(mask) + 10e-14)  # 等效于mean忽略无效位置
        # 注：原始实现注释显示曾考虑直接返回mean()


class LandmarksLossYolov5(nn.Module):
    """关键点检测损失函数（基于Wing Loss改进版）

    功能特性：
    - 专为YOLOv5关键点检测任务设计
    - 使用Wing Loss增强小范围误差的敏感性
    - 支持掩码机制处理缺失标注
    - 自动归一化损失值

    参数说明：
    alpha : 损失缩放因子（当前版本暂未直接使用，保留用于后续扩展）
    """

    def __init__(self, alpha=1.0):
        """
        初始化函数
        :param alpha: 损失缩放因子（当前实现中实际未使用，保留参数用于未来扩展）
        """
        super(LandmarksLossYolov5, self).__init__()
        self.loss_fcn = WingLoss()  # 原始实现使用SmoothL1Loss(reduction='sum')
        self.alpha = alpha  # 当前版本未实际使用，保留参数

    def forward(self, pred, truel, mask):
        """
        前向计算
        :param pred   : 预测关键点坐标，形状(batch_size, num_landmarks*2)
        :param truel  : 真实关键点坐标，形状与pred相同
        :param mask   : 掩码矩阵，形状与pred相同（1表示有效点，0表示忽略）
        :return: 归一化后的损失值
        """
        # 掩码应用（逐元素相乘过滤无效点）
        # pred*mask 和 truel*mask 将无效位置置零
        loss = self.loss_fcn(pred * mask, truel * mask)

        # 归一化处理（分母加极小值防止除零）
        # 有效点数量 = mask矩阵求和（每个点的贡献为1）
        # 10e-14 保证即使所有点无效时仍能计算
        return loss / (torch.sum(mask) + 10e-14)


class ComputeLoss:
    """YOLO综合损失计算类，整合边界框/置信度/分类/关键点损失

    功能特性：
    - 支持多尺度特征图(P3-P7)损失平衡
    - 包含标签平滑、Focal Loss等训练优化策略
    - 支持动态标签分配策略(SimOTA)
    - 兼容单目标/多目标检测场景
    - 整合关键点检测损失（如人体姿态估计）

    初始化参数说明：
    model : 要计算损失的YOLO模型实例
    cfg : 包含损失相关配置的配置对象
    """

    def __init__(self, model, cfg):
        # 基础配置初始化
        self.sort_obj_iou = False  # 是否对IoU排序（传统YOLO做法）
        device = next(model.parameters()).device  # 获取模型所在设备

        # 超参数解析
        autobalance = cfg.Loss.autobalance  # 是否自动平衡多尺度损失
        cls_pw = cfg.Loss.cls_pw  # 分类损失正样本权重
        obj_pw = cfg.Loss.obj_pw  # 置信度损失正样本权重
        label_smoothing = cfg.Loss.label_smoothing  # 标签平滑系数

        # 基础损失函数定义 --------------------------------------------------
        # 带标签平滑的BCE损失（分类和置信度）
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=device))

        # 标签平滑处理（正负样本标签平滑值）
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # (正样本平滑值, 负样本平滑值)

        # Focal Loss增强（当fl_gamma>0时启用）
        g = cfg.Loss.fl_gamma
        if g > 0:  # 用Focal Loss包装基础BCE损失
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # 模型结构相关参数 --------------------------------------------------
        det = model.module.head if is_parallel(model) else model.head  # 获取检测头模块
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # 多尺度损失平衡参数(P3-P7)
        self.ssi = list(det.stride).index(16) if autobalance else 0  # 自动平衡参考尺度（stride16对应的特征层索引）

        # 损失权重配置 ------------------------------------------------------
        self.BCEcls, self.BCEobj = BCEcls, BCEobj  # 分类/置信度损失函数
        self.gr = 1.0  # GIoU比例系数（用于混合IoU损失）
        self.autobalance = autobalance  # 自动平衡标志

        nl = det.nl  # 检测层数量（特征图数量）
        nc = 1 if cfg.single_cls else cfg.Dataset.nc  # 类别数量（单类模式开关）

        # 损失权重系数（来自配置文件）：
        self.box_w = cfg.Loss.box * 3.0 / nl  # 边界框损失权重
        self.obj_w = cfg.Loss.obj  # 置信度损失权重
        self.cls_w = cfg.Loss.cls * nc / 80. * 3. / nl  # 分类损失权重（标准化到COCO的80类基准）

        # 关键点检测相关 ----------------------------------------------------
        self.LandMarkLoss = LandmarksLossYolov5(1.0)  # 关键点回归损失（通常使用L1或Wing Loss）
        self.num_keypoints = det.num_keypoints  # 关键点数量

        # Anchor匹配参数 ---------------------------------------------------
        self.anchor_t = cfg.Loss.anchor_t  # Anchor宽高比阈值（超参数，通常4.0）
        self.single_targets = cfg.Loss.single_targets  # 是否单目标匹配（每个GT只匹配一个Anchor）

        # 标签分配策略 -----------------------------------------------------
        self.ota = cfg.Loss.assigner_type == 'SimOTA'  # 是否启用动态SimOTA分配器
        self.top_k = cfg.Loss.top_k  # OTA分配候选数

        # 传统分配器与OTA分配器双配置
        self.assigner = YOLOAnchorAssigner(
            self.na, self.nl, self.anchors, self.anchor_t, det.stride,
            self.nc, self.num_keypoints, single_targets=self.single_targets, ota=False
        )
        self.ota_assigner = YOLOAnchorAssigner(
            self.na, self.nl, self.anchors, self.anchor_t, det.stride,
            self.nc, self.num_keypoints, single_targets=self.single_targets, ota=self.ota, top_k=self.top_k
        )

    def default_loss(self, p, targets):
        """计算YOLO多任务损失的核心方法，包含边界框/置信度/分类/关键点损失

        参数：
        p : 模型预测输出列表，每个元素对应一个检测层的预测张量
        targets : 真实标注框，形状为[num_targets, 6+]，每行格式为(image_id, class_id, x, y, w, h, ...keypoints)

        返回：
        loss : 总损失值（乘以batch size）
        loss_dict : 包含各损失分量的字典
        """
        device = targets.device
        # 初始化基础损失分量
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 关键点检测相关初始化
        if self.num_keypoints > 0:
            tcls, tbox, indices, anchors, tlandmarks, lmks_mask = self.assigner(p, targets)
        else:
            tcls, tbox, indices, anchors = self.assigner(p, targets)
        lmark = torch.zeros(1, device=device)

        # 逐层计算损失 ---------------------------------------------------------
        for i, pi in enumerate(p):  # 遍历每个检测层
            b, a, gj, gi = indices[i]  # 解析匹配结果：batch索引, anchor索引, 网格坐标
            tobj = torch.zeros_like(pi[..., 0], device=device)  # 初始化目标置信度张量

            n = b.shape[0]  # 当前层匹配的目标数量
            if n:
                ps = pi[b, a, gj, gi]  # 提取匹配位置的预测值，形状为[n_targets, n_params]

                # 边界框回归部分 -----------------------------------------------
                pxy = ps[:, :2].sigmoid() * 2. - 0.5  # 解码中心坐标偏移量
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 解码宽高缩放
                pbox = torch.cat((pxy, pwh), 1)  # 组合成预测边界框[x_center, y_center, width, height]
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # 计算CIoU
                lbox += (1.0 - iou).mean()  # 累积边界框损失

                # 置信度目标生成 ----------------------------------------------
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:  # 按IoU排序（传统YOLO策略）
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # 混合目标值（gr=1时为纯IoU）

                # 关键点回归部分 -----------------------------------------------
                if self.num_keypoints > 0:
                    plandmarks = ps[:, -self.num_keypoints * 2:]  # 提取关键点预测
                    # 按anchor尺寸缩放关键点坐标
                    for idx in range(self.num_keypoints):
                        plandmarks[:, (0 + (2 * idx)):(2 + (2 * idx))] *= anchors[i]
                    lmark += self.LandMarkLoss(plandmarks, tlandmarks[i], lmks_mask[i])

                # 分类损失计算 ------------------------------------------------
                if self.nc > 1:  # 多类别时才计算分类损失
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # 初始化平滑标签
                    t[range(n), tcls[i]] = self.cp  # 设置正样本平滑值
                    lcls += self.BCEcls(ps[:, 5:], t)  # 二值交叉熵损失

            # 置信度损失计算 --------------------------------------------------
            obji = self.BCEobj(pi[..., 4], tobj)  # 当前层置信度损失
            lobj += obji * self.balance[i]  # 应用层间平衡系数
            if self.autobalance:  # 动态平衡策略
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        # 损失权重调整 --------------------------------------------------------
        if self.autobalance:  # 归一化平衡系数
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_w  # 边界框损失权重
        lobj *= self.obj_w  # 置信度损失权重
        lcls *= self.cls_w  # 分类损失权重

        # 总损失组合 ---------------------------------------------------------
        bs = tobj.shape[0]  # 获取batch size
        loss = (lbox + lobj + lcls) * bs  # 加权求和并乘以batch size
        loss_dict = dict(box=lbox.detach(), obj=lobj.detach(),
                         cls=lcls.detach(), loss=loss.detach())

        return loss, loss_dict

    def ota_loss(self, p, targets):
        """SimOTA动态分配策略的损失计算实现

        核心改进：
        - 引入SimOTA动态标签分配，替代静态规则匹配
        - 基于代价矩阵进行全局最优分配
        - 支持top-k候选选择机制

        参数流程：
        p : 各检测层的预测输出列表，每个元素为[batch, anchors, grid_h, grid_w, box_params+cls_params+keypoints?]
        targets : 真实标注框，格式为[num_targets, batch_id+class_id+box+keypoints?]
        """
        device = targets.device
        # 损失分量初始化
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # SimOTA动态分配执行 -----------------------------------------------------
        # 返回：
        # bs: 各层分配的batch索引列表 [层数][分配数]
        # as_: 各层分配的anchor索引列表
        # gjs/gis: 各层分配的网格坐标列表
        # ota_targets: 调整后的目标参数列表 [层数][分配数, 6(含class)]
        # anchors: 匹配的anchor尺寸列表 [层数][分配数, 2]
        bs, as_, gjs, gis, ota_targets, anchors = self.ota_assigner(p, targets)

        # 网格增益计算（用于坐标反归一化）[width, height, width, height]
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p]

        # 逐层处理循环 ----------------------------------------------------------
        for i, pi in enumerate(p):  # 遍历检测层（P3-P5/P7）
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # 当前层分配结果解包
            tobj = torch.zeros_like(pi[..., 0], device=device)  # 初始化目标置信度张量

            n = b.shape[0]  # 当前层分配的目标数
            if n:
                ps = pi[b, a, gj, gi]  # 提取匹配位置的预测值[n, params]

                # 边界框回归解码 ------------------------------------------------
                grid = torch.stack([gi, gj], dim=1)  # 对应的网格左上角坐标
                pxy = ps[:, :2].sigmoid() * 2. - 0.5  # 中心坐标解码（-0.5~1.5倍网格偏移）
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 宽高解码（0~4倍anchor尺寸）
                pbox = torch.cat((pxy, pwh), 1)  # 组合为预测框[x_center, y_center, w, h]

                # 目标框调整 ---------------------------------------------------
                selected_tbox = ota_targets[i][:, 2:6] * pre_gen_gains[i]  # 反归一化到特征图尺度
                selected_tbox[:, :2] -= grid  # 转换为相对于当前网格的偏移量
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # CIoU计算

                # 损失累积
                lbox += (1.0 - iou).mean()  # 边界框损失

                # 动态置信度目标生成 --------------------------------------------
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                # 分类损失计算 -------------------------------------------------
                selected_tcls = ota_targets[i][:, 1].long()  # 获取类别索引
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:5 + self.nc], self.cn, device=device)  # 初始化平滑标签
                    t[range(n), selected_tcls] = self.cp  # 设置正样本平滑值
                    lcls += self.BCEcls(ps[:, 5:5 + self.nc], t)  # 分类损失

            # 置信度损失计算及自动平衡 -------------------------------------------
            obji = self.BCEobj(pi[..., -1], tobj)  # 当前层置信度损失
            lobj += obji * self.balance[i]  # 应用层平衡系数
            if self.autobalance:  # 动态更新平衡系数（指数平滑）
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        # 传统分配补充处理（可能用于稳定训练）-------------------------------------
        tcls, tbox, indices, anchors = self.assigner(p, targets)  # 传统静态分配
        for i, pi in enumerate(p):  # 再次遍历各层
            b, a, gj, gi = indices[i]  # 传统分配结果
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]  # 传统分配的预测值

                # 边界框解码与损失计算（同上）
                # ...（与前面类似流程，此处可能用于增强监督信号）

            # 置信度损失累积（可能造成重复计算，需注意权重平衡）
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

        # 损失加权与输出 --------------------------------------------------------
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]  # 归一化平衡系数
        lbox *= self.box_w  # 边界框损失权重
        lobj *= self.obj_w  # 置信度损失权重
        lcls *= self.cls_w  # 分类损失权重

        bs = tobj.shape[0]  # batch size
        loss = lbox + lobj + lcls
        loss_dict = dict(box=lbox, obj=lobj, cls=lcls, loss=loss * bs)

        return loss * bs, loss_dict

    def __call__(self, p, targets):
        """损失计算入口函数，根据配置选择分配策略"""
        return self.ota_loss(p, targets) if self.ota else self.default_loss(p, targets)


class DomainFocalLoss(nn.Module):
    """改进后的领域焦点损失函数，支持多分类与二分类场景

    功能特性：
    - 支持标准多分类Focal Loss
    - 支持二分类场景（sigmoid模式）
    - 自动处理类别权重平衡（alpha参数）
    - 支持gamma参数调节难易样本权重
    - 兼容PyTorch最新版本（移除Variable包装）

    参数说明：
    class_num     : 类别数量（二分类时需设为1）
    alpha         : 类别权重系数，可传入列表或张量
    gamma         : 调节因子，默认为2
    size_average  : 是否对batch求平均损失
    sigmoid       : 是否使用sigmoid激活（二分类模式）
    reduce        : 是否对样本损失求和/平均
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, sigmoid=False, reduce=True):
        super(DomainFocalLoss, self).__init__()

        # 初始化类别权重
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)  # 默认均匀权重
        else:
            self.alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)

        self.alpha = self.alpha.view(-1)  # 展平为向量 [C]
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce

    def forward(self, inputs, targets):
        """前向计算

        参数：
        inputs : 模型输出，形状[N, C]
        targets: 真实标签，二分类时形状[N]，多分类时形状[N]

        返回：
        loss : 计算得到的损失值
        """
        N = inputs.size(0)
        C = inputs.size(1)

        if self.sigmoid:
            # 二分类模式（领域适应常用）
            probs = torch.sigmoid(inputs)  # 形状[N, 1]
            probs = probs.squeeze()  # 形状[N]

            # 根据target选择正负类计算方式
            pos_mask = targets.eq(1).float()
            neg_mask = targets.eq(0).float()

            pos_loss = - (self.alpha[1] * torch.pow(1 - probs, self.gamma) * torch.log(probs + 1e-6)) * pos_mask
            neg_loss = - (self.alpha[0] * torch.pow(probs, self.gamma) * torch.log(1 - probs + 1e-6)) * neg_mask

            batch_loss = pos_loss + neg_loss
        else:
            # 多分类模式
            probs = F.softmax(inputs, dim=1)  # 形状[N, C]

            # 生成类别掩码
            class_mask = torch.zeros_like(probs)  # 形状[N, C]
            class_mask.scatter_(1, targets.unsqueeze(1), 1.0)  # 对应类别位置填1

            # 按类别选取alpha
            alpha = self.alpha[targets].view(-1, 1)  # 形状[N, 1]

            # 计算各样本对应类别的概率
            selected_probs = (probs * class_mask).sum(dim=1, keepdim=True)  # 形状[N, 1]

            # 计算Focal Loss
            batch_loss = -alpha * torch.pow(1 - selected_probs, self.gamma) * torch.log(selected_probs + 1e-6)

        # 归约处理
        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class DomainLoss():
    """领域适应损失计算器，用于多尺度特征对齐

    功能特性：
    - 处理三个不同尺度的特征图（8x8, 16x16, 32x32）
    - 将空间特征转换为领域分类向量
    - 使用Focal Loss进行领域对齐优化

    初始化参数：无
    """

    def __init__(self):
        # 初始化二分类领域焦点损失（源域 vs 目标域）
        self.fl = DomainFocalLoss(class_num=2)  # class_num=2表示二分类问题

    def __call__(self, feature):
        """计算领域适应损失

        参数：
        feature : 包含三个尺度特征的元组/列表
                  [out_8, out_16, out_32]
                  每个特征的形状应为 (batch_size, 2, H, W)

        返回：
        dloss_s : 计算得到的领域对齐损失值
        """

        # 解包多尺度特征 --------------------------------------------------
        out_8 = feature[0]  # 高分辨率特征 (batch, 2, H8, W8)
        out_16 = feature[1]  # 中分辨率特征 (batch, 2, H16, W16)
        out_32 = feature[2]  # 低分辨率特征 (batch, 2, H32, W32)

        # 特征形状重塑 ---------------------------------------------------
        # 将通道维度放到最后并展平空间维度
        # 原始形状: (batch, 2, H, W) → 重塑后: (batch*H*W, 2)
        out_d_s_8 = out_8.permute(0, 2, 3, 1).reshape(-1, 2)  # 空间位置数 batch*H8*W8
        out_d_s_16 = out_16.permute(0, 2, 3, 1).reshape(-1, 2)  # batch*H16*W16
        out_d_s_32 = out_32.permute(0, 2, 3, 1).reshape(-1, 2)  # batch*H32*W32

        # 多尺度特征合并 -------------------------------------------------
        out_d_s = torch.cat((out_d_s_8, out_d_s_16, out_d_s_32), dim=0)  # 合并后形状 (N_total, 2)
        # N_total = sum(batch*Hi*Wi)

        # 生成领域标签 ---------------------------------------------------
        domain_s = torch.zeros(out_d_s.size(0), dtype=torch.long).cuda()  # 全0标签（假设均为源域）
        # 实际应用可能需要动态调整

        # 计算领域对齐损失 -------------------------------------------------
        dloss_s = 0.5 * self.fl(out_d_s, domain_s)  # 0.5为损失权重系数

        return dloss_s


class TargetLoss():
    """目标域特征对齐损失计算器，与源域损失配套使用

    功能特性：
    - 处理三个尺度的目标域特征图（8x8, 16x16, 32x32）
    - 特征空间位置展开为样本维度
    - 使用全1标签表示目标域分类
    - 通过Focal Loss优化目标域特征判别

    初始化参数：无
    """

    def __init__(self):
        # 初始化二分类Focal Loss（目标域vs源域）
        self.fl = DomainFocalLoss(class_num=2)  # 类别数为2表示域分类任务

    def __call__(self, feature):
        """计算目标域对齐损失

        参数：
        feature : 包含三个尺度特征的元组/列表
                  [out_8, out_16, out_32]
                  每个特征的形状应为 (batch_size, 2, H, W)

        返回：
        dloss_t : 目标域对齐损失值
        """

        # 解包多尺度特征 --------------------------------------------------
        out_8 = feature[0]  # 高分辨率特征 (batch, 2, H8, W8)
        out_16 = feature[1]  # 中分辨率特征 (batch, 2, H16, W16)
        out_32 = feature[2]  # 低分辨率特征 (batch, 2, H32, W32)

        # 特征形状重塑 ---------------------------------------------------
        # 将通道维度移到最后并展平空间维度
        # (batch, 2, H, W) → (batch*H*W, 2)
        out_d_t_8 = out_8.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_t_16 = out_16.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_t_32 = out_32.permute(0, 2, 3, 1).reshape(-1, 2)

        # 多尺度特征合并 -------------------------------------------------
        out_d_t = torch.cat((out_d_t_8, out_d_t_16, out_d_t_32), dim=0)  # 形状 (N_total, 2)

        # 生成目标域标签 -------------------------------------------------
        domain_t = torch.ones(out_d_t.size(0), dtype=torch.long).cuda()  # 全1标签表示目标域

        # 计算目标域对齐损失 ---------------------------------------------
        dloss_t = 0.5 * self.fl(out_d_t, domain_t)  # 0.5为损失平衡系数

        return dloss_t


class ComputeFastXLoss:
    # Compute losses
    def __init__(self, model, cfg):
        # super(ComputeFastXLoss, self).__init__()
        # device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        det = model.module.head if is_parallel(model) else model.head  # Detect() module
        self.det = det
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_type = cfg.Loss.iou_type
        # self.iou_loss = IOUloss(iou_type=self.iou_type, reduction="none")
        self.iou_loss = IOUloss(iou_type=self.iou_type, reduction="none")
        self.num_classes = cfg.Dataset.nc
        self.strides = torch.tensor(cfg.Model.Head.strides)
        self.reg_weight = cfg.Loss.box_loss_weight
        self.obj_weight = cfg.Loss.obj_loss_weight
        self.cls_weight = cfg.Loss.cls_loss_weight
        self.n_anchors = len(cfg.Model.anchors)
        self.grids = [torch.zeros(1)] * len(cfg.Model.Head.in_channels)
        self.iou_obj = cfg.Loss.iou_obj
        self.formal_assigner = SimOTAAssigner(num_classes=self.num_classes, iou_weight=3.0, cls_weight=1.0,
                                              center_radius=2.5, iou_obj=self.iou_obj)

    def __call__(
            self,
            outputs,
            targets
    ):
        dtype = outputs[0].type()
        device = targets.device
        # print('targets type:', targets.type())
        loss_cls, loss_obj, loss_iou, loss_l1 = torch.zeros(1, device=device), torch.zeros(1, device=device), \
            torch.zeros(1, device=device), torch.zeros(1, device=device)

        outputs, outputs_origin, gt_bboxes_scale, xy_shifts, expanded_strides = self.get_outputs_and_grids(
            outputs, self.strides, dtype, device)

        with torch.cuda.amp.autocast(enabled=False):
            bbox_preds = outputs[:, :, :4].float()  # [batch, n_anchors_all, 4]
            bbox_preds_org = outputs_origin[:, :, :4].float()  # [batch, n_anchors_all, 4]
            obj_preds = outputs[:, :, 4].float().unsqueeze(-1)  # [batch, n_anchors_all, 1]
            cls_preds = outputs[:, :, 5:].float()  # [batch, n_anchors_all, n_cls]

            # targets
            batch_size = bbox_preds.shape[0]
            targets = self.preprocess(targets, batch_size, gt_bboxes_scale)

            # 添加assigner的实现
            cls_targets, reg_targets, obj_targets, l1_targets, fg_masks, num_fg, num_gts = self.formal_assigner(
                outputs.detach(),
                targets,
                bbox_preds.detach(),
                cls_preds.detach(),
                obj_preds.detach(),
                expanded_strides,
                xy_shifts)

            # loss
            # loss_iou += bbox_preds[i].mean() * 0
            # loss_cls += cls_preds.mean() * 0
            # loss_obj += obj_preds.mean() * 0
            # loss_l1 += bbox_preds_org.mean() * 0
            # for i, n in enumerate(num_fg): # batch_size, n_fg
            # fg_mask = fg_masks[i]
            # if n:
            #     loss_iou += (self.iou_loss(bbox_preds[i].view(-1, 4)[fg_mask], reg_targets[i])).mean()
            #     loss_l1 += (self.l1_loss(bbox_preds_org[i].view(-1, 4)[fg_mask], l1_targets[i])).mean()
            #     # loss_obj += (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets*1.0)).mean() * batch_size
            #     loss_cls += (self.bcewithlog_loss(cls_preds[i].view(-1, self.num_classes)[fg_mask], cls_targets[i])).mean()
            #     loss_obj += (self.bcewithlog_loss(obj_preds[i].view(-1, 1), obj_targets[i])).mean()
            # else:
            #     loss_iou += bbox_preds_org[i].mean() * 0
            #     loss_l1 += bbox_preds_org[i].mean() * 0
            #     loss_cls += cls_preds[i].mean() * 0
            #     loss_obj += obj_preds[i].mean() * 0
            # print('bbox preds org:', bbox_preds_org[i].mean())
            # print('bbox preds org:', bbox_preds[i].mean())
            # if torch.isnan(loss_iou).any():
            #     print('bbox_preds:', bbox_preds[i].type())
            #     print('bbox_preds:', bbox_preds[i])
            #     print('reg:', reg_targets)
            # print(num_fg)
            loss_iou += (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
            loss_l1 += (self.l1_loss(bbox_preds_org.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
            # loss_obj += (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets*1.0)).mean() * batch_size
            loss_cls += (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks],
                                              cls_targets)).sum() / num_fg
            loss_obj += (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        # # if torch.isnan(loss_l1).any():
        # #     print('l1:', l1_targets)
        # #     print(num_fg)
        # if torch.isnan(loss_obj).any():
        #     print('obj:', obj_targets)
        #     print(num_fg)
        # if torch.isnan(loss_cls).any():
        #     print('cls:', cls_targets)
        #     print(num_fg)

        total_losses = self.reg_weight * loss_iou + loss_l1 + self.obj_weight * loss_obj + self.cls_weight * loss_cls
        # total_losses = total_losses * batch_size
        loss_dict = dict(loss_iou=self.reg_weight * loss_iou, loss_obj=self.obj_weight * loss_obj,
                         loss_cls=self.cls_weight * loss_cls, loss=float(total_losses))
        return total_losses, loss_dict

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(
            np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]).to(
            targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = batch_target

        return targets

    def decode_output(self, output, k, stride, dtype, device):
        grid = self.grids[k].to(device)
        batch_size = output.shape[0]
        hsize, wsize = output.shape[2:4]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype).to(device)
            self.grids[k] = grid

        output = output.reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        output_origin = output.clone()
        grid = grid.view(1, -1, 2)

        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, output_origin, grid, hsize, wsize

    def get_outputs_and_grids(self, outputs, strides, dtype, device):
        xy_shifts = []
        expanded_strides = []
        outputs_new = []
        outputs_origin = []

        for k, output in enumerate(outputs):
            output, output_origin, grid, feat_h, feat_w = self.decode_output(
                output, k, strides[k], dtype, device)

            xy_shift = grid
            expanded_stride = torch.full((1, grid.shape[1], 1), strides[k], dtype=grid.dtype, device=grid.device)

            xy_shifts.append(xy_shift)
            expanded_strides.append(expanded_stride)
            outputs_new.append(output)
            outputs_origin.append(output_origin)

        xy_shifts = torch.cat(xy_shifts, 1)  # [1, n_anchors_all, 2]
        expanded_strides = torch.cat(expanded_strides, 1)  # [1, n_anchors_all, 1]
        outputs_origin = torch.cat(outputs_origin, 1)
        outputs = torch.cat(outputs_new, 1)

        feat_h *= strides[-1]
        feat_w *= strides[-1]
        gt_bboxes_scale = torch.Tensor([[feat_w, feat_h, feat_w, feat_h]]).type_as(outputs)

        return outputs, outputs_origin, gt_bboxes_scale, xy_shifts, expanded_strides


class ComputeNanoLoss:
    def __init__(self, model, cfg):
        super(ComputeNanoLoss, self).__init__()
        det = model.module.head if is_parallel(model) else model.head  # Detect() module
        self.det = det
        # self.loss_dict = {'loss_qfl':0, 'loss_bbox':0, 'loss_dfl':0}

    def __call__(self, p, targets):
        loss, loss_dict = self.det.get_losses(p, targets)
        return loss, loss_dict