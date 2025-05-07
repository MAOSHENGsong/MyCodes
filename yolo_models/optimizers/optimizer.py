import copy

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.nn import functional as F

from yolo_models.backbone.common import LinearAddBlock, RealVGGBlock


def extract_blocks_into_list(model, blocks):
    """ 递归遍历模型，收集特定类型的模块到列表中

    深度优先遍历模型的所有子模块，将类型为 LinearAddBlock 或 RealVGGBlock 的模块
    添加到传入的 blocks 列表中。该函数会跳过目标模块的子模块（即只收集最外层匹配的模块）

    Args:
        model (nn.Module): 待遍历的神经网络模型
        blocks (list): 用于存储目标模块的列表（原地修改）

    工作流程：
        1. 遍历模型的直接子模块（model.children()）
        2. 若子模块是目标类型，直接加入列表并停止深入其子模块
        3. 若子模块不是目标类型，则递归遍历其子模块
    """
    for module in model.children():
        # 判断当前模块是否为需要收集的目标类型
        if isinstance(module, (LinearAddBlock, RealVGGBlock)):
            blocks.append(module)
        else:
            # 若当前模块不是目标类型，则递归遍历其子模块
            extract_blocks_into_list(module, blocks)


def get_optimizer_param(cfg, model):
    """ 根据模型结构和配置生成优化器参数分组

    将模型参数划分为三组，实现不同参数类型差异化优化策略：
    1. 批归一化层权重（无权重衰减）
    2. 普通层权重（应用权重衰减）
    3. 所有偏置项（无权重衰减）

    Args:
        cfg (Config): 配置文件对象，需包含 hyp.weight_decay 配置项
        model (nn.Module): 待优化的神经网络模型

    Returns:
        list[dict]: 优化器参数组配置列表，可直接传入torch.optim.Optimizer

    设计要点：
        - 遵循YOLOv5优化策略：BN层权重和偏置项不应用权重衰减
        - 通过参数分组实现更精细化的优化控制
    """
    # 初始化参数分组容器
    g_bnw, g_w, g_b = [], [], []  # 分别存储BN权重/普通权重/所有偏置

    # 遍历模型所有模块（包括嵌套模块）
    for v in model.modules():
        # 收集偏置参数：所有拥有bias属性的模块且bias是Parameter类型
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)  # 偏置项归入第三组

        # 处理权重参数：区分BN层权重和普通权重
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)  # BN层权重归入第一组
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)  # 普通层权重归入第二组

    # 返回参数组配置（权重衰减仅应用于普通权重）
    return [
        {'params': g_bnw},  # 第一组：BN层权重，无weight_decay
        {'params': g_w, 'weight_decay': cfg.hyp.weight_decay},  # 第二组：普通权重，应用衰减
        {'params': g_b}  # 第三组：所有偏置，无weight_decay
    ]

class RepVGGOptimizer(SGD):  # 继承SGD的优化器，结合结构重参数化技术，专用于RepVGG模型训练
    #   scales is a list, scales[i] is a triple (scale_identity.weight, scale_1x1.weight, scale_conv.weight) or a two-tuple (scale_1x1.weight, scale_conv.weight) (if the block has no scale_identity)
    def __init__(self, model, scales,
                 cfg, momentum=0, dampening=0,
                 weight_decay=0, nesterov=True,
                 reinit=True, use_identity_scales_for_reinit=True,
                 cpu_mode=False ,device='cpu' ,params = None
                 ,weight_masks=None):  # 初始化优化器：model为待优化模型，scales为结构重参数化缩放系数列表，cfg为配置对象，reinit控制是否重初始化参数，weight_masks用于剪枝的权重掩码
        self.device = device
        defaults = dict(lr=cfg.hyp.lr0, momentum=cfg.hyp.momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        # parameters = set_weight_decay(model)
        if params is None:
            parameters = get_optimizer_param(cfg, model)  # 获取待优化参数列表，包含权重与偏置的分组配置
        else:
            parameters = params
        super(SGD, self).__init__(parameters, defaults)  # 调用父类SGD初始化方法
        self.num_layers = len(scales)  # 记录缩放系数层数
        blocks = []
        extract_blocks_into_list(model, blocks)  # 提取模型中的RepVGG块到blocks列表
        convs = [b.conv for b in blocks]  # 获取所有块的卷积层引用

        # 构建参数字典用于参数名映射
        from collections import OrderedDict
        weight2name = OrderedDict()  # 创建权重张量到参数名的有序映射
        for k, v in model.named_parameters():
            weight2name[v] = k
        self.weight2name = weight2name

        # 剪枝模式处理逻辑
        if cfg.Prune.use_sparse:
            self.prune_mode = True  # 启用稀疏剪枝模式
        if weight_masks is not None:  # 存在预定义权重掩码时进入剪枝分支
            self.prune_mode = True
            scales = copy.deepcopy(scales)
            _scales = []
            for scale ,conv in zip(scales ,convs):
                convname = weight2name[conv.weight]
                convname = convname.replace('.weight' ,'').replace('module.' ,'')  # 标准化卷积层名称
                outmask ,inpmask = weight_masks[convname]  # 获取输出/输入通道掩码
                newscale = []
                for s in scale:
                    s = s[outmask]  # 根据输出通道掩码裁剪缩放系数
                    newscale.append(s)
                _scales.append(newscale)
                # 形状校验确保剪枝后参数对齐
                for s in newscale:
                    assert s.shape[0]==len(outmask)
        else:
            self.prune_mode = False
            _scales = copy.deepcopy(scales)
        assert len(_scales) == len(convs)  # 校验缩放系数与卷积层数量一致

        # 参数重初始化逻辑
        if reinit:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):  # 检查所有BN层参数
                    gamma_init = m.weight.mean()  # 计算gamma均值
                    if gamma_init == 1.0:
                        pass
                        # print('Checked. This is training from scratch.')
                    else:
                        print \
                            ('========================== Warning! Is this really training from scratch ? =================')
            print('##################### Re-initialize #############')
            self.reinitialize(_scales, convs, use_identity_scales_for_reinit)  # 执行重初始化操作
        self.generate_gradient_masks(_scales, convs, cpu_mode)  # 生成梯度掩码用于训练优化

    def reinitialize(self, scales_by_idx, conv3x3_by_idx, use_identity_scales):
        """卷积核重参数化方法"""
        # 完全复现原始循环结构
        for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
            # 保留原始变量命名(in_channels/out_channels)
            in_channels = conv3x3.in_channels
            out_channels = conv3x3.out_channels

            # 严格保持原始1x1卷积初始化方式
            kernel_1x1 = nn.Conv2d(in_channels, out_channels, 1).to(conv3x3.weight.device)

            # 完全复现原始条件分支
            if len(scales) == 2:
                # 保持原始公式实现形式
                conv3x3.weight.data = conv3x3.weight * scales[1].view(-1, 1, 1, 1) \
                                      + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[0].view(-1, 1, 1, 1)
            else:
                # 完全保留原始断言和逻辑
                assert len(scales) == 3
                assert in_channels == out_channels
                identity = torch.from_numpy(np.eye(out_channels, dtype=np.float32).reshape(
                    out_channels, out_channels, 1, 1)).to(conv3x3.weight.device)
                conv3x3.weight.data = conv3x3.weight * scales[2].view(-1, 1, 1, 1) + F.pad(kernel_1x1.weight,
                                                                                           [1, 1, 1, 1]) * scales[
                                          1].view(-1, 1, 1, 1)
                if use_identity_scales:
                    identity_scale_weight = scales[0]
                    conv3x3.weight.data += F.pad(identity * identity_scale_weight.view(-1, 1, 1, 1), [1, 1, 1, 1])
                else:
                    conv3x3.weight.data += F.pad(identity, [1, 1, 1, 1])

        def generate_gradient_masks(self, scales_by_idx, conv3x3_by_idx, cpu_mode=False):
            """
            梯度掩码生成器 (用于模型压缩/重参数化场景)

            核心作用：
              为卷积核权重生成梯度掩码，控制反向传播时不同部分的梯度更新强度，
              实现结构化参数调整，常用于模型压缩或卷积核重参数化过程

            关键参数说明：
              - scales_by_idx：缩放系数分组列表，每组包含不同分支的缩放因子
              - conv3x3_by_idx：目标3x3卷积层列表，与scales分组一一对应
              - cpu_mode：设备模式标志（当前实现未显式使用）

            核心逻辑细节：
            1. 掩码构造模式：
               - 2系数模式(scales长度=2)：适用于普通卷积核分解
                 * 基础掩码 = scale[1]^2 的全1矩阵
                 * 中心点增强 = scale[0]^2 的1x1区域叠加
               - 3系数模式(scales长度=3)：支持identity分支的特殊处理
                 * 基础掩码 = scale[2]^2 的全1矩阵
                 * 中心点增强 = scale[1]^2 的1x1区域叠加
                 * 非剪枝模式时，在中心点叠加单位矩阵影响（通道对角线位置+1）

            2. 空间操作特性：
               - 所有操作聚焦于卷积核中心点(位置[1:2,1:2])
               - 使用view(-1,1,1,1)进行广播维度对齐
               - 通过张量切片实现精准位置操作

            3. 特殊处理逻辑：
               - prune_mode=True时跳过identity分支的掩码增强
               - 严格校验输入输出通道相等性(assert)
               - 自动设备管理(device=scales[0].device)

            输出存储：
              grad_mask_map字典：以卷积核权重张量为键，存储对应的梯度掩码
              最终掩码会转移到self.device指定的设备
            """
            self.grad_mask_map = {}
            for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
                para = conv3x3.weight
                if len(scales) == 2:
                    mask = torch.ones_like(para, device=scales[0].device) * (scales[1] * scales[1]).view(-1, 1, 1, 1)
                    mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1, device=scales[0].device) * (
                                scales[0] * scales[0]).view(-1, 1, 1, 1)
                else:
                    mask = torch.ones_like(para, device=scales[0].device) * (scales[2] * scales[2]).view(-1, 1, 1, 1)
                    mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1, device=scales[0].device) * (
                                scales[1] * scales[1]).view(-1, 1, 1, 1)
                    # 剪枝模式下放弃identity分支，否
                    if not self.prune_mode:
                        ids = np.arange(para.shape[1])
                        assert para.shape[1] == para.shape[0]
                        mask[ids, ids, 1:2, 1:2] += 1.0

                self.grad_mask_map[para] = mask.to(self.device)

    def __setstate__(self, state):
        """ 对象反序列化时调用，用于恢复状态（如从pickle加载时）

        该方法先调用父类反序列化逻辑恢复基础状态，然后确保所有参数组中包含'nesterov'配置项。
        若参数组中不存在该配置，则设置默认值False，以兼容旧版本序列化数据缺少该字段的情况。

        Args:
            state (dict): 待恢复的状态字典，包含对象序列化时保存的所有信息
        """
        # 调用父类反序列化方法，确保父类定义的属性正确恢复
        super(SGD, self).__setstate__(state)

        # 遍历所有参数组（每个参数组可独立配置优化器参数）
        for group in self.param_groups:
            # 对每个参数组设置'nesterov'参数的默认值False。
            # 若参数组已存在该配置，则保留原值；若不存在（如旧版本序列化的对象），则添加默认配置。
            # 此处理保证加载旧版本优化器状态时不会因缺少新参数而出错
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """ 执行单步参数更新，包含梯度下降、动量、权重衰减、nesterov动量等逻辑

        Args:
            closure (callable, optional): 用于重新计算损失的闭包函数（如需要中间变量时使用）

        Returns:
            float: 当前计算的损失值，若无闭包返回None

        核心流程：
        1. 通过闭包计算损失（若提供）
        2. 遍历所有参数组，按组配置处理参数更新
        3. 对每个参数应用梯度掩码（如果存在）
        4. 实现权重衰减、动量缓冲、nesterov动量等SGD变种逻辑
        """
        loss = None
        if closure is not None:
            # 执行闭包计算损失，常用于需要中间变量的复杂前向传播
            loss = closure()

        for group in self.param_groups:
            # 从参数组获取优化配置参数
            weight_decay = group['weight_decay']  # L2正则化系数
            momentum = group['momentum']  # 动量系数
            dampening = group['dampening']  # 动量阻尼系数
            nesterov = group['nesterov']  # Nesterov动量开关

            for p in group['params']:
                # 跳过无梯度的参数（可能未参与计算图的参数）
                if p.grad is None:
                    continue

                # 应用梯度掩码（自定义逻辑，用于特定参数梯度过滤）
                if p in self.grad_mask_map:
                    d_p = p.grad.data * self.grad_mask_map[p]  # 按掩码过滤梯度
                else:
                    d_p = p.grad.data  # 原始梯度

                # 权重衰减（L2正则化项）
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)  # d_p += weight_decay * p.data

                # 动量处理
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        # 初始化动量缓冲区（首次更新时创建）
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        # 更新动量缓冲区：buf = buf * momentum + d_p * (1 - dampening)
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=(1 - dampening))

                    # Nesterov动量调整：实际使用 buf + momentum * buf
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf  # 普通动量直接使用缓冲区

                # 参数更新：param = param - lr * d_p
                # TODO 此处使用p.data直接更新，而非p.grad.data，需确认是否符合预期
                p.data.add_(d_p, alpha=-group['lr'])

        return loss
