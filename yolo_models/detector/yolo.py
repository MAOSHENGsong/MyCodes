import logging

import torch
from torch import nn

from utils.torch_utils import initialize_weights, fuse_conv_and_bn, copy_attr, model_info
from yolo_models.backbone import build_backbone, Conv
from yolo_models.backbone.common import AutoShape
from yolo_models.head import build_head
from yolo_models.neck import build_neck

LOGGER = logging.getLogger(__name__)

class Model(nn.Module):
    """
    主要作用：YOLO模型的顶层容器，整合骨干网络、颈部网络和检测头
    核心功能：
        - 动态解析配置文件构建网络结构
        - 管理模型各组件间的数据流
        - 提供标准化的权重初始化
        - 支持多种部署优化策略
    参数说明：
        cfg: 配置来源（支持文件路径或配置字典）
            文件路径示例：'models/yolov11.yaml'
            配置字典示例：包含backbone/neck/head定义的字典
    """

    def __init__(self, cfg='yolov11.yaml'):
        super().__init__()
        self.cfg = cfg  # 保存原始配置引用

        # 网络架构构建（核心组件）
        self.backbone = build_backbone(cfg)  # 骨干网络（特征提取）
        self.neck = build_neck(cfg)          # 颈部网络（特征融合）
        self.head = build_head(cfg)          # 检测头（预测输出）

        # 元数据配置
        self.names = cfg.Dataset.names  # 类别名称列表（用于可视化）
        self.inplace = cfg.Model.inplace  # 是否使用原地操作优化内存

        # 结构验证与参数初始化
        self.check_head()  # 验证head与neck的通道匹配
        initialize_weights(self)  # 自适应权重初始化
        self.info()  # 打印模型概况
        LOGGER.info('')  # 输出空行分隔日志

    def check_head(self):
        """ YOLOv11检测头专项配置检查与初始化

        功能说明:
        - 计算特征图下采样步长(stride)
        - 初始化检测层偏置参数

        流程说明:
        1. 获取检测头实例并标记模型类型
        2. 配置原地操作(inplace)参数
        3. 计算特征图下采样步长
        4. 执行偏置参数初始化
        """
        m = self.head  # 获取YOLOv11Detect检测头实例
        self.model_type = 'yolov11'  # 明确标记模型类型

        # 配置原地操作优化内存使用
        m.inplace = self.inplace

        # 动态计算各检测层步长
        s = 640  # 基准输入尺寸（建议使用训练时标准输入尺寸）
        with torch.no_grad():
            # 生成虚拟输入: (batch, channels, height, width)
            test_input = torch.zeros(1, self.cfg.Model.ch, s, s)
            # 获取各层输出特征图的尺寸 [1, C, H, W]
            feature_shapes = [x.shape for x in self.forward(test_input)]
            # 计算实际步长 = 输入尺寸 / 特征图尺寸
            self.stride = torch.Tensor([s / x.shape[-2] for x in feature_shapes])

        # 初始化检测层偏置参数（YOLOv11专用初始化逻辑）
        m.initialize_biases(self.stride)  # 传入步长参数用于自适应初始化


    def forward(self, x):
        """
        数据流处理流程：
        Backbone → Neck → Head
        典型处理流程：
        x → backbone → 多尺度特征图 → neck → 增强特征 → head → 预测输出
        """
        # 具体实现通过各组件的前向传播串联
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)

    # def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
    #     LOGGER.info('Fusing layers... ')
    #     for m in self.backbone.modules():
    #         if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
    #             m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    #             delattr(m, 'bn')  # remove batchnorm
    #             m.forward = m.forward_fuse  # update forward
    #
    #     for m in self.neck.modules():
    #         if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
    #             m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    #             delattr(m, 'bn')  # remove batchnorm
    #             m.forward = m.forward_fuse  # update forward
    #
    #     for layer in self.backbone.modules():
    #         if isinstance(layer, QARepVGGBlock):
    #             layer.switch_to_deploy()
    #         if isinstance(layer, RepVGGBlock):
    #             layer.switch_to_deploy()
    #         if isinstance(layer, RepConv):
    #             layer.fuse_repvgg_block()
    #         if hasattr(layer, 'reparameterize'):
    #             layer.reparameterize()
    #     for layer in self.neck.modules():
    #         if isinstance(layer, QARepVGGBlock):
    #             layer.switch_to_deploy()
    #         if isinstance(layer, RepVGGBlock):
    #             layer.switch_to_deploy()
    #         if isinstance(layer, RepConv):
    #             layer.fuse_repvgg_block()
    #         if hasattr(layer, 'reparameterize'):
    #             layer.reparameterize()
    #     self.info()
    #     return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    @property
    def device(self):
        """ 自动获取模型所在设备（CPU/GPU） """
        return next(self.parameters()).device

    def _apply(self, fn):
        """ 重写_apply方法实现部署优化 """
        # 维持inplace操作的设备一致性
        super()._apply(fn)
        self.backbone.inplace = self.inplace
        return self

    def info(self, verbose=False,img_size=416):
        """ 输出模型结构信息 """
        model_info(self, verbose, img_size)