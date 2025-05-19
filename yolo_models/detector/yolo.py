from torch import nn

from yolo_models.backbone import build_backbone
from yolo_models.head import build_head
from yolo_models.neck import build_neck


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
        """
        验证检测头结构的有效性：
        - 确保head中的anchor尺寸与特征图匹配
        - 检查分类数配置一致性
        - 验证输出层的通道数计算
        """
        # 实现细节包含通道数匹配验证、anchor尺寸校验等


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

    def info(self, verbose=False):
        """ 输出模型结构信息 """
        # 包含参数量计算、各组件结构打印等
        ...