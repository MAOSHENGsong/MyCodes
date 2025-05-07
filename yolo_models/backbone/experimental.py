import torch
from torch import nn


class Ensemble(nn.ModuleList):
    """模型集成容器，支持多种集成策略

    功能特性：
    - 继承自nn.ModuleList，可存储多个子模型
    - 支持最大集成(max)、平均集成(mean)、NMS集成三种策略
    - 兼容单模型推理与多模型集成推理

    初始化参数：无
    """

    def __init__(self):
        super().__init__()  # 继承ModuleList的初始化方法

    def forward(self, x, augment=False, profile=False, visualize=False):
        """前向传播集成逻辑

        参数：
        x : 输入张量，形状[batch, channels, height, width]
        augment : 是否使用测试时数据增强（TTA）
        profile : 是否输出层耗时分析
        visualize : 是否保存特征可视化（当前版本未启用）

        返回：
        y : 集成后的预测结果
        None : 占位符（保持与训练输出格式兼容）
        """
        y = []  # 存储各模型的输出

        # 遍历所有子模型进行推理
        for module in self:
            # 获取单个模型输出（假设输出为元组，首元素为预测结果）
            # 原代码中visualize参数被注释，可能与版本兼容有关
            output = module(x, augment, profile)[0]  # 取索引0的预测结果
            y.append(output)

        # 集成策略选择 ----------------------------------------------------
        # y = torch.stack(y).max(0)[0]  # 最大集成：取各模型预测最大值
        # y = torch.stack(y).mean(0)    # 平均集成：取各模型预测均值
        y = torch.cat(y, 1)  # NMS集成：沿维度1拼接，需后续NMS处理

        # 调试信息输出（正式使用时建议注释）
        print(y)  # 打印中间结果，用于验证集成效果

        return y, None  # 保持(train_output, inference_output)格式兼容

def attempt_load(weights, device=None, inplace=True, fuse=True):
    """加载模型权重文件，支持单模型或集成模型加载

    功能特性：
    - 自动处理EMA模型权重与普通模型权重
    - 执行模型兼容性调整（适配不同PyTorch版本）
    - 支持模型层融合（fuse）优化推理速度
    - 可加载多个模型组成集成模型（Ensemble）

    参数：
    weights : str/list 权重文件路径，支持单个文件或多个文件的列表
    device : torch.device, 指定加载设备，默认自动选择
    inplace : bool, 是否替换激活函数的inplace操作（True提升内存效率）
    fuse : bool, 是否融合Conv+BN+Activation层（提升推理速度）

    返回：
    加载完成的模型对象（单模型或集成模型）
    """

    from yolo_models.detector.yolo import Detect, Model  # 延迟导入防止循环依赖

    model = Ensemble()  # 初始化集成模型容器

    # 权重加载循环 ---------------------------------------------------------
    for w in weights if isinstance(weights, list) else [weights]:
        # 1. 下载并加载权重文件
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # 强制加载到CPU

        # 2. 选择EMA模型或普通模型
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32转换

        # 3. 模型兼容性修补
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])  # 默认stride值
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # 类别名列表转字典

        # 4. 层融合与评估模式切换
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())

    # 模型层兼容性处理 -----------------------------------------------------
    for m in model.modules():
        t = type(m)
        # 激活函数inplace设置
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # 统一inplace参数
            # Detect层anchor_grid格式修正
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)  # 按检测层数初始化

        # Upsample层兼容性处理（PyTorch 1.11+）
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

            # 返回处理 ------------------------------------------------------------
    if len(model) == 1:  # 单模型直接返回
        return model[-1]

    # 集成模型元数据统一
    print(f'Ensemble created with {weights}\n')
    # 同步类别信息
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    # 计算最大stride
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model]))].stride
    # 验证类别一致性
    assert all(model[0].nc == m.nc for m in model), f'模型类别数不一致: {[m.nc for m in model]}'

    return model