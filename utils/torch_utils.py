import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    分布式训练同步上下文管理器，确保主进程优先执行操作

    功能:
    - 非主进程(local_rank≠0/-1)在yield前等待
    - 主进程(local_rank=0)在yield后同步
    关键细节:
    - 使用dist.barrier实现进程阻塞
    - device_ids指定同步使用的设备
    - 上下文管理器适用于数据预处理等需单进程操作场景
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def date_modified(path=__file__):
    """返回文件最后修改日期，格式'年-月-日' (例: '2023-5-04')"""
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):
    """
    执行git describe命令获取版本描述

    参数:
    - path: git仓库路径，须为目录
    返回:
    - 版本字符串 (例: v5.0-5-g3e25f1e)
    - 非git仓库时返回空字符串
    注意:
    - 使用subprocess执行shell命令
    - [:-1]用于去除输出结尾换行符
    - 异常处理避免非git环境崩溃
    """
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''


def select_device(device=''):
    """
    GPU设备选择函数
    参数:
    - device: 可选参数，接受'cpu'或单个GPU编号如'0'
    返回:
    - torch.device对象
    """
    device = str(device).strip().lower().replace('cuda:', '')  # 格式化输入
    cpu = device == 'cpu'

    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制CPU模式
    else:
        # 自动选择第一个可用GPU
        if device:
            selected_device = device.split(',')[0]  # 取第一个设备编号
        else:
            selected_device = '0'  # 默认选择GPU 0
        os.environ['CUDA_VISIBLE_DEVICES'] = selected_device
        assert torch.cuda.is_available(), f'CUDA设备 {selected_device} 不可用'

    return torch.device(
        'cuda:0' if not cpu and torch.cuda.is_available() else 'cpu')


def time_sync():
    """
    获取精确的跨平台计时时间

    功能:
    - 同步CUDA流后返回时间戳
    - 确保CUDA操作全部完成再计时

    返回:
    - float类型的时间戳

    注意:
    - 在GPU训练时使用cuda.synchronize()保证计时准确性
    - 比time.time()更适合测量PyTorch操作耗时
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_parallel(model):
    """
    检查模型是否为并行模型 (DP或DDP)

    返回:
    - bool: True表示模型是DataParallel或DistributedDataParallel实例
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    """
    解除模型并行化包装

    返回:
    - 单GPU原始模型
    说明:
    - 通过访问model.module属性去除并行包装
    - 对非并行模型直接返回原模型
    """
    return model.module if is_parallel(model) else model


def intersect_dicts(da, db, exclude=()):
    """
    字典键值交叉比对过滤器

    功能:
    - 筛选da中同时存在于db且形状相同的键值对
    - 排除键名包含exclude字符串的项

    典型应用:
    - 模型加载时过滤不匹配的预训练权重

    返回:
    - 新字典，使用da的值但满足db的键名和形状要求
    """
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    """
    初始化神经网络模块参数

    初始化策略:
    - Conv2d: 保留默认初始化 (Kaiming初始化被注释)
    - BatchNorm2d: 设置eps=1e-3, momentum=0.03
    - 激活函数: 统一启用inplace操作节省内存

    作用:
    - 稳定训练初期收敛
    - 统一批归一化层超参数
    - 优化显存利用率
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # 可取消注释使用nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    """
    融合卷积层与批量归一化层为单个卷积层

    算法步骤:
    1. 创建带偏置的新卷积层(原始卷积可能无偏置)
    2. 融合权重矩阵: W_fused = (γ / sqrt(σ^2 + ε)) * W_conv
    3. 融合偏置项: b_fused = (γ / sqrt(σ^2 + ε)) * b_conv + (β - γμ / sqrt(σ^2 + ε))

    参数:
    - conv: nn.Conv2d 实例
    - bn: nn.BatchNorm2d 实例

    返回:
    - fusedconv: 融合后的卷积层，保留在原设备上

    注意:
    - 通过数学等价转换减少层间计算量，提升推理速度
    - 输出层requires_grad=False 保持融合后参数冻结
    - 对部署模型优化有显著效果
    """
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 权重融合: 将BN的缩放因子整合到卷积权重中
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 偏置融合: 调整卷积偏置并加入BN的偏移量
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    """
    模型分析与统计函数

    功能:
    - 计算总参数量(n_p)与可训练参数量(n_g)
    - 输出详细层信息(当verbose=True时)
    - 估算模型GFLOPS(Float32运算量)
    - 生成标准化的模型摘要日志

    参数:
    - img_size: 支持int或list格式输入，用于计算FLOPS缩放
    - verbose: 是否打印各层详细参数分布

    处理流程:
    1. 参数统计使用numel()遍历所有张量
    2. FLOPs估算基于thop库，使用指定尺寸的虚拟输入
    3. 自动识别YOLO模型版本并美化输出名称
    4. Windows系统日志处理特殊字符
    """
    n_p = sum(x.numel() for x in model.parameters())  # 总参数量
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # 需梯度参数量
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs计算可能因缺少thop库失败
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # 兼容不同模型定义
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride基数GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # 尺寸格式标准化
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 按实际尺寸缩放
    except (ImportError, Exception):
        fs = ''

    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def copy_attr(a, b, include=(), exclude=()):
    """
    对象属性复制器，支持包含/排除规则

    功能:
    - 将源对象b的属性复制到目标对象a
    - 自动跳过以_开头的私有属性
    - 通过include/exclude参数控制复制范围

    参数:
    - a: 目标对象(接收属性)
    - b: 源对象(提供属性)
    - include: 只复制包含在此列表中的属性(空列表表示全部)
    - exclude: 始终排除的属性(优先级高于include)

    关键逻辑:
    1. 当include非空时，仅复制include列出的属性
    2. 无论include如何设置，排除exclude指定的属性
    3. 始终跳过Python约定俗成的私有属性(以_开头)
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """
    模型指数移动平均(EMA)更新器

    功能:
    - 维护模型参数的指数移动平均版本，提升训练稳定性
    - 自动处理并行化模型包装(DDP/DP)
    - 支持衰减率热更新(热身阶段渐进式调整)

    关键特性:
    1. 初始化时创建模型深拷贝保持原始结构
    2. 所有EMA参数冻结梯度计算
    3. 支持属性同步(排除分布式训练相关属性)
    4. 采用动态衰减率调整策略(前2000次迭代逐步上升)
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        初始化EMA模型

        参数:
        - model: 原始模型(支持并行化包装)
        - decay: 基础衰减率(实际衰减率随更新次数变化)
        - updates: 初始更新计数(通常保持0)

        注意:
        - 强制EMA模型为eval模式且参数不更新梯度
        - 自动解除并行化包装获取原始模型
        """
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # 解除并行化并冻结
        self.updates = updates  # 更新计数器
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 动态衰减计算函数
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        执行EMA参数更新

        流程:
        1. 更新计数器+1
        2. 计算当前动态衰减率
        3. 遍历所有浮点类型参数:
           - EMA参数 = decay * EMA参数 + (1 - decay) * 当前参数
        4. 自动处理并行化模型的状态字典

        注意:
        - 使用detach()切断计算图节省显存
        - 仅更新浮点数参数(跳过整数型buffer)
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        同步模型属性(如__dict__中的超参数)

        参数:
        - include: 指定需要同步的属性白名单
        - exclude: 强制排除的属性(默认过滤分布式训练相关)

        典型用途:
        - 同步模型版本号、优化器状态等元数据
        - 保持EMA模型与原始模型的配置一致性
        """
        copy_attr(self.ema, model, include, exclude)


class SemiSupModelEMA:
    """
    半监督训练专用模型指数移动平均(EMA)更新器

    功能改进:
    - 简化动态衰减机制，采用固定衰减率(对比原ModelEMA)
    - 专为半监督场景设计，教师模型参数更新更稳定

    核心变化:
    1. 移除衰减率的热身阶段(2000次迭代的渐进式调整)
    2. 直接使用固定decay值进行参数混合
    3. 更适用于师生模型中的缓慢更新策略

    继承特性:
    - 自动处理并行化模型包装
    - 冻结EMA模型梯度
    - 支持属性同步排除分布式训练相关字段
    """

    def __init__(self, model, decay=0.99, updates=0):
        """
        初始化半监督EMA

        参数调整:
        - decay: 固定衰减率(原版为动态计算)
        - 移除lambda函数，直接存储标量衰减值

        设计考量:
        - 半监督训练通常需要更保守的参数更新策略
        - 消除动态衰减对师生模型协同训练的潜在干扰
        """
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # 稳定初始化
        self.updates = updates
        self.decay = decay  # 固定衰减率
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        执行固定衰减率的EMA更新

        算法变化:
        - 计算公式简化为: EMA = decay*EMA + (1-decay)*model
        - 相比原版移除指数衰减的预热机制

        优势:
        - 更新步长恒定，提高师生模型协同稳定性
        - 避免训练初期因衰减率变化导致的震荡
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay  # 直接使用固定值

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        同步非参数属性的专用方法

        特殊设计:
        - 默认排除分布式训练相关字段，防止EMA模型污染
        - 支持白名单(include)和黑名单(exclude)双向控制

        典型同步项:
        - 模型版本号
        - 输入归一化统计量
        - 课程学习相关参数
        """
        copy_attr(self.ema, model, include, exclude)


class CosineEMA:
    """
    余弦衰减EMA模型参数更新器

    核心改进:
    - 采用余弦退火策略调整衰减率，范围[decay_start, decay_end]
    - 适用于需要渐进式调整参数更新强度的训练场景

    衰减率公式:
    decay = decay_end - 0.5*(decay_end - decay_start)*(cos(π*cur_epoch/total_epoch)+1)
    实现从decay_start到decay_end的平滑过渡
    """

    def __init__(self, model, decay_start=0.99, decay_end=0.9999, total_epoch=0):
        """
        初始化余弦EMA

        参数新增:
        - decay_start: 初始衰减率(通常较小)
        - decay_end: 终止衰减率(通常接近1)
        - total_epoch: 总训练轮数，用于计算衰减进度

        设计特点:
        - 保留模型并行化处理逻辑
        - 初始化时强制EMA模型为评估模式
        """
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # 模型克隆
        self.total_epoch = total_epoch  # 控制余弦周期长度
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.decay = decay_start  # 当前衰减率
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0  # 保留字段供扩展使用

    def update(self, model):
        """
        执行EMA参数更新

        与基础版差异:
        - 使用通过update_decay计算的最新衰减率
        - 更新逻辑保持标准EMA公式
        """
        with torch.no_grad():
            d = self.decay  # 获取当前衰减率
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_decay(self, cur_epoch):
        """
        基于余弦曲线更新衰减率

        数学实现:
        - 将当前epoch映射到余弦函数的[0, π]区间
        - 通过振幅调整实现衰减率区间变换
        - 随着训练进行，衰减率单调递增至decay_end

        训练策略:
        - 通常在每个epoch结束时调用
        - 与学习率调度器协同工作
        """
        self.decay = self.decay_end - (self.decay_end - self.decay_start) * (
                    np.cos(np.pi * cur_epoch / self.total_epoch) + 1) / 2

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        继承自基类的属性同步方法

        特殊处理:
        - 默认排除分布式训练相关属性
        - 同步模型版本号等元数据
        """
        copy_attr(self.ema, model, include, exclude)