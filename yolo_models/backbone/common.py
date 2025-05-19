import warnings

import torch
import torch.nn as nn

from configs.yacs import _assert_with_logging


def get_activation(act=True):
    """
    # 主要作用：根据输入参数动态获取激活函数模块及名称
    # 重点：
    #   - 支持字符串标识符和模块实例两种输入形式
    #   - 内置silu/relu/hard_swish/lrelu四种预设激活函数
    #   - 自动处理布尔值输入（True默认返回SiLU）
    #   - 返回激活函数实例及其标准化名称（部分类型）
    # 注意：
    #   - 未识别的字符串类型将抛出AttributeError
    #   - 直接传入nn.Module实例时不设置act_name
    """

    act_name = None
    if isinstance(act, str):  # 字符串标识符处理分支
        if act == "silu":
            m = nn.SiLU()
        elif act == "relu":
            m = nn.ReLU(inplace=True)
            act_name = 'relu'  # 显式记录标准化名称
        elif act == "hard_swish":
            m = nn.Hardswish(inplace=True)
            act_name = 'hard_swish'
        elif act == "lrelu":
            m = nn.LeakyReLU(0.1, inplace=True)
            act_name = 'leaky_relu'
        else:
            raise AttributeError("Unsupported act type: {}".format(act))
    else:  # 非字符串处理分支
        # 布尔值处理：True->SiLU, False->Identity
        m = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity()
        )
    return m, act_name


def autopad(k, p=None):  # kernel, padding
    """
    # 主要作用：自动计算卷积核的填充大小以实现'same'填充效果
    # 重点：
    #   - 根据卷积核尺寸k自动计算padding值
    #   - 支持整数和多元（列表/元组）卷积核尺寸
    #   - 默认实现各维度对称填充（k//2）
    # 注意：
    #   - 当k为奇数时，实际会产生非对称填充（左小右大）
    #   - 适用于大部分常见卷积核尺寸的自动填充需求
    #   - 需确保输入k为整数或可迭代的数字序列
    """
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 多维核自动逐维计算
    return p


class Conv(nn.Module):
    """
    # 主要作用：实现标准卷积模块（卷积+BN+激活函数）
    # 重点：
    #   - 集成卷积层、批归一化、激活函数三合一结构
    #   - 支持自动填充计算（autopad）
    #   - 提供常规前向传播与融合前向传播两种模式
    # 参数说明：
    #   c1: 输入通道数  c2: 输出通道数
    #   k : 卷积核尺寸（默认1） s: 步长（默认1）
    #   p : 填充值（None时自动计算） g: 分组卷积数（默认1）
    #   act: 激活函数类型（支持str/bool/nn.Module）
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 卷积层（自动计算padding，禁用偏置）
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(
                k, p), groups=g, bias=False)
        # 批归一化层
        self.bn = nn.BatchNorm2d(c2)
        # 动态获取激活函数及名称
        self.act, self.act_name = get_activation(act=act)
        self.init_weights()  # 初始化权重

    def forward(self, x):
        """标准前向传播：Conv -> BN -> Activation"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合模式前向传播：Conv -> Activation（跳过BN层）"""
        return self.act(self.conv(x))

    def init_weights(self):
        if self.act_name == "relu":
            # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity="relu")
            pass


class Bottleneck(nn.Module):
    # 主要作用：实现标准残差瓶颈模块（两层级联卷积+可选残差连接）
    # 参数说明：
    #   c1: 输入通道  c2: 输出通道  shortcut: 是否启用残差连接
    #   g : 分组卷积数  k: 卷积核尺寸元组  e: 通道扩展系数  act: 激活函数类型
    # 结构特点：
    #   - 通道压缩扩展结构（通过扩展系数e控制隐藏层维度）
    #   - 支持不同尺寸卷积核组合（默认1x1+3x3）
    #   - 自动通道数匹配的残差连接（当c1 == c2时）

    def __init__(self, c1, c2, shortcut=True, g=1, k=(1, 3), e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)  # 计算压缩后的通道数（通常e=0.5实现通道减半）
        self.cv1 = Conv(c1, c_, k[0], 1, act=act)  # 1x1降维卷积
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, act=act)  # 3x3空间卷积
        # 残差连接条件：启用shortcut且通道数匹配（保证维度一致）
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 残差连接逻辑：原始输入与特征处理路径相加
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # 当add=False时退化为普通串联结构


class BottleneckCSP(nn.Module):
    # 主要作用：实现跨阶段部分连接（Cross Stage Partial）瓶颈结构
    # 设计优势：
    #   - 通过特征分割减少计算量
    #   - 增强梯度流动
    #   - 常用于YOLO等高效检测模型
    # 参数扩展：
    #   n : 瓶颈层重复次数  e: 通道压缩系数
    # 核心结构：
    #   - 双路径处理：主路径（多瓶颈层） + 直连路径
    #   - 特征拼接后融合

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)  # 最终输出通道为c2，隐藏层通道为c_*2
        # 路径1：特征变换主路径
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        # 路径2：直连路径（不经过瓶颈层）
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        # 瓶颈层序列（n个Bottleneck堆叠）
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0, act=True) for _ in range(n)])
        # 特征融合组件
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1, act=act)
        # 归一化与激活
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 主路径处理：cv1 -> 瓶颈层序列 -> cv3
        y1 = self.cv3(self.m(self.cv1(x)))
        # 直连路径处理：cv2
        y2 = self.cv2(x)
        # 特征拼接与融合
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class C3(nn.Module):
    """
    主要作用：改进型CSP瓶颈结构，采用三卷积设计实现跨阶段特征融合
    核心改进：
        - 相比标准CSP结构增加第三条卷积路径
        - 支持多阶段激活函数配置
        - 更灵活的特征组合方式
    参数说明：
        c1: 输入通道数  c2: 输出通道数  n: 瓶颈层重复次数
        shortcut: 是否在Bottleneck内部启用残差连接（默认True）
        g: 分组卷积数（默认1）  e: 通道扩展系数（默认0.5）
        act: 激活函数配置（支持组合格式如'relu_silu'）
    结构特性：
        ┌─────────┬───────────────┐
        │ 主路径   │ 辅助路径       │
        │ (cv1→m) │ (cv2)         │
        └───┬─────┴──────┬────────┘
            │ 特征拼接     │
            └───→ cv3 ←──┘
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)  # 计算压缩后的基准通道数

        # 激活函数分解逻辑（支持两阶段激活配置）
        if isinstance(act, str):
            act_configs = {
                'relu_silu': ('relu', 'silu'),
                'relu_lrelu': ('relu', 'lrelu'),
                'relu_hswish': ('relu', 'hard_swish')
            }
            first_act, last_act = act_configs.get(act, (act, act))
        else:
            first_act = last_act = act

        # 路径定义
        self.cv1 = Conv(c1, c_, 1, 1, act=first_act)  # 主路径入口卷积
        self.cv2 = Conv(c1, c_, 1, 1, act=first_act)  # 辅助路径卷积
        self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # 融合输出卷积
        self.m = nn.Sequential(  # 特征处理序列
            *[Bottleneck(c_, c_, shortcut, g, e=1.0, act=first_act) for _ in range(n)]
        )

    def forward(self, x):
        """ 特征融合流程：主路径处理 + 辅助路径直连 → 拼接 → 融合 """
        return self.cv3(torch.cat(
            (self.m(self.cv1(x)),  # 主路径：深度特征提取
             self.cv2(x)),  # 辅助路径：浅层特征保留
            dim=1
        ))


class C2f(nn.Module):
    """
    主要作用：轻量级CSP结构，采用双卷积+动态特征扩展设计
    设计优势：
        - 通过特征分割减少计算冗余
        - 动态扩展特征通道实现多尺度融合
        - 相比C3结构参数更少，适合移动端部署
    参数特性：
        shortcut: 强制设为False（结构特性限制）
        e: 控制初始通道分割比例（默认0.5）
    结构流程：
        Split → [Base] → [Bottleneck×n] → Concat → Conv
        ┌───────────────┐
        │ 输入特征分割    │
        │   → 基准通道    │
        │   → 处理通道 → 逐级扩展
        └───────┬───────┘
                │ 拼接所有通道
                └──→ 输出卷积
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu'):
        super().__init__()
        if act not in ['relu', 'silu', 'lrelu', 'hard_swish']:
            raise ValueError(f"Unsupported activation: {act}")

        self.c = int(c2 * e)  # 基准通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=act)  # 通道翻倍卷积
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=act)  # 动态通道融合卷积
        self.m = nn.ModuleList(  # 可扩展的瓶颈层序列
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act)
            for _ in range(n)
        )

    def forward(self, x):
        """ 动态特征扩展流程 """
        # 初始分割：[基准通道, 处理通道]
        y = list(self.cv1(x).split((self.c, self.c), 1))

        # 逐级扩展处理通道
        for m in self.m:
            y.append(m(y[-1]))  # 每次处理最后一个特征块并追加

        # 全通道融合
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """
    主要作用：改进型CSP瓶颈结构，采用三卷积设计实现跨阶段特征融合
    核心改进：
        - 相比标准CSP结构增加第三条卷积路径
        - 支持多阶段激活函数配置
        - 更灵活的特征组合方式
    参数说明：
        c1: 输入通道数  c2: 输出通道数  n: 瓶颈层重复次数
        shortcut: 是否在Bottleneck内部启用残差连接（默认True）
        g: 分组卷积数（默认1）  e: 通道扩展系数（默认0.5）
        act: 激活函数配置（支持组合格式如'relu_silu'）
    结构特性：
        ┌─────────┬───────────────┐
        │ 主路径   │ 辅助路径       │
        │ (cv1→m) │ (cv2)         │
        └───┬─────┴──────┬────────┘
            │ 特征拼接     │
            └───→ cv3 ←──┘
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)  # 计算压缩后的基准通道数

        # 激活函数分解逻辑（支持两阶段激活配置）
        if isinstance(act, str):
            act_configs = {
                'relu_silu': ('relu', 'silu'),
                'relu_lrelu': ('relu', 'lrelu'),
                'relu_hswish': ('relu', 'hard_swish')
            }
            first_act, last_act = act_configs.get(act, (act, act))
        else:
            first_act = last_act = act

        # 路径定义
        self.cv1 = Conv(c1, c_, 1, 1, act=first_act)  # 主路径入口卷积
        self.cv2 = Conv(c1, c_, 1, 1, act=first_act)  # 辅助路径卷积
        self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # 融合输出卷积
        self.m = nn.Sequential(  # 特征处理序列
            *[Bottleneck(c_, c_, shortcut, g, e=1.0, act=first_act) for _ in range(n)]
        )

    def forward(self, x):
        """ 特征融合流程：主路径处理 + 辅助路径直连 → 拼接 → 融合 """
        return self.cv3(torch.cat(
            (self.m(self.cv1(x)),  # 主路径：深度特征提取
             self.cv2(x)),  # 辅助路径：浅层特征保留
            dim=1
        ))


class C2f(nn.Module):
    """
    主要作用：轻量级CSP结构，采用双卷积+动态特征扩展设计
    设计优势：
        - 通过特征分割减少计算冗余
        - 动态扩展特征通道实现多尺度融合
        - 相比C3结构参数更少，适合移动端部署
    参数特性：
        shortcut: 强制设为False（结构特性限制）
        e: 控制初始通道分割比例（默认0.5）
    结构流程：
        Split → [Base] → [Bottleneck×n] → Concat → Conv
        ┌───────────────┐
        │ 输入特征分割    │
        │   → 基准通道    │
        │   → 处理通道 → 逐级扩展
        └───────┬───────┘
                │ 拼接所有通道
                └──→ 输出卷积
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu'):
        super().__init__()
        if act not in ['relu', 'silu', 'lrelu', 'hard_swish']:
            raise ValueError(f"Unsupported activation: {act}")

        self.c = int(c2 * e)  # 基准通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=act)  # 通道翻倍卷积
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=act)  # 动态通道融合卷积
        self.m = nn.ModuleList(  # 可扩展的瓶颈层序列
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act)
            for _ in range(n)
        )

    def forward(self, x):
        """ 动态特征扩展流程 """
        # 初始分割：[基准通道, 处理通道]
        y = list(self.cv1(x).split((self.c, self.c), 1))

        # 逐级扩展处理通道
        for m in self.m:
            y.append(m(y[-1]))  # 每次处理最后一个特征块并追加

        # 全通道融合
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    主要作用：高效空间金字塔池化层（Spatial Pyramid Pooling Fast）
    设计亮点：
        - 通过重复相同池化核实现多尺度特征提取（原始SPP使用不同尺寸池化核）
        - 计算速度比标准SPP快约2倍（YOLOv5实测数据）
        - 保持与SPP相似的特征表达能力
    参数说明：
        c1: 输入通道数  c2: 输出通道数
        k: 最大池化核尺寸（默认5，等效SPP(k=[5,9,13])）
        act: 激活函数配置（支持组合激活如'relu_silu'）
    结构流程：
        Conv(c1->c_) → MaxPool(k) → 串联三次池化结果 → Conv(4*c_->c2)
        ┌───────────────┐
        │ 输入特征        │
        │   → 降维卷积    │
        │   → 池化序列    │
        │   → 特征拼接    │
        │   → 升维卷积    │
        └───────────────┘
    """

    def __init__(self, c1, c2, k=5, act=True):
        super().__init__()
        c_ = c1 // 2  # 隐藏层通道数（典型设计减半计算量）

        # 激活函数分解（支持前后不同激活函数）
        if isinstance(act, str):
            act_configs = {
                'relu_silu': ('relu', 'silu'),
                'relu_lrelu': ('relu', 'lrelu'),
                'relu_hswish': ('relu', 'hard_swish')
            }
            first_act, last_act = act_configs.get(act, (act, act))
        else:
            first_act = last_act = act

        # 网络组件定义
        self.cv1 = Conv(c1, c_, 1, 1, act=first_act)  # 1x1降维卷积
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=last_act)  # 输出卷积（4倍通道来自拼接）
        self.m = nn.MaxPool2d(
            kernel_size=k,
            stride=1,
            padding=k //
            2)  # 尺寸保持的池化层

        # 池化参数验证
        _assert_with_logging(k % 2 == 1, "SPPF kernel size must be odd")

    def forward(self, x):
        """ 特征处理流程 """
        x = self.cv1(x)  # 降维到c_

        # 级联池化操作（生成多尺度特征）
        with warnings.catch_warnings():
            # 抑制PyTorch 1.9+的池化层冗余警告
            warnings.simplefilter('ignore')
            y1 = self.m(x)  # 第1次池化（k x k）
            y2 = self.m(y1)  # 第2次池化（等效k+2 x k+2）
            y3 = self.m(y2)  # 第3次池化（等效k+4 x k+4）

            # 拼接所有层级特征（原始+三次池化）
            return self.cv2(torch.cat([x, y1, y2, y3], 1))  # 通道维度拼接

    @property
    def equivalent_kernel_sizes(self):
        """ 等效感受野计算（用于理解特征尺度） """
        return [self.m.kernel_size[0] + 2 * i for i in range(4)]  # [5,7,9,11]


class C3k2(C2f):
    """基于C2f的改进型CSP双卷积瓶颈结构（更快实现）

    继承自C2f类，通过可选的C3k模块或标准Bottleneck模块构建，
    保持与父类参数结构一致，增强模块灵活性。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act=True, c3k=False):
        """
        初始化C3k2模块

        参数说明:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): 堆叠的Bottleneck/C3k模块数量
            shortcut (bool): 是否使用跨层连接，默认False（与父类C2f一致）
            g (int): 分组卷积的组数，默认1（普通卷积）
            e (float): 隐藏层通道扩展系数，默认0.5（输出通道的50%作为隐藏层）
            act (bool/str): 激活函数类型，支持bool值或具体激活函数名称
            c3k (bool): 是否使用C3k模块替代标准Bottleneck，默认False
        """
        super().__init__(c1, c2, n, shortcut, g, e, act)

        # 动态选择构建模块：C3k或标准Bottleneck
        self.m = nn.ModuleList(
            C3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=e, act=act) if c3k
            else Bottleneck(
                self.c, self.c,
                shortcut=shortcut,
                g=g,
                k=(3, 3),  # 使用(3,3)统一kernel参数形式
                e=1.0,  # 保持与父类实现一致
                act=act  # 传递激活函数参数
            ) for _ in range(n)
        )


class C3k(C3):
    """支持自定义卷积核的CSP瓶颈模块

    继承自C3类，通过Bottleneck模块的kernel参数实现自定义卷积核尺寸，
    保持与C3模块的参数兼容性，增强特征提取灵活性。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act=True, k=3):
        """
        初始化C3k模块

        参数说明:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): 堆叠的Bottleneck模块数量
            shortcut (bool): 是否使用跨层连接，默认False（与父类C3一致）
            g (int): 分组卷积的组数，默认1
            e (float): 隐藏层通道扩展系数，默认0.5
            act (bool/str): 激活函数类型
            k (int): 主卷积核尺寸，默认3（实际使用(k,k)形式传递）
        """
        super().__init__(c1, c2, n, shortcut, g, e, act)

        # 计算隐藏层通道数（保持与C3相同的计算逻辑）
        c_ = int(c2 * e)

        # 构建序列化的Bottleneck模块
        self.m = nn.Sequential(
            *(Bottleneck(
                c_, c_,
                shortcut=shortcut,
                g=g,
                k=(k, k),  # 使用(k,k)形式统一kernel参数
                e=1.0,  # 保持与C3实现一致
                act=act  # 传递激活函数参数
            ) for _ in range(n)))


class Attention(nn.Module):
    """多头位置敏感注意力机制

    实现基于卷积的轻量化自注意力模块，包含位置编码增强。
    适用于视觉任务中的长程依赖建模与空间特征增强。

    特性:
        - 基于卷积的QKV生成
        - 可配置注意力头数与键维度比率
        - 内置位置编码卷积层
        - 自适应注意力缩放因子

    示例:
        >>> attn = Attention(dim=128, num_heads=4, act=False)
        >>> x = torch.randn(1, 128, 32, 32)
        >>> print(attn(x).shape)
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5, act=False):
        """
        初始化注意力模块

        参数说明:
            dim (int): 输入特征维度
            num_heads (int): 注意力头数，默认8
            attn_ratio (float): 键维度压缩比率，默认0.5
            act (bool/str): 卷积层激活函数，默认关闭
        """
        super().__init__()
        # 基础参数校验
        assert dim % num_heads == 0, "特征维度必须能被注意力头数整除"

        # 注意力参数配置
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 单头特征维度
        self.key_dim = int(self.head_dim * attn_ratio)  # 键值维度
        self.scale = self.key_dim ** -0.5  # 注意力缩放因子

        # 计算QKV总维度
        nh_kd = self.key_dim * num_heads
        total_dim = dim + nh_kd * 2  # Q+K+V总通道数

        # 核心卷积层配置
        self.qkv = Conv(dim, total_dim, 1, act=act)  # QKV生成卷积
        self.proj = Conv(dim, dim, 1, act=act)  # 输出投影卷积
        self.pe = Conv(dim, dim, 3, 1,  # 位置编码卷积
                       g=dim,  # 深度可分离卷积
                       act=act)

    def forward(self, x):
        """
        前向传播过程

        处理流程:
        1. QKV生成与维度切分
        2. 注意力分数计算
        3. 注意力权重归一化
        4. 值矩阵加权融合
        5. 位置编码增强
        6. 最终特征投影

        参数:
            x (Tensor): 输入特征图，形状[B, C, H, W]

        返回:
            (Tensor): 增强后的特征图，形状保持[B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W  # 空间位置总数

        # 生成QKV并切分维度 [B, (Q+K+V), H, W] -> [B, num_heads, key_dim*2+head_dim, N]
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, -1, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        # 注意力分数计算 (Q^T K / sqrt(d_k))
        attn = (q.transpose(2, 3) @ k) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)  # 注意力权重归一化

        # 值矩阵加权融合 (V @ Attn^T)
        x = (v @ attn.transpose(2, 3))  # [B, num_heads, head_dim, N]
        x = x.view(B, C, H, W)  # 恢复空间维度

        # 位置编码增强
        x += self.pe(v.reshape(B, C, H, W))  # 保持维度一致

        # 最终特征投影
        return self.proj(x)

class PSABlock(nn.Module):
    """位置敏感注意力模块

    实现多头注意力机制与前馈网络的组合结构，支持残差连接。
    适用于视觉任务中的特征增强与上下文建模。

    特性:
        - 可配置注意力头数与注意力比率
        - 带激活函数的前馈网络
        - 可选的残差连接机制

    示例:
        >>> psablock = PSABlock(c=128, num_heads=4, shortcut=True, act='silu')
        >>> x = torch.randn(1, 128, 32, 32)
        >>> print(psablock(x).shape)
    """

    def __init__(self, c, num_heads=4, attn_ratio=0.5, shortcut=True, act=True):
        """
        初始化PSABlock

        参数说明:
            c (int): 输入/输出通道数
            num_heads (int): 注意力头数，默认4
            attn_ratio (float): 键值维度压缩比率，默认0.5
            shortcut (bool): 是否启用残差连接，默认True
            act (bool/str): 激活函数类型，支持'silu'等，默认True
        """
        super().__init__()

        # 注意力模块
        self.attn = Attention(
            dim=c,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            act=act  # 传递激活函数参数
        )

        # 前馈网络（带激活函数）
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1, act=act),  # 扩展维度
            Conv(c * 2, c, 1, act=False)  # 恢复维度
        )

        # 残差连接开关
        self.add = shortcut

    def forward(self, x):
        """
        前向传播过程

        处理流程:
        1. 注意力分支处理（可选残差）
        2. 前馈网络处理（可选残差）

        参数:
            x (Tensor): 输入特征图，形状[B, C, H, W]

        返回:
            (Tensor): 处理后的特征图，形状保持[B, C, H, W]
        """
        # 注意力处理分支
        x = x + self.attn(x) if self.add else self.attn(x)

        # 前馈处理分支
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):
    """带注意力机制的C2PSA特征增强模块

    通过堆叠PSABlock实现通道注意力机制，增强特征提取能力。
    继承自nn.Module，重构原PSA模块以支持多模块堆叠。

    特性:
        - 通道分割与重组机制
        - 可配置注意力模块堆叠次数
        - 兼容标准卷积参数结构

    示例:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5, act='silu')
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5, act=True):
        """
        初始化C2PSA模块

        参数说明:
            c1 (int): 输入通道数
            c2 (int): 输出通道数（需等于c1）
            n (int): PSABlock堆叠次数，默认1
            e (float): 隐藏层扩展系数，默认0.5
            act (bool/str): 激活函数类型，支持'silu'等，默认True
        """
        super().__init__()
        assert c1 == c2, "输入输出通道必须相等"
        self.c = int(c1 * e)  # 计算隐藏层通道数

        # 通道分割与重组卷积
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=act)  # 输入通道分割
        self.cv2 = Conv(2 * self.c, c1, 1, act=act)  # 输出通道合并

        # 注意力模块堆叠
        self.m = nn.Sequential(
            *(PSABlock(
                self.c,
                attn_ratio=0.5,  # 注意力比率保持默认
                num_heads=self.c // 64,  # 自动计算注意力头数
                act=act  # 传递激活函数参数
            ) for _ in range(n))
        )

    def forward(self, x):
        """
        前向传播过程

        处理流程:
        1. 通道分割为a/b两支
        2. b支进行注意力处理
        3. 通道合并输出

        参数:
            x (Tensor): 输入特征图，形状[B, C, H, W]

        返回:
            (Tensor): 处理后的特征图，形状保持[B, C, H, W]
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """集成PSA注意力机制的C2f改进模块

    在标准C2f结构基础上，用PSABlock替代原有Bottleneck，
    实现注意力机制与跨层连接的融合设计。

    特性:
        - 继承C2f参数结构
        - 支持多注意力头配置
        - 保持通道分裂-重组机制

    示例:
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5, act='hard_swish')
        >>> x = torch.randn(1, 64, 128, 128)
        >>> print(model(x).shape)
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act=True):
        """
        初始化C2fPSA模块

        参数说明:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): PSABlock堆叠次数，默认1
            shortcut (bool): 是否启用跨层连接，默认False
            g (int): 分组卷积组数，默认1
            e (float): 隐藏层扩展系数，默认0.5
            act (bool/str): 激活函数类型，默认True
        """
        super().__init__(c1, c2, n, shortcut, g, e, act)

        # 重构模块列表为PSABlock
        self.m = nn.ModuleList(
            PSABlock(
                self.c,
                attn_ratio=0.5,  # 注意力比率保持默认
                num_heads=self.c // 64,  # 自动计算注意力头数
                act=act  # 传递激活函数参数
            ) for _ in range(n)
        )

    def forward_split(self, x):
        """替代chunk的split实现，保持功能兼容性"""
        return self.cv1(x).split((self.c, self.c), dim=1)


class LinearAddBlock(nn.Module):
    """ 线性相加卷积块，实现多分支卷积特征融合

    结构特性：
        - 并行3x3卷积与1x1卷积分支
        - 各分支带可学习缩放因子
        - 可选身份残差连接（当输入输出通道相同且stride=1时）
        - 支持CSLA模式（固定缩放参数）

    Args:
        in_channels (int): 输入特征图通道数
        out_channels (int): 输出特征图通道数
        kernel_size (int): 主卷积核尺寸，默认3
        stride (int): 卷积步长，默认1
        padding (int): 填充大小，默认1
        use_se (bool): 是否使用SE模块（暂未实现）
        is_csla (bool): 是否启用CSLA模式（冻结缩放层）
        conv_scale_init (float): 卷积分支缩放层初始值，默认1.0
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False, is_csla=False, conv_scale_init=1.0):
        super(LinearAddBlock, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()

        # 主卷积路径：3x3卷积 + 可学习缩放
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # 与BN配合使用通常关闭偏置
        )
        self.scale_conv = ScaleLayer(
            num_features=out_channels,
            use_bias=False,
            scale_init=conv_scale_init
        )

        # 辅助卷积路径：1x1卷积 + 可学习缩放
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,  # 保持与主卷积相同的下采样率
            padding=0,
            bias=False
        )
        self.scale_1x1 = ScaleLayer(
            num_features=out_channels,
            use_bias=False,
            scale_init=conv_scale_init
        )

        # 当满足通道数相同且无下采样时，添加身份连接路径
        if in_channels == out_channels and stride == 1:
            self.scale_identity = ScaleLayer(
                num_features=out_channels,
                use_bias=False,
                scale_init=1.0  # 身份缩放初始化为1
            )

        # 公共输出处理
        self.bn = nn.BatchNorm2d(out_channels)

        # CSLA模式：固定缩放参数（用于特定初始化策略）
        if is_csla:
            self.scale_1x1.requires_grad_(False)  # 停止梯度更新
            self.scale_conv.requires_grad_(False)

        # SE模块占位符（当前版本暂未实现）
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()  # 无SE时的空操作

    def forward(self, inputs):
        """ 前向传播流程：
        1. 双分支卷积结果相加
        2. 添加身份连接（条件满足时）
        3. BN + (SE) + ReLU
        """
        # 主分支与辅助分支融合
        out = self.scale_conv(self.conv(inputs)) + self.scale_1x1(self.conv_1x1(inputs))

        # 身份连接增强（跳跃连接）
        if hasattr(self, 'scale_identity'):
            out += self.scale_identity(inputs)  # 原始输入缩放后相加

        # 标准化与激活
        out = self.relu(self.se(self.bn(out)))
        return out


class ScaleLayer(torch.nn.Module):
    """ 可学习缩放偏置层，实现逐通道特征缩放与偏移

    功能特性：
        - 对输入张量进行逐通道缩放（必需）
        - 可选逐通道偏置（由use_bias控制）
        - 初始化时可设置缩放因子初始值
        - 支持四维张量输入（BCHW格式）

    Args:
        num_features (int): 特征通道数（对应输入张量的C维度）
        use_bias (bool): 是否启用偏置项，默认True
        scale_init (float): 缩放因子的初始化值，默认1.0
    """

    def __init__(self, num_features, use_bias=True, scale_init=1.0):
        super(ScaleLayer, self).__init__()
        # 缩放参数：维度与输入通道数一致
        self.weight = nn.Parameter(torch.Tensor(num_features))
        torch.nn.init.constant_(self.weight, scale_init)  # 按指定值初始化缩放因子

        self.num_features = num_features  # 记录特征维度

        # 偏置参数（可选）
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            torch.nn.init.zeros_(self.bias)  # 零初始化偏置
        else:
            self.bias = None  # 禁用偏置

    def forward(self, inputs):
        """ 前向传播逻辑：
        1. 将权重重塑为[1,C,1,1]维度以匹配输入形状
        2. 执行逐通道缩放（乘权重）
        3. 可选执行逐通道偏移（加偏置）
        """
        # 重塑权重维度（适配BCHW输入）
        weight_view = self.weight.view(1, self.num_features, 1, 1)

        # 基础缩放操作
        scaled = inputs * weight_view

        # 添加偏置（如果启用）
        if self.bias is not None:
            bias_view = self.bias.view(1, self.num_features, 1, 1)
            return scaled + bias_view
        return scaled

class RealVGGBlock(nn.Module):
    """ 标准VGG风格基础块，实现"Conv-BN-Activation"基础结构

    结构特性：
        - 经典单路径卷积结构（无残差连接）
        - 严格顺序：Conv2d -> BN -> (SE) -> ReLU
        - 卷积层默认禁用偏置（与BN配合优化）

    Args:
        in_channels (int): 输入特征图通道数
        out_channels (int): 输出特征图通道数
        kernel_size (int): 卷积核尺寸，默认3
        stride (int): 卷积步长，默认1（控制下采样率）
        padding (int): 填充大小，默认1（保持特征图尺寸）
        use_se (bool): 是否使用SE模块（暂未实现）
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False):
        super(RealVGGBlock, self).__init__()
        # 核心组件定义
        self.relu = nn.ReLU(inplace=True)  # 原地操作节省内存
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # 与BN配合通常关闭偏置
        )
        self.bn = nn.BatchNorm2d(out_channels)  # 标准化层

        # SE模块占位符（当前版本未实现）
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()  # 无SE时的空操作

    def forward(self, inputs):
        """ 前向传播流程：
        1. 卷积 -> 批归一化 -> (SE) -> ReLU激活
        2. 输出特征图尺寸由stride参数控制
        """
        # 标准处理流程
        x = self.conv(inputs)  # 空间特征提取
        x = self.bn(x)  # 数值稳定性优化
        x = self.se(x)  # 预留通道注意力接口
        return self.relu(x)  # 非线性激活