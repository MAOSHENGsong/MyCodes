import torch.nn as nn

# 定义激活函数字典（包含常用激活函数及空操作）
activations = {
    'ReLU': nn.ReLU,  # 标准ReLU激活
    'LeakyReLU': nn.LeakyReLU,  # 带泄漏的ReLU
    'ReLU6': nn.ReLU6,  # 限制输出最大值的ReLU
    'SELU': nn.SELU,  # 自归一化激活函数
    'ELU': nn.ELU,  # 指数线性单元
    'GELU': nn.GELU,  # GPT使用的激活函数
    None: nn.Identity  # 空操作（用于跳过激活）
}


def act_layers(name):
    """根据名称返回配置好的激活函数层

    参数：
    name : str | None
        激活函数名称，必须是activations字典的键

    返回：
    nn.Module : 配置好的激活函数实例

    特点：
    - 为LeakyReLU预设negative_slope=0.1（常用值）
    - 默认启用inplace操作节省内存（GELU除外）
    - 包含None返回Identity层的设计（灵活控制网络结构）
    """
    assert name in activations.keys(), f"无效的激活函数名: {name}，可选: {list(activations.keys())}"

    # 特殊配置的激活函数
    if name == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)  # 泄漏系数0.1

    if name == 'GELU':
        return nn.GELU()  # GELU无需inplace（PyTorch官方实现无此参数）

    # 常规激活函数（启用inplace节省内存）
    return activations[name](inplace=True)  # 默认使用inplace操作
    # return activations[name](inplace=False)  # 备选方案（避免可能的梯度问题）