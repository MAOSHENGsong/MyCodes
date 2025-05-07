"""
回调函数工具模块
实现训练过程的事件钩子管理
"""
import datetime
import os
import platform
import subprocess
from pathlib import Path

import torch


class Callbacks:
    """YOLOv5 回调函数管理器

    功能特性：
    - 提供训练全生命周期的17个关键事件钩子
    - 支持多回调函数注册与批量触发
    - 允许动态添加自定义监控逻辑

    核心方法：
    register_action - 注册回调到指定事件点
    run - 触发指定事件点的所有回调
    """

    # 预定义的事件钩子集合（全训练周期覆盖）
    _callbacks = {
        # 预训练阶段
        'on_pretrain_routine_start': [],  # 预训练流程开始时
        'on_pretrain_routine_end': [],  # 预训练流程结束时

        # 主训练阶段
        'on_train_start': [],  # 整个训练开始
        'on_train_epoch_start': [],  # 每个epoch训练开始
        'on_train_batch_start': [],  # 每个训练batch开始
        'optimizer_step': [],  # 优化器更新参数时
        'on_before_zero_grad': [],  # 梯度清零前
        'on_train_batch_end': [],  # 每个训练batch结束
        'on_train_epoch_end': [],  # 每个epoch训练结束

        # 验证阶段
        'on_val_start': [],  # 验证流程开始
        'on_val_batch_start': [],  # 每个验证batch开始
        'on_val_image_end': [],  # 每张验证图像处理完
        'on_val_batch_end': [],  # 每个验证batch结束
        'on_val_end': [],  # 验证流程结束

        # 综合与收尾
        'on_fit_epoch_end': [],  # 每个完整epoch结束（训练+验证）
        'on_model_save': [],  # 模型保存时
        'on_train_end': [],  # 整个训练结束

        # 资源清理
        'teardown': [],  # 结束时的资源释放
    }

    def register_action(self, hook, name='', callback=None):
        """注册回调函数到指定事件钩子

        参数：
            hook     : 要注册的事件钩子名称（必须是预定义的钩子）
            name     : 回调标识名称（用于调试和日志）
            callback : 要注册的回调函数（必须可调用）

        示例：
            register_action('on_train_end', 'save_log', my_save_function)
        """
        # 参数校验
        assert hook in self._callbacks, f"无效钩子名称 '{hook}'，可用钩子：{list(self._callbacks.keys())}"
        assert callable(callback), f"回调对象必须可调用，当前类型：{type(callback)}"

        # 添加到对应钩子的回调列表
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """获取已注册的回调信息

        参数：
            hook : 指定钩子名称，None表示获取全部

        返回：
            当hook指定时返回该钩子的回调列表，否则返回完整回调字典
        """
        return self._callbacks if hook is None else self._callbacks.get(hook, [])

    def run(self, hook, *args, **kwargs):
        """触发指定钩子的所有回调函数

        参数：
            hook : 要触发的事件钩子名称
            *args: 传递给回调的位置参数
            **kwargs: 传递给回调的关键字参数

        典型应用场景：
            run('on_train_start')  # 训练开始时触发所有相关回调
        """
        assert hook in self._callbacks, f"无效钩子名称 '{hook}'，请先注册"

        # 遍历执行该钩子的所有回调
        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)  # 透传所有参数


def select_device(device='', batch_size=None):
    """
    自动选择并配置训练设备（CPU/GPU），支持多GPU设置

    功能特性：
    - 自动检测CUDA可用性
    - 支持指定单GPU、多GPU或CPU
    - 验证batch_size与GPU数量的兼容性
    - 输出详细的设备环境信息

    参数：
    device : str, 可选
        设备标识，支持格式：
        'cpu' - 强制使用CPU
        '0'   - 使用第0号GPU
        '0,1' - 使用多GPU（第0和第1号GPU）
        默认自动选择可用GPU，无GPU时使用CPU
    batch_size : int, 可选
        批量大小，使用多GPU时会验证是否可被GPU数量整除

    返回：
        torch.device : 配置好的计算设备对象

    实现流程：
    1. 环境初始化与设备参数解析
    2. CUDA可用性检测与配置
    3. 多GPU兼容性检查
    4. 设备信息收集与日志输出
    """

    # 构建基础信息字符串（包含版本信息）
    s = f'EfficientTeacher  {git_describe() or date_modified()} torch {torch.__version__} '  # 项目信息+版本

    # 设备参数标准化处理
    device = str(device).strip().lower().replace('cuda:', '')  # 统一转换为小写并移除cuda:前缀
    cpu = device == 'cpu'  # CPU模式标志

    # 强制CPU模式处理
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 通过环境变量禁用CUDA
    elif device:  # 指定了GPU设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # 设置可见GPU设备
        assert torch.cuda.is_available(), f'CUDA不可用，请求的设备 {device} 无效'  # 验证CUDA环境

    # 确定最终计算设备类型
    cuda = not cpu and torch.cuda.is_available()

    # CUDA设备详细配置
    if cuda:
        devices = device.split(',') if device else '0'  # 解析设备列表（支持多GPU）
        n = len(devices)  # GPU数量

        # 多GPU批量大小验证
        if n > 1 and batch_size:
            assert batch_size % n == 0, f'批量大小 {batch_size} 必须能被GPU数量 {n} 整除'

        # 收集GPU信息
        space = ' ' * (len(s) + 1)  # 信息缩进对齐
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)  # 获取设备属性
            # 格式化设备信息（名称，显存）
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MB)\n"
    else:
        s += 'CPU\n'  # CPU模式信息

    # 跨平台安全日志输出（处理Windows终端编码问题）
    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)

    # 返回PyTorch设备对象（多GPU时返回第一个设备）
    return torch.device('cuda:0' if cuda else 'cpu')

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository

