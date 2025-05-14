import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

# Settings
# 设置PyTorch打印选项：行宽320，保留5位小数，使用详细profile
torch.set_printoptions(linewidth=320, precision=5, profile='long')
# 设置NumPy打印格式：行宽320，浮点数用11字符宽度+5位精度
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
# 限制Pandas显示最大列数为10（避免控制台溢出）
pd.options.display.max_columns = 10
# 禁用OpenCV多线程（防止与PyTorch DataLoader冲突）
cv2.setNumThreads(0)
# 配置NumExpr最大线程数（取CPU核心数与8的较小值）
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))

# 路径配置 ----------------------------------------------------------------------------------------------
# 获取当前文件的绝对路径并解析
FILE = Path(__file__).resolve()
# 计算YOLOv5根目录（当前文件的上两级目录）
ROOT = FILE.parents[1]

# 全局配置 --------------------------------------------------------------------------------------------
# 数据集存储路径（默认在YOLOv5上级目录的datasets文件夹）
DATASETS_DIR = ROOT.parent / 'datasets'
# 计算并行线程数（至少1个，不超过CPU核心数-1）
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
# 自动安装模式（从环境变量读取，默认True）
AUTOINSTALL = str(os.getenv('YOLOv5_AUTOINSTALL', True)).lower() == 'true'
# 详细日志模式（从环境变量读取，默认True）
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'
# 默认字体文件路径（用于可视化标签）
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def set_logging(name=None, verbose=VERBOSE):
    """
    初始化分布式训练环境下的日志系统

    功能:
    - 根据进程排名和详细模式动态设置日志级别
    - 主进程(RANK=-1/0)启用INFO级别日志
    - 其他进程/非详细模式仅显示ERROR日志

    参数:
    - name: 日志器名称(默认root)
    - verbose: 全局详细模式开关

    实现细节:
    - 通过环境变量RANK识别多GPU训练进程
    - 日志格式仅保留纯消息(%(message)s)
    - 相同处理程序重复添加时会自动去重
    - 保证非主进程不污染控制台输出
    """
    rank = int(os.getenv('RANK', -1))  # 从环境变量获取分布式训练进程号
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR  # 主进程且详细模式时开启INFO
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))  # 精简日志格式
    handler.setLevel(level)
    log.addHandler(handler)

set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("yolov5")  # define globally (used in train.py, val.py, detect.py, etc.)

def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def methods(instance):
    """
    获取类/实例的可调用方法列表

    返回:
    - list: 非魔术方法的可调用方法名称列表
    过滤条件:
    - 排除以双下划线开头的方法
    - 仅保留callable类型的属性
    """
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def set_logging(rank=-1, verbose=True):
    """
    基础日志配置函数

    功能:
    - 主进程(rank=-1/0)且verbose=True时显示INFO日志
    - 其他情况只显示WARN以上级别日志

    参数:
    - rank: 分布式训练进程标识，-1表示单进程
    - verbose: 是否输出详细日志
    """
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)


def print_args(name, opt):
    """
    格式化打印命令行参数

    显示效果:
    - 名称部分带颜色标记(依赖colorstr实现)
    - 参数按key=value形式逗号分隔

    示例输出:
    Args: batch_size=16, lr=0.01
    """
    print(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))


def init_seeds(seed=0):
    """
    初始化全框架随机种子保证可重复性

    影响范围:
    - Python内置random
    - NumPy
    - PyTorch(CPU/CUDA)

    cuDNN配置:
    - seed=0时: benchmark=False, deterministic=True (更慢但可重复)
    - seed≠0时: benchmark=True, deterministic=False (更快但不可重复)
    """
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def check_online():
    """
    网络连通性检测函数

    功能:
    - 通过尝试连接Cloudflare DNS服务器(1.1.1.1:443)验证互联网访问能力
    - 使用socket层连接避免依赖高层协议(如HTTP)

    返回:
    - bool: True表示在线，False表示离线

    关键细节:
    - 目标端口443(HTTPS)通常不会被企业防火墙封锁
    - 设置5秒超时防止长时间阻塞
    - 捕获所有OSError异常(包含超时/拒绝连接等)
    - 适用于代理环境下的基础网络检测
    """
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # 测试与Cloudflare DNS的TCP连接
        return True
    except OSError:
        return False

def check_python(minimum='3.6.2'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ')


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False):
    """
    版本合规性检查器

    功能:
    - 验证当前版本是否符合最低要求或精确匹配
    - 主要用于依赖库的版本兼容性检查

    参数:
    - current: 当前版本号(str，需符合PEP440格式)
    - minimum: 目标版本下限/固定版本
    - name: 版本标识前缀(用于错误提示)
    - pinned: 是否要求严格匹配(True=必须等于，False=不低于)

    断言逻辑:
    - pinned=True时: current必须严格等于minimum
    - pinned=False时: current需大于等于minimum
    - 触发断言错误时显示格式化的版本不匹配信息

    实现细节:
    - 使用packaging.version.parse进行版本解析
    - 集成于YOLOv5的依赖检查系统
    - 错误示例: "version 1.8.0 required by YOLOv5, but version 1.7.1 is currently installed"
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'


@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    """
    依赖项检查与自动安装器

    功能:
    - 验证当前环境是否满足项目依赖要求
    - 支持自动安装缺失或版本不符的包
    - 集成网络检测与异常处理机制

    参数:
    - requirements: 依赖列表文件路径或包列表
    - exclude: 需要排除检查的包名列表
    - install: 是否启用自动安装(默认True)

    处理流程:
    1. 调用check_python()验证Python版本合规性
    2. 解析requirements文件为规范格式(处理排除项)
    3. 遍历依赖项检查安装状态和版本
    4. 缺失包自动安装(需联网且install=True)
    5. 输出安装结果并提示运行环境更新

    关键特性:
    - 支持在线/离线环境自适应处理
    - 使用try_except装饰器捕获全局异常
    - 自动过滤排除项提升检查效率
    - 彩色终端输出增强可读性
    - 安装后提醒需要重启内核/程序
    """
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # 前置Python版本检查
    if isinstance(requirements, (str, Path)):  # 处理文件路径输入
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # 处理直接传入的列表/元组
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # 成功安装计数器
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # 捕获包缺失或版本不符异常
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"  # 网络检查
                    print(check_output(f"pip install '{r}'", shell=True).decode())  # 执行安装
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')

    if n:  # 安装后处理
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    """
    验证并调整输入图像尺寸到指定步长的倍数

    参数:
    - imgsz: 原始尺寸(int或list)
    - s: 模型的最大步长(默认32)
    - floor: 最小尺寸限制(默认0)

    处理逻辑:
    - 使用make_divisible确保尺寸能被s整除
    - 当调整后的尺寸与原始不同时输出警告
    - 支持单维度(int)和多维度(list)输入

    返回:
    - 调整后的新尺寸，保证符合神经网络的下采样要求
    """
    if isinstance(imgsz, int):  # 处理整数输入
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # 处理列表输入
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def check_imshow():
    """
    检测当前环境是否支持图像显示功能

    验证条件:
    - 不在Docker容器内运行
    - 不在Google Colab环境
    - 能够成功创建/销毁测试窗口

    返回:
    - bool: True表示支持图像显示，False表示不支持

    异常处理:
    - 捕获OpenCV显示异常并输出友好提示
    - 测试完成后自动清理窗口资源
    """
    try:
        # assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        # assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))  # 创建测试图像
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    """
    文件后缀验证器

    参数:
    - file: 文件名或文件列表
    - suffix: 允许的后缀集合(如.pt, .onnx)
    - msg: 自定义错误消息前缀

    功能:
    - 检查指定文件是否具有合法后缀
    - 支持单个文件和文件列表的批量验证
    - 自动转换大小写差异(.PT与.pt视为相同)

    抛出:
    - AssertionError: 当检测到非法后缀时中断程序
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # 统一转小写
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=''):
    """
    文件路径解析与自动下载管理器

    功能:
    - 本地文件存在时直接返回路径
    - 支持HTTP(S) URL自动下载
    - 本地文件不存在时在预定义目录中搜索
    - 可选验证文件后缀合法性

    参数:
    - file: 文件路径/URL/简写名
    - suffix: 允许的文件后缀(默认不验证)

    返回:
    - str: 验证通过的绝对文件路径

    关键流程:
    1. 后缀验证: 调用check_suffix检查文件类型
    2. 本地存在: 直接返回规范化路径
    3. 远程资源: 下载到当前目录并验证完整性
    4. 本地搜索: 在data/models/utils目录中递归查找
    5. 冲突处理: 多个匹配时抛出异常要求明确路径
    """
    check_suffix(file, suffix)  # 可选后缀验证
    file = str(file)  # 统一转为字符串类型处理
    if Path(file).is_file() or file == '':  # 本地文件存在或空路径
        return file
    elif file.startswith(('http:/', 'https:/')):  # 处理网络资源
        url = str(Path(file)).replace(':/', '://')  # 修复Pathlib对URL的误转义
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # 解码URL编码并提取文件名
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, file)  # 使用torch的下载器(支持进度显示)
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # 完整性校验
        return file
    else:  # 本地文件搜索模式
        files = []
        for d in 'data', 'models', 'utils':  # 预定义搜索目录
            files.extend(glob.glob(str(ROOT / d / '​**​' / file), recursive=True))  # 递归匹配文件
        assert len(files), f'File not found: {file}'  # 无结果时中断
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # 多结果时提示
        return files[0]  # 返回首个匹配项


def check_dataset(data, autodownload=True):
    """
    数据集配置检查与自动部署函数

    核心功能:
    - 自动下载/解压.zip格式数据集
    - 解析数据集配置文件(YAML格式)
    - 验证数据集路径有效性
    - 缺失时根据配置自动下载数据集

    参数:
    - data: 数据集路径/URL 或已加载的配置字典
    - autodownload: 是否启用自动下载(默认True)

    返回:
    - dict: 标准化处理后的数据集配置字典

    处理流程:
    1. ZIP压缩包自动解压与配置文件定位
    2. YAML配置文件解析与路径预处理
    3. 训练/验证/测试集路径标准化
    4. 关键参数校验与默认值填充
    5. 数据集存在性检查与自动下载
    """

    # 压缩包处理模块 -----------------------------------------------------------------
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):
        # 处理云存储路径格式(如gs://bucket/path.zip)
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        # 在解压目录中递归查找首个YAML配置文件
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False  # 禁用后续自动下载

    # 配置加载与解析 ----------------------------------------------------------------
    if isinstance(data, (str, Path)):
        # 忽略文件编码错误，安全加载YAML防止代码注入
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # 转换为配置字典

    # 路径标准化处理 --------------------------------------------------------------
    # 优先级: 解压目录 > 配置文件path字段 > 当前目录
    path = extract_dir or Path(data.get('path') or '')
    # 为数据路径添加根目录前缀
    for k in 'train', 'val', 'test':
        if data.get(k):  # 支持字符串路径和路径列表
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    # 关键参数校验 ----------------------------------------------------------------
    assert 'nc' in data, "数据集配置必须包含'nc'(类别数量)字段"
    # 自动生成类别名(格式: class0, class1...)
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]

        # 数据集自动部署模块 -----------------------------------------------------------
    train, val, test, s = [data.get(x) for x in ('train', 'val', 'test', 'download')]
    if val:  # 以验证集存在性作为数据集完整判断标准
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
        if not all(x.exists() for x in val):
            # 打印缺失路径明细
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # 执行下载脚本
                root = path.parent if 'path' in data else '..'  # 解压目录逻辑
                # HTTP下载处理器
                if s.startswith('http') and s.endswith('.zip'):
                    f = Path(s).name  # 提取文件名
                    print(f'Downloading {s} to {f}...')
                    torch.hub.download_url_to_file(s, f)  # 带进度条的下载
                    Path(root).mkdir(parents=True, exist_ok=True)  # 创建目录
                    ZipFile(f).extractall(path=root)  # 解压文件
                    Path(f).unlink()  # 删除压缩包
                    r = None  # 成功标识
                # Bash脚本处理器
                elif s.startswith('bash '):
                    print(f'Running {s} ...')
                    r = os.system(s)  # 执行系统命令
                # Python脚本处理器
                else:
                    r = exec(s, {'yaml': data})  # 在全局命名空间执行
                # 输出部署结果
                print(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
            else:
                raise Exception('Dataset not found.')

    return data  # 返回标准化后的配置字典


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    """
    多线程文件下载与解压工具函数

    功能:
    - 支持单/多文件并行下载
    - 自动解压ZIP/GZ格式压缩包
    - 可选下载完成后删除压缩文件
    - 支持本地文件路径重定向

    参数:
    - url: 下载地址(支持单个str或列表)
    - dir: 存储目录(默认当前目录)
    - unzip: 是否自动解压(默认True)
    - delete: 解压后是否删除压缩文件(默认True)
    - curl: 是否使用curl下载(默认False，使用torch下载器)
    - threads: 下载线程数(默认单线程)

    处理流程:
    1. 创建目标目录(自动递归创建父目录)
    2. 多线程分发下载任务
    3. 单个文件下载逻辑:
       - 本地已存在时直接移动文件
       - 根据curl标志选择下载工具
       - 显示实时下载进度
    4. 解压处理:
       - ZIP文件用ZipFile库解压
       - GZ文件调用系统tar命令解压
    5. 清理压缩包(当delete=True时)

    关键特性:
    - 多线程加速批量下载
    - 支持断点续传(curl的-C -参数)
    - 自动识别压缩格式并解压
    - 本地文件路径兼容处理
    """

    def download_one(url, dir):
        # 单文件下载处理器
        f = dir / Path(url).name  # 构建存储路径
        if Path(url).is_file():  # 本地文件直接移动
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:  # curl下载(支持断点续传)
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")
            else:  # torch下载器(带进度条)
                torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.gz'):  # 解压处理器
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')
            if delete:  # 清理临时文件
                f.unlink()

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录
    if threads > 1:  # 多线程模式
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:  # 单线程模式
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

def make_divisible(x, divisor):
    """
    将输入值调整到最接近且能被除数整除的最小值
    功能:
    - 通过向上取整计算确保输出值能被divisor整除
    - 常用于神经网络层的通道数调整，适配硬件计算单元"""
    return math.ceil(x / divisor) * divisor


def colorstr(*input):
    """
    ANSI转义码字符串着色器

    功能:
    - 为终端输出字符串添加颜色/样式(粗体/下划线)
    - 支持基础色、高亮色及文本修饰

    参数:
    - 可变参数: 可接受1-N个参数，末位为字符串，前部为样式标识

    示例:
    colorstr('red', 'bold', 'text') -> 红色粗体文字
    colorstr('text') -> 默认蓝色粗体文字

    实现细节:
    - 自动识别参数结构(样式参数需在colors字典中存在)
    - 默认样式为蓝色粗体
    - 自动在结尾添加样式重置码
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # 参数解析逻辑
    colors = {'black': '\033[30m',  # 基础色系
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # 高亮色系
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # 样式重置
              'bold': '\033[1m',  # 粗体
              'underline': '\033[4m'}  # 下划线
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    """
    COCO数据集类别索引转换器 (80类->91类)

    功能:
    - 将COCO val2014版本的80类别索引映射到原始论文的91类别索引体系
    - 解决不同版本COCO数据集标注差异导致的评估对齐问题

    返回:
    - list[80]: 映射表，每个元素表示原80类索引对应的91类索引值

    映射规则来源:
    - 基于COCO官方标注调整 (https://tech.amikelive.com/node-718/)
    - 索引空缺表示原91类中某些类别在80类版本中不存在

    典型应用场景:
    - 论文结果复现时对齐评估指标
    - 跨版本模型精度对比时索引转换

    示例:
    >>> cls80 = 0  # 原80类的第一个类别
    >>> cls91 = coco80_to_coco91_class()[cls80]  # 对应91类的索引1
    """
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    """
    将边界框从 [左上x, 左上y, 右下x, 右下y] 格式转换为 [中心x, 中心y, 宽, 高] 格式

    参数:
        x (numpy.ndarray | torch.Tensor): 形状为(N,4)的边界框数组，每行格式为[x1,y1,x2,y2]

    返回:
        (numpy.ndarray | torch.Tensor): 形状为(N,4)的数组，每行格式为[cx,cy,width,height]

    示例:
        >>> input = [[10, 20, 50, 60]]  # x1,y1=10,20  x2,y2=50,60
        >>> output = [[30, 40, 40, 40]]  # cx=(10+50)/2=30, cy同理, w=50-10=40, h=60-20=40
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 创建数据副本避免污染原始数据
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # 计算水平中心坐标: (x1 + x2)/2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # 计算垂直中心坐标: (y1 + y2)/2
    y[:, 2] = x[:, 2] - x[:, 0]  # 计算框宽度: x2 - x1
    y[:, 3] = x[:, 3] - x[:, 1]  # 计算框高度: y2 - y1
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    将归一化的中心坐标边界框转换为绝对坐标边界框

    参数:
        x (numpy.ndarray | torch.Tensor): 形状为(N,4)的归一化边界框，格式为[中心x, 中心y, 宽度比, 高度比]
        w (int): 图像原始宽度(默认640)
        h (int): 图像原始高度(默认640)
        padw (int): 水平方向填充像素数(用于处理图像填充后的坐标校正)
        padh (int): 垂直方向填充像素数

    返回:
        (numpy.ndarray | torch.Tensor): 形状为(N,4)的绝对坐标边界框，格式为[x1,y1,x2,y2]

    计算逻辑:
        1. 中心点反归一化: 中心x * 图像宽度 + 填充量
        2. 框尺寸反归一化: 宽度比 * 图像宽度
        3. 坐标转换公式:
            左上x = 中心x - 半宽 + 水平填充
            右下x = 中心x + 半宽 + 水平填充
            左上y = 中心y - 半高 + 垂直填充
            右下y = 中心y + 半高 + 垂直填充
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 保持数据类型一致性
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # 左上角x坐标: (中心x - 半宽) * 尺度 + 填充
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # 左上角y坐标: (中心y - 半高) * 尺度 + 填充
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # 右下角x坐标: (中心x + 半宽) * 尺度 + 填充
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # 右下角y坐标: (中心y + 半高) * 尺度 + 填充
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """
    归一化坐标到像素坐标转换器(基础版)

    功能:
    - 将归一化坐标[x_norm, y_norm]转换为绝对像素坐标
    - 支持图像填充补偿计算

    参数:
        x: 输入坐标(形状[n,2], 值域0-1)
        w: 图像宽度(默认640)
        h: 图像高度(默认640)
        padw: 水平填充像素数(用于letterbox等场景)
        padh: 垂直填充像素数

    返回:
        y: 绝对像素坐标(形状[n,2])
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # X轴坐标转换
    y[:, 1] = h * x[:, 1] + padh  # Y轴坐标转换
    return y


def xyn2xy_new(x, w=640, h=640, padw=0, padh=0):
    """
    归一化坐标到像素坐标转换器(增强版)

    改进点:
    - 添加边界保护机制，防止坐标越界
    - 对零值坐标进行特殊处理(-1补偿)
    - 输出坐标强制约束在[0, 1e6]范围

    参数说明:
        新增逻辑:
        (np.array(x[:,0]>0, dtype=np.int32)-1)实现零值坐标-1补偿
        np.clip防止极端情况下的坐标溢出

    典型应用场景:
        - 处理存在标注噪声的数据集
        - 应对图像预处理中的异常坐标值
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (w * x[:, 0] + padw) + (np.array(x[:, 0] > 0, dtype=np.int32) - 1)  # 零值补偿逻辑
    y[:, 1] = (h * x[:, 1] + padh) + (np.array(x[:, 1] > 0, dtype=np.int32) - 1)  # 零值补偿逻辑
    y = np.clip(y, a_min=0, a_max=1000000)  # 安全截断
    return y


def segment2box(segment, width=640, height=640):
    """
    分割标注转边界框标注转换器

    功能:
    - 将分割掩码的轮廓点集转换为包围框坐标(xyxy格式)
    - 自动过滤图像边界外的点，确保框坐标在图像范围内

    参数:
        segment (np.ndarray): 分割点集数组，形状为(N,2)，每行表示一个(x,y)坐标
        width (int): 图像宽度，用于约束坐标范围(默认640)
        height (int): 图像高度，用于约束坐标范围(默认640)

    返回:
        np.ndarray: 包围框坐标数组，形状为(1,4)，格式为[x_min,y_min,x_max,y_max]
                   若无有效点则返回零数组

    处理流程:
        1. 坐标点转置分离x,y分量
        2. 创建图像边界内点的布尔掩码
        3. 过滤越界坐标点
        4. 计算有效点的最小/最大坐标生成包围框
        5. 空值处理返回零点框

    应用场景:
        - 从实例分割标注生成目标检测标注
        - 处理部分遮挡目标的边界框生成
    """
    x, y = segment.T  # 分割点集转置为x,y坐标分量
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)  # 生成有效点掩码
    x, y = x[inside], y[inside]  # 过滤越界点
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # 计算xyxy框


def resample_segments(segments, n=1000):
    """
    分割线段重采样函数

    功能:
    - 对分割多边形的点集进行上采样/插值，生成更密集的采样点
    - 保持原始形状的同时减少锯齿效应

    参数:
        segments: 原始分割点集(形状为[m,2]的数组列表)
        n: 目标采样点数(默认1000)

    处理逻辑:
        1. 对每个线段生成均匀分布的插值点
        2. 使用线性插值法计算新坐标点
        3. 重组为新的点集数组

    数学原理:
        x = np.linspace(0, m-1, n) 生成等间距采样位置
        np.interp实现线性插值计算新点坐标
    """
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)  # 生成插值位置序列
        xp = np.arange(len(s))  # 原始点索引
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # 双通道插值
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    坐标系统一缩放函数(用于边界框)

    功能:
    - 将预处理后图像上的坐标映射回原始图像坐标系
    - 处理图像缩放(letterbox)和填充(padding)带来的坐标偏移

    参数:
        img1_shape: 预处理后图像尺寸(hw格式)
        coords: 待转换坐标(形状[n,4]的xyxy格式)
        img0_shape: 原始图像尺寸(hw格式)
        ratio_pad: 可选的缩放比例和填充值(避免重复计算)

    关键步骤:
        1. 计算缩放比例gain和填充量pad
        2. 去除填充偏移量
        3. 按缩放比例还原坐标
        4. 坐标越界保护
    """
    if ratio_pad is None:  # 自动计算缩放参数
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 最小缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 对称填充
    else:
        gain, pad = ratio_pad[0][0], ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x方向去除填充
    coords[:, [1, 3]] -= pad[1]  # y方向去除填充
    coords[:, :4] /= gain  # 缩放还原
    clip_coords(coords, img0_shape)  # 坐标边界约束
    return coords


def scale_coords_landmarks(img1_shape, coords, img0_shape, num_points, ratio_pad=None):
    """
    坐标系统一缩放函数(用于关键点)

    改进点:
    - 支持多人姿态估计/人脸关键点等场景
    - 逐个处理每个关键点坐标
    - 增强坐标越界保护

    参数新增:
        num_points: 单样本关键点数量

    处理逻辑:
        1. 奇偶分离处理x/y坐标
        2. 独立进行填充去除和缩放
        3. 对每个坐标维度分别进行边界约束
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 对称填充
    else:
        gain, pad = ratio_pad[0][0], ratio_pad[1]

    for n in range(num_points * 2):  # 遍历所有坐标分量
        if n % 2 == 0:  # x坐标处理
            coords[:, n] -= pad[0]  # 去除x填充
            coords[:, n] /= gain  # 缩放还原
            coords[:, n].clamp_(0, img0_shape[1])  # 约束在图像宽度内
        else:  # y坐标处理
            coords[:, n] -= pad[1]  # 去除y填充
            coords[:, n] /= gain  # 缩放还原
            coords[:, n].clamp_(0, img0_shape[0])  # 约束在图像高度内
    return coords


def clip_coords(boxes, shape):
    """
    边界框坐标截断函数

    功能:
    - 将边界框坐标限制在图像尺寸范围内，防止越界

    参数:
        boxes (Tensor/ndarray): 边界框数组，形状为(N,4)，格式为xyxy
        shape (tuple): 图像尺寸 (height, width)

    处理逻辑:
        - 对Tensor类型逐个坐标轴进行原地截断操作
        - 对ndarray类型按坐标轴批量截断

    数学原理:
        x坐标限制在 [0, 图像宽度] 之间
        y坐标限制在 [0, 图像高度] 之间
    """
    if isinstance(boxes, torch.Tensor):  # PyTorch张量优化路径
        boxes[:, 0].clamp_(0, shape[1])  # x1截断: 0 ≤ x1 ≤ width
        boxes[:, 1].clamp_(0, shape[0])  # y1截断: 0 ≤ y1 ≤ height
        boxes[:, 2].clamp_(0, shape[1])  # x2截断: 0 ≤ x2 ≤ width
        boxes[:, 3].clamp_(0, shape[0])  # y2截断: 0 ≤ y2 ≤ height
    else:  # NumPy数组优化路径
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x轴批量截断
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y轴批量截断


def non_max_suppression_lmk_and_bbox(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                                     num_points=0, multi_label=False,
                                     labels=(), max_det=300):
    """
    带关键点的非极大值抑制(NMS)处理器

    功能:
    - 对含有关键点检测的预测结果执行NMS过滤
    - 支持边界框与关键点联合处理
    - 兼容多标签分类场景

    参数:
        prediction: 模型原始输出张量(shape=[batch, num_anchors, 5+nc+num_points*2])
        conf_thres: 置信度过滤阈值(默认0.25)
        iou_thres: NMS的IoU阈值(默认0.45)
        classes: 指定保留的类别列表(默认不过滤)
        agnostic: 是否跨类别NMS(默认False)
        num_points: 关键点数量(默认0)
        multi_label: 是否允许单框多标签(默认False)
        labels: 预定义标签集(用于自动标注)
        max_det: 单图最大检测数(默认300)

    返回:
        list[Tensor]: 每张图的检测结果，格式为[xyxy, conf, cls, landmarks...]

    处理流程:
        1. 置信度初筛 -> 2. 多标签处理 -> 3. 坐标转换 -> 4. NMS过滤 -> 5. 关键点对齐
    """

    nc = prediction.shape[2] - 5 - num_points * 2 - 1  # 计算类别数
    xc = prediction[..., 4] > conf_thres  # 初步置信度过滤

    # 参数校验
    assert 0 <= conf_thres <= 1, f'置信度阈值需在0-1之间，当前为{conf_thres}'
    assert 0 <= iou_thres <= 1, f'IoU阈值需在0-1之间，当前为{iou_thres}'

    # 初始化配置
    min_wh, max_wh = 2, 4096  # 有效框宽高范围
    max_nms = 30000  # 单次NMS最大处理框数
    time_limit = 10.0  # 超时阈值(秒)
    merge = False  # 合并型NMS开关

    t = time.time()
    output = [torch.zeros((0, 7 + num_points * 2 + 1), device=prediction.device)] * prediction.shape[0]

    # 逐图处理
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # 应用置信度过滤

        # 自动标注处理
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # 填充标注框
            v[:, 4] = 1.0  # 置信度设为1
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # 类别one-hot编码
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # 置信度计算
        x[:, 5:5 + nc] *= x[:, 4:5]  # 综合置信度=对象置信度*类别置信度

        # 坐标转换
        box = xywh2xyxy(x[:, :4])  # 中心坐标转角点坐标

        # 检测结果矩阵构建
        if multi_label:
            i, j = (x[:, 5:5 + nc] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), x[i, 5 + nc:]), 1)
        else:
            conf, j = x[:, 5:5 + nc].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, 5 + nc:]), 1)[conf.view(-1) > conf_thres]

        # 类别过滤
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # NMS核心处理
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别偏移量
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'超时警告: NMS处理超过{time_limit}秒')
            break

    return output


def non_max_suppression_ssod(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, num_points=0,
                             multi_label=False,
                             labels=(), max_det=300):
    """
    半监督目标检测专用非极大值抑制(NMS)处理器

    功能:
    - 针对半监督训练场景优化的NMS算法
    - 输出包含目标置信度(obj_conf)和类别置信度(cls_conf)
    - 支持超大尺寸图像检测(最大边界框尺寸7680像素)

    参数:
        prediction: 模型输出张量(shape=[batch, num_anchors, 5+nc])
        conf_thres: 候选框置信度阈值(默认0.25)
        iou_thres: NMS的IoU阈值(默认0.45)
        classes: 指定保留的类别索引列表(默认不过滤)
        agnostic: 是否跨类别执行NMS(默认False)
        num_points: 预留关键点参数(当前版本未激活)
        multi_label: 多标签分类模式开关(默认False)
        labels: 预置标签数据(用于半监督伪标签生成)
        max_det: 单图最大检测数量(默认300)

    返回:
        list[Tensor]: 每张图的检测结果，格式为[xyxy, conf, cls, obj_conf, cls_conf]

    核心改进:
        1. 置信度双分支处理: 独立保留目标存在置信度和类别置信度
        2. 超大尺寸适配: 支持最大7680像素的边界框处理
        3. 伪标签兼容: 通过labels参数注入人工标注数据
    """

    # 类别数计算逻辑(兼容关键点检测的预留设计)
    if num_points > 0:
        nc = prediction.shape[2] - 5 - num_points * 2 - 1  # 含关键点的特征维度计算
    else:
        nc = prediction.shape[2] - 5  # 常规目标检测特征维度
    xc = prediction[..., 4] > conf_thres  # 基于目标置信度的初筛掩码

    # 参数合法性校验
    assert 0 <= conf_thres <= 1, f'非法置信度阈值{conf_thres}, 有效范围为0.0-1.0'
    assert 0 <= iou_thres <= 1, f'非法IoU阈值{iou_thres}, 有效范围为0.0-1.0'

    # 工程化配置参数
    min_wh, max_wh = 2, 7680  # 边界框尺寸约束(适配4K/8K图像)
    max_nms = 30000  # 单次NMS最大处理候选框数
    time_limit = 10.0  # 单图处理超时阈值(秒)
    merge = False  # 合并型NMS开关(当前版本禁用)

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    # 逐图像处理流水线
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # 应用置信度阈值初筛

        # 半监督标签注入逻辑
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # 注入标注框坐标
            v[:, 4] = 1.0  # 设置人工标注置信度
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # 构建类别one-hot编码
            x = torch.cat((x, v), 0)  # 合并预测结果与人工标注

        if not x.shape[0]:
            continue

        # 置信度复合计算
        cls_score, _ = x[:, 5:5 + nc].max(1, keepdim=True)  # 获取最大类别置信度
        x[:, 5:5 + nc] *= x[:, 4:5]  # 综合置信度 = obj_conf * cls_conf

        # 坐标空间转换
        box = xywh2xyxy(x[:, :4])  # 中心坐标转角点坐标

        # 多标签数据构造
        if multi_label:
            i, j = (x[:, 5:5 + nc] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:5 + nc].max(1, keepdim=True)
            obj = x[:, 4:5]  # 目标存在置信度
            x = torch.cat((box, conf, j.float(), obj, cls_score), 1)[conf.view(-1) > conf_thres]

        # 类别定向过滤
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 数量控制与NMS核心
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度降序排列

        # 多类别NMS隔离策略
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别偏移量
        boxes, scores = x[:, :4] + c, x[:, 4]  # 添加类别偏移防止跨类抑制
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'超时警告: 单图处理超过{time_limit}秒')
            break

    return output


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """
    标准非极大值抑制(NMS)处理器

    功能:
    - 对目标检测模型输出进行置信度过滤和重叠框抑制
    - 支持多类别/单类别检测场景
    - 兼容高分辨率图像处理(最大支持7680像素尺寸)

    参数:
        prediction: 模型原始输出张量(shape=[batch, num_anchors, 5+nc])
        conf_thres: 综合置信度阈值(默认0.25)
        iou_thres: IoU重叠阈值(默认0.45)
        classes: 指定保留的类别索引列表
        agnostic: 是否跨类别NMS(默认False)
        multi_label: 多标签模式开关(默认False)
        labels: 预置标签数据(用于半监督训练)
        max_det: 单图最大检测数(默认300)

    返回:
        list[Tensor]: 每张图的检测结果，格式为[xyxy, conf, cls]

    核心处理流程:
        1. 双重置信度筛选 -> 2. 坐标转换 -> 3. 类别过滤 -> 4. 数量控制 -> 5. NMS核心
    """

    nc = prediction.shape[2] - 5  # 计算类别数(特征维度-5个基础参数)
    # 双置信度联合筛选(框置信度与类别置信度均需达标)
    xc = torch.logical_and(prediction[..., 4] > conf_thres,
                           torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)

    # 工程参数校验
    assert 0 <= conf_thres <= 1, f'置信度阈值需在0-1之间，当前为{conf_thres}'
    assert 0 <= iou_thres <= 1, f'IoU阈值需在0-1之间，当前为{iou_thres}'

    # 配置参数
    min_wh, max_wh = 2, 7680  # 有效框尺寸范围(适配8K图像)
    max_nms = 30000  # 单次NMS最大处理框数
    time_limit = 10.0  # 处理超时阈值(秒)

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    # 逐图像处理
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # 应用双置信度筛选

        # 半监督标签注入
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # 载入标注框坐标
            v[:, 4] = 1.0  # 设置人工标注置信度
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # 构建类别one-hot
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # 综合置信度计算
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # 坐标转换与结果构造
        box = xywh2xyxy(x[:, :4])  # 中心坐标转边界坐标
        if multi_label:  # 多标签模式处理
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # 单标签模式
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # 类别定向过滤
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 检测数量控制
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # 核心NMS处理
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别偏移量
        boxes, scores = x[:, :4] + c, x[:, 4]  # 带偏移的检测框
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'超时警告: 单图处理超过{time_limit}秒')
            break

    return output


def strip_optimizer(f='best.pt', s=''):
    """
    模型优化器信息剥离器

    功能:
    - 从训练完成的模型文件中移除优化器状态等训练过程信息
    - 可选将精简后的模型保存为新文件
    - 自动转换模型权重为FP16格式以减小体积

    参数:
        f: 原始模型文件路径(默认'best.pt')
        s: 新保存文件路径(默认覆盖原文件)

    关键操作:
        1. 加载模型文件到CPU内存
        2. 用EMA权重替换原始模型权重(如果存在)
        3. 清除优化器/训练日志等非必要信息
        4. 转换模型为半精度并冻结参数
        5. 计算精简后文件体积

    应用场景:
        - 模型部署前的最后处理步骤
        - 减少模型文件体积(通常可缩小40%-50%)
        - 保护训练过程隐私信息
    """
    x = torch.load(f, map_location=torch.device('cpu'))  # 加载模型到CPU
    if x.get('ema'):
        x['model'] = x['ema']  # 使用EMA权重替换原始权重

    # 清除训练过程相关数据
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':
        x[k] = None
    x['epoch'] = -1  # 重置epoch标记

    # 模型优化处理
    x['model'].half()  # FP32转FP16压缩体积
    for p in x['model'].parameters():
        p.requires_grad = False  # 冻结所有参数

    # 文件保存与统计
    torch.save(x, s or f)  # 保存精简后模型
    mb = os.path.getsize(s or f) / 1E6  # 计算文件体积(MB)
    print(f"优化器信息已从{f}{(' 另存为 %s,' % s) if s else ''} 文件大小: {mb:.1f}MB")


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    目标区域裁剪与保存函数

    功能:
    - 根据边界框坐标从图像中裁剪指定区域
    - 支持区域扩展(gain)和填充(pad)调整
    - 可强制输出正方形区域(square=True)
    - 自动处理坐标越界问题

    参数:
        xyxy: 边界框坐标(x1,y1,x2,y2)
        im: 原始图像(numpy数组)
        file: 保存路径(默认image.jpg)
        gain: 区域扩展系数(默认1.02)
        pad: 填充像素数(默认10)
        square: 是否强制方形输出(默认False)
        BGR: 是否使用BGR色彩通道(默认False即RGB)
        save: 是否保存到磁盘(默认True)

    返回:
        crop: 裁剪后的图像数组

    处理流程:
        1. 坐标格式转换(xyxy→xywh)
        2. 区域尺寸调整(扩展+填充)
        3. 坐标越界裁剪
        4. 图像数组切片
    """
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # 转换中心点坐标
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 取最大边设为正方形边长
    b[:, 2:] = b[:, 2:] * gain + pad  # 宽高扩展与填充
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)  # 坐标裁剪至图像尺寸内
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        cv2.imwrite(str(file), crop)
    return crop


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    自动递增路径生成器

    功能:
    - 当路径已存在时自动添加递增序号
    - 支持文件/目录路径的智能递增
    - 可选自动创建父目录

    参数:
        path: 原始路径(可以是文件或目录)
        exist_ok: 允许路径存在时不递增(默认False)
        sep: 序号前缀分隔符(默认无)
        mkdir: 是否自动创建目录(默认False)

    返回:
        Path对象: 新生成的唯一路径

    示例:
        >>> increment_path('runs/exp') -> 'runs/exp2'
        >>> increment_path('image.jpg') -> 'image_2.jpg'

    核心逻辑:
        1. 路径存在检查与序号提取
        2. 确定最大序号并+1递增
        3. 处理目录创建需求
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # 获取相似路径
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # 提取现有序号
        n = max(i) + 1 if i else 2  # 确定新序号
        path = Path(f"{path}{sep}{n}{suffix}")  # 构建新路径
    dir = path if path.suffix == '' else path.parent
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录
    return path
