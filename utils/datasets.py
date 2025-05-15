import glob
import hashlib
import logging
import os
import random
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import letterbox, copy_paste, random_perspective_keypoints, mixup, augment_hsv, Albumentations
from utils.general import check_requirements, xywhn2xyxy, xyn2xy, xyn2xy_new, clean_str, xyxy2xywhn
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def get_hash(paths):
    """
    计算一组文件或目录路径的哈希值

    参数:
        paths: 文件或目录路径列表

    返回:
        str: 生成的MD5哈希值

    处理逻辑:
        1. 计算所有存在路径的文件大小总和
        2. 使用文件大小总和初始化MD5哈希对象
        3. 将所有路径字符串连接后更新哈希对象
        4. 返回16进制格式的哈希值

    用途:
        - 验证数据集完整性
        - 检测文件变化
    """
    # 计算所有存在路径的文件大小总和
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    # 使用文件大小总和初始化MD5哈希
    h = hashlib.md5(str(size).encode())
    # 将所有路径连接成字符串并更新哈希
    h.update(''.join(paths).encode())
    # 返回16进制表示的哈希值
    return h.hexdigest()

class DistributeBalancedBatchSampler(torch.utils.data.distributed.DistributedSampler):
    """
    分布式平衡批量采样器（继承自DistributedSampler）

    功能:
        1. 按类别平衡采样数据，确保每个批次中各类别样本数量均衡
        2. 支持多标签样本（标签为列表或数组的情况）
        3. 自动对样本数量少的类别进行过采样，达到最大类别样本数
        4. 支持分布式训练，通过num_replicas和rank参数分配样本

    参数:
        dataset: 原始数据集
        num_replicas: 分布式训练中的进程总数
        rank: 当前进程的rank
        balance_type: 平衡类型（默认'class_balance'，仅支持类别平衡）
        labels: 数据集标签列表（可选，若数据集无__getitem__标签获取逻辑则需传入）
    """

    def __init__(self, dataset, num_replicas, rank, balance_type='class_balance', labels=None):
        self.labels = labels
        self.oridata = dataset
        self.dataset = dict()  # 按标签分组的样本索引字典 {label: [index1, index2,...]}
        self.balanced_max = 0  # 所有类别中的最大样本数
        self.balance_type = balance_type

        # 按标签分组样本索引（支持单标签和多标签）
        for idx in range(len(dataset)):
            label_list = self._get_label(dataset, idx)  # 获取样本标签（可能是单个标签或标签列表）
            if not isinstance(label_list, (list, np.ndarray)):  # 单标签情况
                label = label_list
                if label not in self.dataset:
                    self.dataset[label] = []
                self.dataset[label].append(idx)
                self.balanced_max = max(self.balanced_max, len(self.dataset[label]))
            else:  # 多标签情况，每个标签均添加样本索引
                for label in label_list:
                    if label not in self.dataset:
                        self.dataset[label] = []
                    self.dataset[label].append(idx)
                    self.balanced_max = max(self.balanced_max, len(self.dataset[label]))

        # 对样本数不足的类别进行过采样（随机重复样本直至达到balanced_max）
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))  # 随机重复已有索引

        self.keys = list(self.dataset.keys())  # 所有类别标签列表
        self.currentkey = 0  # 当前处理的类别索引
        self.indices = [-1 + rank] * len(self.keys)  # 各标签已采样到的索引位置（初始为rank-1）
        self.num_replicas = num_replicas  # 分布式进程总数
        self.rank = rank  # 当前进程rank
        self.epoch = 0  # 训练轮次（用于洗牌种子更新）
        self.shuffle = True  # 是否开启洗牌
        self.seed = 0  # 随机种子
        self.balanced_list = range(self.balanced_max)  # 平衡后的样本索引范围

    def _get_label(self, dataset, idx):
        """获取样本标签（优先从labels参数获取，否则调用dataset.__getitem__）"""
        if self.labels is not None:
            return self.labels[idx]
        else:
            return dataset[idx][1]  # 假设dataset[idx]返回格式为(img, label, ...)

    def __iter__(self):
        """生成采样索引迭代器"""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)  # 每个epoch使用不同种子确保洗牌不同
            self.random_indices = []
            # 为每个类别生成随机排列的索引列表
            for i in range(len(self.keys)):
                self.random_indices.append(torch.randperm(len(self.balanced_list), generator=g).tolist())

        # 按类别轮流采样，每个类别每次取num_replicas个样本（分布式均衡分配）
        while self.indices[self.currentkey] < self.balanced_max - self.num_replicas:
            self.indices[self.currentkey] += self.num_replicas  # 跳过当前进程不需要的样本
            if self.shuffle:
                # 使用洗牌后的索引获取样本
                yield self.dataset[self.keys[self.currentkey]][
                    self.random_indices[self.currentkey][self.indices[self.currentkey]]]
            else:
                # 顺序获取样本
                yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)  # 循环切换类别

        # 重置索引，准备下一轮采样
        self.indices = [-1 + self.rank] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        """
        获取样本标签（支持不同平衡策略的标签解析）

        参数:
            dataset: 数据集对象
            idx: 样本索引
            labels: 备用标签列表（未使用，兼容接口）

        返回:
            label_name: 样本标签（根据平衡策略解析）

        逻辑:
            1. 优先使用预存的labels列表（self.labels）
            2. 无预存标签时根据平衡类型解析:
               - dir_balance: 从图像文件路径中提取类别名（假设路径结构为'/.../类别名/...'）
               - class_balance: 从数据集标注中提取唯一类别索引（处理多标签场景）
        """
        if self.labels is not None:
            return self.labels[idx].item()  # 转换为标量值返回

        else:
            # 按平衡策略猜测标签
            if self.balance_type == 'dir_balance':
                # 从文件路径中提取第4级目录作为类别标签（如'/a/b/c/label/d.jpg'）
                label_name = dataset.img_files[idx].split('/')[3]
            elif self.balance_type == 'class_balance':
                # 从标注中获取所有唯一类别索引（支持单标签/多标签场景）
                label_name = np.unique(dataset.labels[idx][:, 0])
            return label_name

    def __len__(self):
        """
        返回分布式采样器的样本数量

        计算逻辑:
            总样本数除以分布式进程数（整数除法），表示每个进程分配的样本量
        """
        return int(len(self.oridata) / self.num_replicas)

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    平衡批量采样器（非分布式）

    功能:
        1. 按类别平衡采样数据，通过过采样使各标签样本数一致
        2. 支持从数据集路径或预存标签解析样本类别
        3. 可配置洗牌功能实现随机采样

    参数:
        dataset: 原始数据集
        labels: 预存标签列表（可选，优先于数据集解析）
    """

    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()  # 按标签分组的样本索引字典 {label: [index1, index2,...]}
        self.balanced_max = 0  # 所有类别中的最大样本数

        # 按标签分组样本索引
        for idx in range(len(dataset)):
            label = self._get_label(dataset, idx)  # 获取样本标签
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)
            self.balanced_max = max(self.balanced_max, len(self.dataset[label]))  # 更新最大样本数

        # 对样本数不足的类别进行过采样（随机重复直至达到balanced_max）
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))  # 随机重复索引

        self.keys = list(self.dataset.keys())  # 所有类别标签列表
        self.currentkey = 0  # 当前处理的类别索引
        self.indices = [-1] * len(self.keys)  # 各标签已采样到的索引位置（初始为-1）
        self.seed = 0  # 随机种子
        self.balanced_list = range(self.balanced_max)  # 平衡后的样本索引范围
        self.shuffle = True  # 是否开启洗牌

    def __iter__(self):
        """生成平衡采样的索引迭代器"""
        if self.shuffle:
            # 生成随机排列的索引（每个类别独立洗牌）
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))  # 随机种子初始化
            self.random_indices = []
            for i in range(len(self.keys)):
                self.random_indices.append(
                    torch.randperm(len(self.balanced_list), generator=g).tolist())  # type: ignore

        # 按类别轮流采样，每个类别按顺序或随机索引获取样本
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1  # 移动到下一个样本索引
            if self.shuffle:
                # 使用洗牌后的索引获取样本
                yield self.dataset[self.keys[self.currentkey]][
                    self.random_indices[self.currentkey][self.indices[self.currentkey]]]
            else:
                # 顺序获取样本
                yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)  # 循环切换类别

        self.indices = [-1] * len(self.keys)  # 重置索引，准备下一轮采样

    def _get_label(self, dataset, idx, labels=None):
        """
        解析样本标签（优先使用预存标签，否则从文件路径提取）

        参数:
            dataset: 数据集对象
            idx: 样本索引
            labels: 备用标签列表（未使用，兼容接口）

        返回:
            label: 样本标签（假设文件路径结构为'/.../label/...'，取第4级目录）
        """
        if self.labels is not None:
            return self.labels[idx].item()  # 转换为标量值
        else:
            # 从图像文件路径中提取第4级目录作为标签（如'/a/b/c/label/d.jpg'）
            file_name = dataset.img_files[idx]
            return file_name.split('/')[3]

    def __len__(self):
        """返回采样器的总样本数（各标签平衡后的总数）"""
        return self.balanced_max * len(self.keys)

class LoadImagesAndLabels(Dataset):
    """
    YOLOv5训练/验证数据集加载器（负责图像和标签的加载、预处理及增强）

    参数:
        path: 数据集路径（支持多路径'||'分隔、通配符*重复如'dir*3'）
        img_size: 输入图像尺寸
        batch_size: 批次大小（影响马赛克增强参数）
        augment: 是否启用数据增强
        hyp: 数据增强超参数配置
        rect: 是否使用矩形训练（非等比例缩放）
        image_weights: 是否按图像目标数量加权采样
        cache_images: 图像缓存模式（False/True/'disk'）
        single_cls: 是否合并为单类别训练
        stride: 网络步长（用于尺寸对齐）
        pad: 矩形训练填充值
        cfg: 配置对象（包含关键点、ID跟踪等参数）
        prefix: 日志前缀

    功能:
        1. 多源数据解析（目录/文件/txt列表）
        2. 图像与标签自动匹配（支持YOLO格式标签）
        3. 数据缓存机制（.cache文件存储标签有效性）
        4. 马赛克数据增强（训练阶段）
        5. Albumentations增强支持
        6. 矩形训练（减少填充，提升推理效率）
        7. 关键点检测与ID跟踪支持
    """
    cache_version = 0.6  # 数据集标签缓存文件版本号

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, cfg=None, prefix=''):
        self.img_size = img_size  # 目标图像尺寸
        self.augment = augment  # 数据增强开关
        self.hyp = hyp  # 增强超参数（如旋转、缩放因子）
        self.image_weights = image_weights  # 图像权重采样（按目标数量加权）
        self.rect = False if image_weights else rect  # 矩形训练开关（权重采样时强制关闭）
        self.mosaic = self.augment and not self.rect  # 马赛克增强开关（训练阶段且非矩形模式）
        self.mosaic_border = [-img_size // 2, -img_size // 2]  # 马赛克中心偏移范围（控制拼接区域）
        self.stride = stride  # 网络步长（用于尺寸对齐计算）
        self.path = path  # 数据集路径
        self.albumentations = Albumentations() if augment else None  # Albumentations增强器
        self.debug = False  # 调试模式（输出额外信息）
        self.with_id = False  # 是否包含跟踪ID（多目标跟踪任务）
        self.pseudo_ids = False  # 是否使用伪ID（数据增强时生成虚拟ID）

        # 从配置中解析关键点和ID相关参数
        if cfg:
            self.num_points = cfg.Dataset.np  # 关键点数量（0表示无关键点任务）
            if cfg.Dataset.num_ids > 0:
                self.with_id = True  # 启用ID跟踪
            self.pseudo_ids = cfg.Dataset.pseudo_ids  # 伪ID开关
            self.cfg = cfg
            self.debug = cfg.Dataset.debug  # 调试模式
            self.nc = cfg.Dataset.nc  # 类别数
            self.include_class = cfg.Dataset.include_class  # 包含的类别列表（过滤标签）
        else:
            self.num_points = 0
            self.nc = None
            self.include_class = []

        # 解析多路径输入（支持通配符重复，如'dir*3'展开为3个dir路径）
        newpath = []
        for p in path.split('||'):
            if '*' in p:
                dirpath, times = p.split('*')
                newpath.extend([dirpath] * int(times))  # 重复路径指定次数
            else:
                newpath.append(p)
        path = newpath

        # 扫描图像文件（支持目录递归、txt列表文件）
        self.img_files = []
        for p in path:
            p = Path(p).strip()
            if p.is_dir():  # 目录模式：递归获取所有图像文件
                self.img_files += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():  # 文件模式：读取txt列表文件
                with open(p, 'r') as t:
                    parent = str(p.parent) + os.sep
                    # 将相对路径转换为绝对路径（处理'./'前缀）
                    self.img_files += [x.replace('./', parent) if x.startswith('./') else x for x in
                                       t.read().strip().splitlines()]
            else:
                raise Exception(f'{prefix}{p} does not exist')

        # 过滤有效图像文件（支持格式：IMG_FORMATS + txt列表格式）
        self.img_files = sorted([x.replace('/', os.sep) for x in self.img_files
                                 if x.split('.')[-1].lower() in IMG_FORMATS + ['txt']])

        # 解析标签文件路径（支持带标签路径的txt列表或同名标签文件）
        if self.img_files and self.img_files[0].endswith('.txt') and len(self.img_files[0].split(' ')) == 2:
            # 解析包含图像和标签路径的txt列表（格式：image_path label_path）
            self.img_files, self.label_files = zip(*[line.split(' ', 1) for line in self.img_files])
            self.img_files = list(self.img_files)
            self.label_files = list(self.label_files)
        else:
            # 常规模式：标签文件与图像同名，位于同目录（img2label_paths函数自动映射）
            self.label_files = img2label_paths(self.img_files)

        # 数据缓存检查（加载或生成.cache文件）
        cache_path = (Path(self.label_files[0]).parent if self.label_files else Path('data.cache')).with_suffix(
            '.cache')
        try:
            # 加载现有缓存文件
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            if cfg and cfg.check_datacache:
                # 验证缓存版本和数据哈希（确保缓存与当前数据一致）
                assert cache['version'] == self.cache_version, "Cache version mismatch"
                assert cache['hash'] == get_hash(self.label_files + self.img_files), "Data hash mismatch"
        except:
            # 生成新缓存文件
            cache, exists = self.cache_labels(cache_path, prefix), False

        # 解析缓存结果（统计有效/缺失/空标签等）
        nf, nm, ne, nc, n = cache.pop('results')  # 找到/缺失/空标签/损坏图像/总数
        if exists:
            # 打印缓存信息（通过tqdm显示进度）
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # 打印缓存生成过程中的警告信息
        assert nf > 0 or not augment, f'{prefix}No labels found. Cannot train without labels.'

        # 从缓存中提取标签、图像尺寸和分割段
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # 移除无关缓存项
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)

        # 分离ID信息（若存在跟踪ID）
        if self.with_id:
            self.ids = [l[:, 5] for l in self.labels]  # 提取ID列
            self.labels = [l[:, :5] for l in self.labels]  # 保留类别和边界框
        else:
            if self.num_points == 0:
                self.labels = [l[:, :5] for l in self.labels]  # 无关键点时仅保留前5列

        self.shapes = np.array(shapes, dtype=np.float64)  # 图像原始尺寸（宽, 高）
        self.img_files = list(cache.keys())  # 更新图像文件列表（匹配缓存中的有效项）
        self.label_files = img2label_paths(self.img_files)  # 更新标签文件路径
        n = len(shapes)  # 有效图像总数
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # 计算每个图像所属的批次索引
        nb = bi[-1] + 1  # 总批次数
        self.batch = bi  # 图像-批次索引映射
        self.n = n
        self.indices = range(n)  # 图像索引列表

        # 过滤指定类别（single_cls或include_class）
        self.filter_include_class(single_cls)

        # 计算类别分布（用于日志输出）
        if self.nc:
            cls_tmp = np.zeros(self.nc)
            for label in self.labels:
                for l in label:
                    cls_tmp[int(l[0])] += 1  # 统计每个类别的标签数量
            self.cls_ratio_gt = cls_tmp / np.sum(cls_tmp)  # 类别比例
            self.label_num_per_image = np.sum(cls_tmp) / len(self.img_files)  # 平均每图像标签数
            info = ' '.join([f'({v:.2f}-{i})' for i, v in enumerate(cls_tmp)])
            logging.info(f'cls gt ratio(positive): {info}')
            logging.info(f'cls gt total number: {np.sum(cls_tmp)} label number per image: {self.label_num_per_image}')

        # 矩形训练处理（按宽高比排序，减少填充）
        if self.rect:
            s = self.shapes  # 图像尺寸数组（宽, 高）
            ar = s[:, 1] / s[:, 0]  # 宽高比（w/h）
            irect = ar.argsort()  # 按宽高比排序的索引
            # 按排序后的索引重新排列数据
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            if self.with_id:
                self.ids = [self.ids[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            # 计算每个批次的目标尺寸（保持宽高比，对齐步长）
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]  # 当前批次内的宽高比列表
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:  # 宽高比 < 1（高>宽）
                    shapes[i] = [maxi, 1]
                elif mini > 1:  # 宽高比 > 1（宽>高）
                    shapes[i] = [1, 1 / mini]
            # 计算批次形状（向上取整到stride倍数）
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # 图像缓存（内存或磁盘，提升训练速度）
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                # 磁盘缓存：将图像保存为.npy文件
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                print('self im cache dir:', self.im_cache_dir)
            gb = 0  # 缓存占用的内存/磁盘空间（GB）
            # 使用线程池并行加载图像
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])  # 保存图像为.npy
                    gb += self.img_npy[i].stat().st_size  # 累加磁盘占用
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # 内存缓存图像和尺寸
                    gb += self.imgs[i].nbytes  # 累加内存占用
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def filter_include_class(self, single_cls):
        """
        过滤包含的类别或合并为单类别训练

        参数:
            single_cls: 是否合并为单类别（合并后所有类别ID设为0）

        处理逻辑:
            1. 根据include_class配置过滤目标类别
            2. single_cls为True时强制将所有类别转换为0
            3. 同时处理带ID和不带ID的标签格式
        """
        include_class_array = np.array(self.include_class).reshape(1, -1)
        if self.with_id:
            # 处理带跟踪ID的标签
            for i, (label, segment, ids) in enumerate(zip(self.labels, self.segments, self.ids)):
                if self.include_class:
                    # 筛选属于include_class的标签
                    j = (label[:, 0:1] == include_class_array).any(1)
                    self.labels[i] = label[j]
                    self.ids[i] = ids[j]
                if single_cls:  # 单类别训练，合并所有类别为0
                    self.labels[i][:, 0] = 0
        else:
            # 处理常规标签（无ID）
            for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
                if self.include_class:
                    # 筛选属于include_class的标签
                    j = (label[:, 0:1] == include_class_array).any(1)
                    self.labels[i] = label[j]
                if single_cls:  # 单类别训练，合并所有类别为0
                    self.labels[i][:, 0] = 0

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """
        缓存数据集标签信息，验证图像和标签有效性

        参数:
            path: 缓存文件路径（默认'./labels.cache'）
            prefix: 日志前缀

        返回:
            x: 缓存字典（包含标签、图像尺寸、验证统计结果）

        处理逻辑:
            1. 使用多线程池并行验证图像和标签
            2. 统计缺失标签、空标签、损坏图像数量
            3. 保存缓存文件包含标签、图像尺寸、验证结果和哈希值
            4. 支持带ID和不带ID的标签验证逻辑
        """
        x = {}  # 缓存字典
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 缺失/找到/空标签/损坏图像数, 警告消息
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        if self.with_id:
            id_list = cal_cur_max_id(self.label_files, self.pseudo_ids)
            print('number of label files:', len(self.label_files))
            print('id_index:', len(id_list))
        with Pool(NUM_THREADS) as pool:
            if self.with_id:
                pbar = tqdm(pool.imap(verify_image_label_with_id, zip(self.img_files, self.label_files, id_list, repeat(prefix))),
                            desc=desc, total=len(self.img_files))
            else:
                pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix), repeat(self.num_points))),
                            desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))  # 打印警告消息
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')

        # 填充缓存字典
        x['hash'] = get_hash(self.label_files + self.img_files)  # 数据哈希值（用于验证一致性）
        x['results'] = nf, nm, ne, nc, len(self.img_files)  # 统计结果元组
        x['msgs'] = msgs  # 警告消息列表
        x['version'] = self.cache_version  # 缓存版本号

        try:
            np.save(path, x)  # 保存缓存文件
            # 重命名文件（移除.npy后缀）
            path.with_suffix('.cache.npy').rename(path)
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # 处理写入错误
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        获取数据集中指定索引的样本（支持数据增强和批量处理）

        参数:
            index: 样本索引（支持线性索引、随机索引或图像权重索引）

        返回:
            img: 预处理后的图像张量（CHW格式，RGB通道）
            labels_out: 归一化后的标签张量
            path: 图像文件路径
            shapes: 原始图像尺寸和变换参数（用于COCO mAP计算）

        处理流程:
            1. 索引处理：支持线性、随机或图像权重采样
            2. 马赛克增强：随机混合4张图像（训练阶段）
            3. 单图像加载：尺寸调整、Letterbox变换
            4. 标签坐标转换：归一化边界框和关键点坐标
            5. 数据增强：随机透视、翻转、HSV变换等
            6. 格式转换：图像转张量，标签转归一化格式
        """
        index = self.indices[index]  # 根据采样策略获取真实索引（线性/随机/图像权重）
        hyp = self.hyp  # 数据增强超参数
        # 马赛克增强（概率由超参数控制）
        mosaic = self.mosaic and random.random() < hyp.mosaic
        if mosaic:
            # 加载马赛克图像（4张图像混合）
            img, labels = load_mosaic(self, index, self.num_points)
            shapes = None  # 马赛克图像无原始尺寸信息

            # MixUp增强（叠加另一张马赛克图像）
            if random.random() < hyp.mixup:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1), self.num_points))
        else:
            # 加载单张图像
            img, (h0, w0), (h, w) = load_image(self, index)  # 原始图像、原始尺寸、调整后尺寸

            # Letterbox变换（固定尺寸或矩形训练）
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # 目标尺寸
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)  # 缩放+填充
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # 保存尺寸信息用于COCO mAP计算

            # 标签处理（边界框和关键点坐标转换）
            labels = self.labels[index].copy()
            if labels.size:
                if self.with_id:  # 带跟踪ID的标签处理
                    # 边界框归一化坐标转像素坐标
                    labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    id_value = self.ids[index].copy().reshape(-1, 1)  # 获取ID并调整形状
                    labels = np.concatenate((labels, id_value), 1)  # 合并ID到标签
                else:
                    if labels.shape[-1] == 5 and self.num_points > 0:  # 仅边界框，需填充关键点
                        labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w, ratio[1] * h, padw=pad[0],
                                                    padh=pad[1])
                        # 填充无效关键点（标记为-1）
                        labels_lmk_pad = np.ones((labels.shape[0], self.num_points * 2)) * -1
                        labels = np.concatenate((labels, labels_lmk_pad), 1)
                    elif labels.shape[-1] > 5 and self.num_points > 0:  # 已有关键点标签
                        non_valid_index = (labels[:, 5:].sum(1) == 0)  # 标记全零关键点（无效）
                        labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w, ratio[1] * h, padw=pad[0],
                                                    padh=pad[1])
                        # 逐个转换关键点坐标（xyn转xy）
                        for n in range(self.num_points):
                            start = 5 + n * 2
                            end = start + 2
                            labels[:, start:end] = xyn2xy_new(labels[:, start:end], ratio[0] * w, ratio[1] * h,
                                                              padw=pad[0], padh=pad[1])
                        labels[non_valid_index, 5:] = -1  # 重置无效关键点为-1
                    elif labels.shape[-1] == 5 and self.num_points == 0:  # 仅边界框
                        labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w, ratio[1] * h, padw=pad[0],
                                                    padh=pad[1])

        # 数据增强（非马赛克模式下执行）
        if self.augment and not mosaic:
            img, labels = random_perspective_keypoints(img, labels,
                                                       degrees=hyp['degrees'],  # 旋转角度范围
                                                       translate=hyp['translate'],  # 平移比例
                                                       scale=hyp['scale'],  # 缩放比例
                                                       shear=hyp['shear'],  # 剪切角度
                                                       perspective=hyp['perspective'],  # 透视强度
                                                       num_points=self.num_points)  # 关键点数量

        nl = len(labels)  # 有效标签数量
        ori_labels = labels.copy()  # 保存原始标签（用于关键点无效恢复）
        if nl:
            # 边界框坐标归一化（xyxy转xywhn）
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True)
            # 关键点坐标归一化（除以图像尺寸）
            for n in range(5, 5 + self.num_points * 2):
                if n % 2 == 0:  # 关键点y坐标（归一化到0-1）
                    labels[:, n] /= img.shape[0]
                else:  # 关键点x坐标（归一化到0-1）
                    labels[:, n] /= img.shape[1]

        # Albumentations增强（随机模糊、灰度等）
        if self.augment:
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # 更新增强后的标签数量

            # HSV颜色空间增强
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # 上下翻转
            if random.random() < hyp['flipud']:
                img = np.flipud(img)  # 翻转图像
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]  # 翻转边界框中心y坐标
                    # 翻转关键点y坐标
                    for n in range(5, 5 + self.num_points * 2):
                        if n % 2 == 0:
                            labels[:, n] = 1 - labels[:, n]

            # 左右翻转
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)  # 翻转图像
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]  # 翻转边界框中心x坐标
                    # 翻转关键点x坐标（根据关键点顺序调整）
                    if self.num_points == 4:
                        # 交换左右关键点（如左眼<->右眼）
                        labels = np.concatenate(
                            [labels[:, :5], labels[:, 7:9], labels[:, 5:7], labels[:, 11:13], labels[:, 9:11]], axis=1)
                    elif self.num_points == 8:
                        labels = np.concatenate(
                            [labels[:, :5], labels[:, 7:9], labels[:, 5:7], labels[:, 11:13], labels[:, 9:11],
                             labels[:, 15:17], labels[:, 13:15], labels[:, 19:21], labels[:, 17:19]], axis=1)
            # 恢复无效关键点（增强后可能丢失的原始无效标签）
            if self.num_points > 0 and nl:
                try:
                    non_valid_index = ((ori_labels[:, 5:] == -1).sum(1) == self.num_points * 2)
                    labels[non_valid_index, 5:] = ori_labels[non_valid_index, 5:]
                except IndexError:
                    print('ori_labels:', ori_labels)
                    print(ori_labels[:, 5:])
                    print('non_valid_index:', self.img_files[index])

        # 初始化标签输出张量
        if self.with_id:
            labels_out = torch.zeros((nl, 7))  # 带ID标签：cls, xywhn, id
        else:
            labels_out = torch.zeros((nl, 6 + self.num_points * 2))  # 常规标签：cls, xywhn, keypoints

        if nl:
            path = self.img_files[index]
            if self.with_id:
                labels[:, 1:5] = labels[:, 1:5].clip(0, 1.0)  # 边界框坐标裁剪到[0,1]
            else:
                labels[:, 1:] = labels[:, 1:].clip(0, 1.0)  # 边界框和关键点坐标裁剪到[0,1]
            labels_out[:, 1:] = torch.from_numpy(labels)  # 转换为PyTorch张量

            if self.debug:  # 调试模式：可视化标签
                if self.num_points > 0:
                    self.showlabels(img, labels[:, 1:5], [], labels[:, 5:5 + self.num_points * 2], path)
                else:
                    if self.with_id:
                        self.showlabels(img, labels[:, 1:5], labels[:, 5:], [], path)
                    else:
                        self.showlabels(img, labels[:, 1:5], [], [], path)

        # 图像格式转换（HWC -> CHW，BGR -> RGB）
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    def order_points_quadrangle(self, pts_sample):
        """
        车牌标注点（四边形顶点）排序函数（按左上、右上、右下、左下顺序排列）

        参数:
            pts_sample: 输入标注点数组，形状为[N, 8]（N为样本数，每个样本8个坐标值x1y1x2y2x3y3x4y4）

        返回:
            new_sample: 排序后的标注点数组，形状为[N, 8]，顺序为左上、右上、右下、左下

        排序逻辑:
            1. 按x坐标排序，将4个顶点分为左半部分（x较小的2点）和右半部分（x较大的2点）
            2. 左半部分按y坐标排序（y小的为左上，y大的为左下）
               - 若y坐标相同，则按x坐标逆序排序（确保左边界正确）
            3. 右半部分按y坐标排序（y小的为右上，y大的为右下）
               - 若y坐标相同，则按x坐标逆序排序（确保右边界正确）
            4. 组合排序后的顶点顺序：左上->右上->右下->左下
        """
        new_sample = []
        for pts in pts_sample:
            pts = np.reshape(pts, (4, 2))  # 转换为4个顶点坐标的数组[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

            # 按x坐标升序排序，分离左半部分（x较小的2点）和右半部分（x较大的2点）
            xSorted = pts[np.argsort(pts[:, 0]), :]
            leftMost = xSorted[:2, :]  # 左半部分（x最小的两个点）
            rightMost = xSorted[2:, :]  # 右半部分（x最大的两个点）

            # 左半部分排序：优先按y升序（上->下），y相同则按x降序（左->右）
            if leftMost[0, 1] != leftMost[1, 1]:
                leftMost = leftMost[np.argsort(leftMost[:, 1]), :]  # y小的为左上，y大的为左下
            else:
                leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]  # y相同则x大的为左上（处理水平共线情况）
            (tl, bl) = leftMost  # 左上(top-left), 左下(bottom-left)

            # 右半部分排序：优先按y升序（上->下），y相同则按x降序（右->左）
            if rightMost[0, 1] != rightMost[1, 1]:
                rightMost = rightMost[np.argsort(rightMost[:, 1]), :]  # y小的为右上，y大的为右下
            else:
                rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]  # y相同则x小的为右上（处理水平共线情况）
            (tr, br) = rightMost  # 右上(top-right), 右下(bottom-right)

            # 组合为左上->右上->右下->左下的顺序（对应标注点顺序：x1y1, x2y2, x3y3, x4y4）
            new_sample.append([tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]])

        return np.array(new_sample)

    def up_order_points_quadrangle_new(self, pts_sample):
        """
        四边形顶点重排序函数（基于中心点极角逆时针排序）

        参数:
            pts_sample: 输入顶点数组，形状为[N, 8]（每个样本4个顶点坐标x1y1x2y2x3y3x4y4）

        返回:
            new_sample: 排序后的顶点数组，形状为[N, 8]，按逆时针顺序排列（基于中心点极角）

        处理逻辑:
            1. 对每个样本的4个顶点：
               a. 计算几何中心点坐标(center_pt_x, center_pt_y)
               b. 计算各顶点相对于中心点的极角theta（atan2(y, x)）
               c. 根据极角从小到大排序顶点（逆时针顺序）
            2. 按排序后的顺序重组顶点坐标为x1y1x2y2x3y3x4y4格式
            3. 适用于需要按固定顺序（如逆时针）排列四边形顶点的场景（如车牌标注）
        """
        new_sample = []
        for pts in pts_sample:
            pts = np.reshape(pts, (4, 2))  # 转换为[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]格式
            # 计算四边形几何中心点
            center_pt_x = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4
            center_pt_y = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4

            d2s = []  # 存储(顶点坐标, 极角)列表
            for i in range(pts.shape[0]):
                vector_x = pts[i][0] - center_pt_x  # 顶点相对于中心点的x偏移
                vector_y = pts[i][1] - center_pt_y  # 顶点相对于中心点的y偏移
                theta = np.arctan2(vector_y, vector_x)  # 计算极角（弧度制，范围[-π, π]）
                d2s.append([pts[i], theta])  # 保存顶点坐标和对应的极角

            # 按极角从小到大排序（逆时针顺序）
            d2s = sorted(d2s, key=lambda x: x[1])
            # 提取排序后的顶点坐标，重组为x1y1x2y2x3y3x4y4格式
            tmp = [
                d2s[0][0][0], d2s[0][0][1],  # 第一个顶点（极角最小，通常为左上）
                d2s[1][0][0], d2s[1][0][1],  # 第二个顶点（逆时针下一个）
                d2s[2][0][0], d2s[2][0][1],  # 第三个顶点（逆时针下一个）
                d2s[3][0][0], d2s[3][0][1]  # 第四个顶点（极角最大，通常为左下）
            ]
            new_sample.append(tmp)
        return np.array(new_sample)

    def up_order_points_quadrangle(self, pts_sample):
        """
        四边形顶点排序函数（按上下-左右顺序排列）

        参数:
            pts_sample: 输入顶点数组，形状为[N, 8]（每个样本4个顶点坐标x1y1x2y2x3y3x4y4）

        返回:
            new_sample: 排序后的顶点数组，形状为[N, 8]，顺序为左上、右上、右下、左下

        排序逻辑:
            1. 按y坐标升序排序，分离上半部分（y较小的2点）和下半部分（y较大的2点）
            2. 上半部分按x坐标升序排序（左->右），确定左上、右上顶点
            3. 下半部分按x坐标降序排序（右->左），确定右下、左下顶点
            4. 处理x坐标相同的情况，通过逆序确保顺序一致性
        """
        new_sample = []
        for pts in pts_sample:
            pts = np.reshape(pts, (4, 2))  # 转换为[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]格式

            # 按y坐标升序排序，分离上下部分顶点
            xSorted = pts[np.argsort(pts[:, 1]), :]
            upMost = xSorted[:2, :]  # 上半部分（y较小的两个顶点）
            bottomMost = xSorted[2:, :]  # 下半部分（y较大的两个顶点）

            # 上半部分排序：按x升序确定左上、右上
            if upMost[0, 0] != upMost[1, 0]:
                upMost = upMost[np.argsort(upMost[:, 0]), :]  # x小的为左上，x大的为右上
            else:
                upMost = upMost[np.argsort(upMost[:, 0])[::-1], :]  # x相同则逆序，保持顺序一致
            (tl, tr) = upMost  # 左上(top-left), 右上(top-right)

            # 下半部分排序：按x降序确定右下、左下
            if bottomMost[0, 0] != bottomMost[1, 0]:
                bottomMost = bottomMost[np.argsort(bottomMost[:, 0]), :]  # x小的为左下，x大的为右下
            else:
                bottomMost = bottomMost[np.argsort(bottomMost[:, 0])[::-1], :]  # x相同则逆序，保持顺序一致
            (bl, br) = bottomMost  # 左下(bottom-left), 右下(bottom-right)

            # 组合为左上->右上->右下->左下的顺序
            new_sample.append([tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]])
        return np.array(new_sample)

    def showlabels(self, img, boxs, ids, landmarks, path):
        """
        可视化标签函数（绘制边界框、关键点和ID）

        参数:
            img: 输入图像（numpy数组）
            boxs: 边界框坐标（归一化xywh，形状[N,4]）
            ids: 跟踪ID列表（形状[N,]）
            landmarks: 关键点坐标（归一化xy，形状[N, 2*num_points]）
            path: 图像路径（用于保存可视化结果）

        功能:
            1. 将归一化边界框转换为图像像素坐标并绘制绿色矩形
            2. 在边界框左上角绘制跟踪ID文本
            3. 绘制关键点（红色圆圈）并标注序号
            4. 保存可视化结果到临时目录
        """
        img = img.astype(np.uint8)
        for i, box in enumerate(boxs):
            # 边界框坐标转换（归一化xywh转像素xyxy）
            x, y, w, h = box[0] * img.shape[1], box[1] * img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0),
                          2)  # 绘制矩形框
            # 绘制ID文本
            cv2.putText(img, str(int(ids[i])), (int(x - w / 2) - 10, int(y - h / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 120 / 150, (0, 255, 0), round(240 / 150))

        if landmarks != []:
            for landmark in landmarks:
                # 绘制关键点（红色圆圈+序号文本）
                for i in range(8):  # 假设最多8个关键点（根据实际num_points调整）
                    cx = int(landmark[2 * i] * img.shape[1])
                    cy = int(landmark[2 * i + 1] * img.shape[0])
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)  # 绘制圆圈
                    cv2.putText(img, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 120 / 150, (0, 255, 0),
                                round(240 / 150))  # 绘制序号

        # 保存可视化结果
        cv2.imwrite('tmp/' + path.split('/')[-1], img)

    @staticmethod
    def collate_fn(batch):
        """
        数据整理函数（用于DataLoader批量处理）

        参数:
            batch: 包含(img, label, path, shapes)的样本列表

        返回:
            imgs: 堆叠后的图像张量（形状[B, C, H, W]）
            labels: 拼接后的标签张量（形状[total_labels, ...]）
            paths: 图像路径列表
            shapes: 原始图像尺寸列表

        功能:
            1. 解压缩批次数据
            2. 为每个标签添加对应的图像索引（用于多图像目标匹配）
            3. 堆叠图像为批次张量，拼接标签为统一张量
        """
        img, label, path, shapes = zip(*batch)  # 解压缩批次数据
        for i, l in enumerate(label):
            l[:, 0] = i  # 添加图像索引到标签第一列，用于build_targets函数匹配
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        """
        四合一图像数据整理函数（用于训练时将四张图像合并为一张）

        参数:
            batch: 包含(img, label, path, shapes)的样本列表，长度应为4的倍数

        返回:
            img4: 合并后的图像张量列表（形状[B, C, H, W]）
            label4: 调整后的标签张量列表（形状[total_labels, ...]）
            path4: 第一张原始图像的路径列表
            shapes4: 第一张原始图像的尺寸列表

        功能:
            1. 随机选择处理方式：
               - 50%概率：将第一张图像放大2倍作为输出
               - 50%概率：将四张图像按2×2网格拼接成一张大图像
            2. 调整标签坐标以匹配新的图像布局
            3. 为每个标签添加对应的图像索引（用于多图像目标匹配）
        """
        img, label, path, shapes = zip(*batch)  # 解压缩批次数据
        n = len(shapes) // 4  # 批次中的四合一图像组数

        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]  # 初始化输出列表

        # 定义坐标调整参数（用于标签坐标转换）
        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])  # y方向偏移量（第二行图像）
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])  # x方向偏移量（第二列图像）
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # 缩放因子（拼接后图像尺寸变为原来一半）

        for i in range(n):  # 处理每组四张图像
            i *= 4  # 当前组的起始索引

            # 随机选择处理方式
            if random.random() < 0.5:
                # 方式一：将第一张图像放大2倍
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]  # 标签保持不变
            else:
                # 方式二：四张图像按2×2网格拼接
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)

                # 调整标签坐标以匹配新的图像布局
                l = torch.cat((
                    label[i],  # 第一张图像（左上）标签不变
                    label[i + 1] + ho,  # 第二张图像（右上）标签y坐标偏移
                    label[i + 2] + wo,  # 第三张图像（左下）标签x坐标偏移
                    label[i + 3] + ho + wo  # 第四张图像（右下）标签x、y坐标均偏移
                ), 0) * s  # 所有标签坐标缩放（因图像尺寸变为原来一半）

            img4.append(im)
            label4.append(l)

        # 为每个标签添加图像索引（用于build_targets函数）
        for i, l in enumerate(label4):
            l[:, 0] = i

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', cfg=None):
    """
    创建数据加载器，支持分布式训练和多种采样策略

    参数:
        path: 数据集路径
        imgsz: 图像尺寸
        batch_size: 批次大小
        stride: 网络步长
        single_cls: 是否合并为单类别
        hyp: 超参数配置
        augment: 是否进行数据增强
        cache: 是否缓存图像
        pad: 矩形训练填充
        rect: 是否使用矩形训练
        rank: 分布式训练进程ID
        workers: 数据加载工作线程数
        image_weights: 是否使用图像权重采样
        quad: 是否使用四图像批处理
        prefix: 日志前缀
        cfg: 配置对象

    返回:
        dataloader: 数据加载器
        dataset: 数据集对象

    处理逻辑:
        1. 确保分布式训练中只有首个进程处理数据集
        2. 初始化数据集(LoadImagesAndLabels)
        3. 根据配置选择采样器类型:
           - normal: 普通分布式采样器
           - class_balance: 类别平衡采样器
           - dir_balance: 目录平衡采样器
        4. 根据是否使用图像权重选择数据加载器类型
        5. 配置数据加载器参数并返回
    """
    # 确保分布式训练中只有首个进程处理数据集，避免重复操作
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # 图像增强
                                      hyp=hyp,  # 增强超参数
                                      rect=rect,  # 矩形训练
                                      cache_images=cache,  # 缓存图像
                                      single_cls=single_cls,  # 单类别模式
                                      stride=int(stride),  # 步长对齐
                                      pad=pad,  # 填充
                                      image_weights=image_weights,  # 图像权重采样
                                      cfg=cfg,
                                      prefix=prefix)  # 日志前缀

    # 限制批次大小不超过数据集大小
    batch_size = min(batch_size, len(dataset))
    # 计算最佳工作线程数(不超过CPU核心数、批次大小和设定上限)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # 工作线程数

    # 训练集采样器配置
    if 'train' in prefix:
        print('world_size:', WORLD_SIZE)
        print('rank:', rank)
        # 根据配置选择采样器类型
        if cfg.Dataset.sampler_type == 'normal':
            # 普通分布式采样器
            sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
        elif cfg.Dataset.sampler_type == 'class_balance':
            # 类别平衡采样器(分布式或单机)
            print('use banlanced batch sampler')
            sampler = DistributeBalancedBatchSampler(dataset, WORLD_SIZE, rank,
                                                     'class_balance') if rank != -1 else BalancedBatchSampler(dataset)
        elif cfg.Dataset.sampler_type == 'dir_balance':
            # 目录平衡采样器(分布式或单机)
            sampler = DistributeBalancedBatchSampler(dataset, WORLD_SIZE, rank,
                                                     'dir_balance') if rank != -1 else BalancedBatchSampler(dataset)
        else:
            assert NotImplementedError
    # 验证集/测试集采样器配置
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None

    # 根据是否使用图像权重选择数据加载器类型
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # 使用普通DataLoader或InfiniteDataLoader(支持训练过程中更新数据集属性)

    # 创建数据加载器
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,  # 锁页内存，加速数据传输
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)  # 四图像批处理或普通批处理
    return dataloader, dataset

class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """♾️ 改进型无限数据加载器，实现worker进程复用

    核心改进:
        - 继承原生DataLoader并重写迭代逻辑
        - 通过_RepeatSampler实现批次采样无限循环
        - 解决多epoch场景下worker反复初始化的性能损耗

    优势:
        ▶ 保持与原生DataLoader完全兼容的API
        ▶ 每个epoch自动重置迭代器(无需手动重新初始化)
        ▶ 通过固定worker进程大幅提升数据加载效率

    实现细节:
        1. 用_RepeatSampler包装原始batch_sampler
        2. __iter__通过计数控制yield次数，与epoch对齐
        3. 每次__iter__调用复用预初始化的iterator

    注意:
        - 实际迭代次数由__len__控制(等于数据集长度)
        - 需配合训练循环的epoch机制使用
        - 迭代器状态持续累积，需警惕内存泄漏风险
    """

    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)  # 基于原始采样器长度

    def __iter__(self):
        for i in range(len(self)):  # 按数据集长度控制迭代次数
            yield next(self.iterator)  # 复用预存迭代器

class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """♾️ 无限循环采样器包装类，用于持续生成数据索引

    功能:
        - 包装原始采样器实现无限循环迭代
        - 解决多epoch训练时的数据重复供给需求

    参数:
        sampler: 被包装的原始采样器对象(torch.utils.data.Sampler)

    核心机制:
        - 通过__iter__魔术方法实现无限循环
        - 使用yield from保持原始采样顺序

    典型应用:
        - 配合PyTorch的DataLoader使用
        - 多进程数据加载场景(防止worker提前耗尽数据)

    注意:
        - 迭代器不会自动终止，需外部控制训练轮数
        - 被包装采样器需实现__iter__方法
        - 可能造成内存累积(若采样器含状态)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:  # 无限循环核心逻辑
            yield from iter(self.sampler)  # 保持原始采样顺序


class LoadImages:
    """
    YOLOv5图像/视频数据加载器（用于推理场景，如`python detect.py --source image.jpg/vid.mp4`）

    功能:
        1. 支持多种输入源：单个文件、目录、通配符路径、txt列表文件
        2. 自动区分图像和视频文件
        3. 提供迭代器接口，逐个返回处理后的图像/视频帧
        4. 包含图像预处理（尺寸调整、格式转换）功能

    参数:
        path: 输入路径（支持文件/目录/通配符/*.jpg或txt列表文件）
        img_size: 目标图像尺寸（默认640）
        stride: 网络步长（用于尺寸对齐，默认32）
        auto: 是否自动计算最小填充（默认True）
    """

    def __init__(self, path, img_size=640, stride=32, auto=True):
        """初始化数据加载器，解析输入路径并分类文件类型"""
        if path.endswith('.txt'):
            # 处理txt列表文件（每行一个文件路径，支持带标签的路径格式）
            with open(path, 'r') as f:
                allinfo = f.readlines()
                allinfo = [a.strip() for a in allinfo]
            if len(allinfo[0].split(' ')) == 2:  # 兼容带标签的路径格式
                allinfo = [a.split(' ')[0] for a in allinfo]
            files = allinfo
        else:
            # 处理常规文件路径（支持绝对路径、通配符、目录）
            p = str(Path(path).resolve())  # 获取绝对路径
            if '*' in p:
                files = sorted(glob.glob(p, recursive=True))  # 通配符匹配所有文件
            elif os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # 遍历目录下所有文件
            elif os.path.isfile(p):
                files = [p]  # 单个文件
            else:
                raise Exception(f'ERROR: {p} does not exist')

        # 分离图像和视频文件
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)  # 图像/视频文件数量

        self.img_size = img_size  # 目标尺寸
        self.stride = stride  # 步长（用于尺寸对齐）
        self.files = images + videos  # 所有文件路径列表
        self.nf = ni + nv  # 文件总数
        self.video_flag = [False] * ni + [True] * nv  # 标记是否为视频文件
        self.mode = 'image'  # 当前模式（image/video）
        self.auto = auto  # 是否自动计算填充

        # 初始化首个视频文件（如果有）
        if any(videos):
            self.new_video(videos[0])  # 新建视频捕获对象
        else:
            self.cap = None  # 无视频时设为None

        # 检查文件是否存在
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        """返回迭代器对象，初始化计数"""
        self.count = 0
        return self

    def __next__(self):
        """逐个返回处理后的文件路径、图像/帧、原始图像/帧、视频捕获对象"""
        if self.count == self.nf:
            raise StopIteration  # 遍历结束

        path = self.files[self.count]

        if self.video_flag[self.count]:
            # 处理视频文件
            self.mode = 'video'
            ret_val, img0 = self.cap.read()  # 读取视频帧
            if not ret_val:  # 当前视频结束
                self.count += 1
                self.cap.release()  # 释放资源
                if self.count == self.nf:  # 所有文件处理完毕
                    raise StopIteration
                else:  # 切换到下一个文件
                    path = self.files[self.count]
                    self.new_video(path)  # 初始化新视频
                    ret_val, img0 = self.cap.read()
            self.frame += 1  # 视频帧计数
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')
        else:
            # 处理图像文件
            self.count += 1
            img0 = cv2.imread(path)  # BGR格式读取
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # 图像预处理
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]  # 缩放+填充
        img = img.transpose((2, 0, 1))[::-1]  # HWC转CHW，BGR转RGB
        img = np.ascontiguousarray(img)  # 保证内存连续

        return path, img, img0, self.cap  # 返回路径、预处理后图像、原始图像、视频捕获对象

    def new_video(self, path):
        """初始化视频捕获对象（处理新视频文件）"""
        self.frame = 0  # 帧计数器重置
        self.cap = cv2.VideoCapture(path)  # 打开视频文件
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

    def __len__(self):
        """返回文件总数"""
        return self.nf  # 图像和视频文件总数

class LoadStreams:
    """
    YOLOv5视频流加载器（支持RTSP/RTMP/HTTP等实时流，如`python detect.py --source 'rtsp://example.com/media.mp4'`）

    功能:
        1. 支持多视频流并行加载（通过线程池实现）
        2. 自动处理YouTube视频链接（需额外依赖pafy/youtube_dl）
        3. 实时获取视频流帧并进行预处理
        4. 支持矩形推理（根据流形状自动优化）

    参数:
        sources: 视频流源（文件路径或URL列表，支持txt文件每行一个源）
        img_size: 目标图像尺寸（默认640）
        stride: 网络步长（用于尺寸对齐，默认32）
        auto: 是否自动计算最小填充（默认True）
    """

    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'  # 当前模式标记
        self.img_size = img_size  # 目标尺寸
        self.stride = stride  # 步长（用于尺寸对齐）
        self.auto = auto  # 自动填充开关

        # 解析输入源（支持txt文件或直接URL）
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n  # 流状态存储
        self.sources = [clean_str(x) for x in sources]  # 清理源名称

        for i, s in enumerate(sources):  # 逐个初始化流
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:  # 处理YouTube链接
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # 获取最佳画质MP4流URL
            s = eval(s) if s.isnumeric() else s  # 处理本地摄像头（如'0'表示默认摄像头）
            cap = cv2.VideoCapture(s)  # 初始化视频捕获对象
            assert cap.isOpened(), f'Failed to open {s}'  # 确保流打开成功

            # 获取流元信息
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 帧率（默认30FPS）
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # 总帧数（无限流标记为inf）

            # 预读取第一帧
            _, self.imgs[i] = cap.read()
            # 启动后台更新线程
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # 换行

        # 检查流形状一致性（优化矩形推理）
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # 所有流形状一致时启用矩形推理
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        """后台线程函数：持续从视频流读取帧"""
        n, f, read = 0, self.frames[i], 1  # 帧计数器、总帧数、读取间隔
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # 高效获取帧（不立即解码）
            if n % read == 0:  # 按间隔读取（降低CPU占用）
                success, im = cap.retrieve()  # 解码并获取帧
                if success:
                    self.imgs[i] = im  # 更新最新帧
                else:
                    print('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] *= 0  # 标记为全黑帧
                    cap.open(stream)  # 重新打开流（处理断流恢复）
            time.sleep(1 / self.fps[i])  # 按帧率控制线程休眠时间

    def __iter__(self):
        """迭代器初始化：计数器重置"""
        self.count = -1
        return self

    def __next__(self):
        """获取下一帧批量数据"""
        self.count += 1
        # 检查线程存活状态或退出指令（按'q'键）
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration

        img0 = self.imgs.copy()  # 原始帧列表
        # 批量预处理：尺寸调整+填充（支持矩形推理）
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]
        # 堆叠为批量张量（BCHW格式）
        img = np.stack(img, 0)
        # 格式转换：BGR->RGB，BHWC->BCHW
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)  # 保证内存连续

        return self.sources, img, img0, None  # 返回源列表、预处理后图像、原始图像、空捕获对象

    def __len__(self):
        """返回流数量（模拟无限流，实际为源数量）"""
        return len(self.sources)  # 1E12帧 = 32流@30FPS运行30年（理论无限）

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    # sa, sb = os.sep + '/mnt/datasets/' + os.sep, os.sep + '/home/bowen/annotations/labels/lubiantingche_total_train/' + os.sep  # /images/, /labels/ substrings
    # sa, sb = '/mnt/bowen/', '/mnt/annotations/annotations/labels/lubiantingche_total_2in1/'
    # sa, sb = '/mnt/bowen/', '/mnt/bowen/'
    # sa, sb = '/mnt/datasets/', '/mnt/annotations/annotations/labels/xiaokongbao_total_train/'
    # sa, sb = '/mnt/bowen/exp/coco/images/val2017/', '/mnt/bowen/exp/coco/labels/val2017/'
    sa, sb = 'images', 'labels'
    # sa, sb = '/mnt/datasets/', '/home/bowen/annotations/labels/lubiantingche_total_test_badcase_2021_02_24/'
    # sa, sb = '/AIDATA/', '/mnt/annotations/annotations/labels/xiaokongbao_total_train/'
    # sa, sb = '/mnt/bowen/', '/mnt/annotations/annotations/labels/xiaokongbao_total_train/'
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]


def load_image(self, i):
    """
    加载数据集中指定索引的图像，支持内存缓存和磁盘读取

    参数:
        i: 数据集索引

    返回:
        im: 加载并预处理后的图像
        (h0, w0): 原始图像的高和宽
        (h, w): 调整尺寸后的图像高和宽

    处理逻辑:
        1. 优先从内存缓存中获取图像（self.imgs）
        2. 若无缓存则尝试加载.npy文件（用于加速重复读取）
        3. 最后通过OpenCV读取图像文件
        4. 按固定比例缩放图像（保持宽高比）
        5. 返回图像及其原始尺寸、调整后尺寸
    """
    im = self.imgs[i]
    if im is None:  # 图像未缓存在内存中
        npy = self.img_npy[i]
        if npy and npy.exists():  # 存在.npy缓存文件则加载
            im = np.load(npy)
        else:  # 从磁盘读取图像文件
            path = self.img_files[i]
            im = cv2.imread(path)  # 以BGR格式读取图像
            assert im is not None, 'Image Not Found ' + path  # 确保图像存在
        h0, w0 = im.shape[:2]  # 获取原始图像高宽
        r = self.img_size / max(h0, w0)  # 计算缩放比例（目标尺寸/原尺寸最大值）
        if r != 1:  # 需要调整尺寸时
            # 使用双线性插值调整图像大小（保持平滑）
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # 返回图像、原始尺寸、调整后尺寸
    else:
        # 直接返回缓存中的图像和尺寸信息
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]


def load_mosaic(self, index, num_points):
    """
    YOLOv5 4-马赛克数据增强加载器（将1张主图像和3张随机图像组合成一个四合一图像）

    参数:
        index: 主图像在数据集中的索引
        num_points: 关键点数量（用于关键点检测任务）

    返回:
        img4: 马赛克组合后的图像
        labels4: 组合后的标签（包含边界框和可选的关键点信息）

    处理逻辑:
        1. 随机选择马赛克中心坐标(xc, yc)
        2. 随机选择3张额外图像与主图像组合
        3. 将4张图像按不同位置拼接到2倍大小的画布上
        4. 根据图像拼接位置调整标签坐标
        5. 应用复制粘贴和随机透视变换增强
    """
    labels4, segments4 = [], []  # 初始化组合后的标签和分割段
    s = self.img_size  # 目标图像尺寸
    # 随机生成马赛克中心坐标（允许超出图像边界）
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
    indices = [index] + random.choices(self.indices, k=3)  # 选择3张额外图像
    random.shuffle(indices)  # 打乱图像顺序

    for i, index in enumerate(indices):
        # 加载图像及其尺寸信息
        img, _, (h, w) = load_image(self, index)

        # 根据当前图像在马赛克中的位置计算粘贴区域
        if i == 0:  # 左上
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 创建灰色背景画布
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # 大图区域坐标
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # 小图区域坐标
        elif i == 1:  # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将当前图像粘贴到马赛克画布的指定位置
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw, padh = x1a - x1b, y1a - y1b  # 计算填充偏移量

        # 处理标签数据
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if self.with_id:  # 处理带跟踪ID的标签
            if labels.size:
                labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w, h, padw, padh)  # 归一化坐标转像素坐标
                id_value = self.ids[index].copy().reshape(-1, 1)  # 获取跟踪ID
                labels = np.concatenate((labels, id_value), 1)  # 合并边界框和ID信息
            else:
                labels = np.array([]).reshape(0, 6)  # 空标签处理
        else:
            if labels.size:
                if labels.shape[-1] == 5 and num_points > 0:  # 仅边界框标签且需要关键点
                    labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w, h, padw, padh)
                    # 填充未使用的关键点位置为-1（表示无效）
                    labels_lmk_pad = np.ones((labels.shape[0], num_points * 2)) * -1
                    labels = np.concatenate((labels, labels_lmk_pad), 1)
                elif labels.shape[-1] > 5 and num_points > 0:  # 已有关键点标签
                    non_valid_index = (labels[:, 5:].sum(1) == 0)  # 标记无效关键点
                    labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w, h, padw, padh)
                    # 转换关键点坐标
                    for n in range(5, 5 + num_points * 2, 2):
                        labels[:, n:n + 2] = xyn2xy_new(labels[:, n:n + 2], w, h, padw, padh)
                    # 重置无效关键点
                    labels[non_valid_index, 5:] = np.ones((non_valid_index.sum(), num_points * 2)) * -1
                elif labels.shape[-1] == 5 and num_points == 0:  # 仅边界框且无需关键点
                    labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w, h, padw, padh)
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]  # 转换分割段坐标
        labels4.append(labels)  # 收集当前图像标签
        segments4.extend(segments)  # 收集当前图像分割段

    # 合并所有标签
    try:
        labels4 = np.concatenate(labels4, 0)
    except ValueError:
        print(labels4)  # 处理标签合并失败的情况

    # 应用数据增强
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp.copy_paste)  # 复制粘贴增强
    img4, labels4 = random_perspective_keypoints(img4, labels4, segments4,
                                                 degrees=self.hyp.degrees,
                                                 translate=self.hyp.translate,
                                                 scale=self.hyp.scale,
                                                 shear=self.hyp.shear,
                                                 perspective=self.hyp.perspective,
                                                 num_points=num_points,
                                                 border=self.mosaic_border)  # 随机透视变换

    return img4, labels4  # 返回增强后的图像和标签

def cal_cur_max_id(lb_files, pseudo_labels=False):
    """
    计算当前最大目标ID，生成ID映射列表
    Args:
        lb_files: 标注文件路径列表
        pseudo_labels: 是否为伪标签模式(自动生成ID)

    Returns:
        id_list: 二维列表，每个文件对应的新ID集合

    核心逻辑：
    1. 按文件夹分组处理：当检测到文件路径中的父目录变化时，重置ID起始偏移
    2. 伪标签模式自动分配递增ID，真实标签模式解析原始ID
    3. 维护全局id_cur_max记录当前最大ID值，实现跨文件ID不重复
    4. 处理异常情况：文件不存在时返回[-1]，原始标注不完整时返回-1
    """
    id_start_index = 0
    id_list = []
    id_cur_max = -1
    id_key = ''
    pseudo_id = 0
    for lb_file in lb_files:
        if os.path.isfile(lb_file):
            id_key_tmp = lb_file.split('/')[-2]  # 提取父目录作为分组标识
            id_list_image = []
            if id_key_tmp != id_key:  # 检测到目录变化时重置ID基数
                id_key = id_key_tmp
                id_start_index = id_cur_max + 1  # 新组ID从当前全局最大ID+1开始
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                for x in l:
                    if pseudo_labels:  # 伪标签模式生成自增ID
                        id_new = pseudo_id + id_start_index
                        pseudo_id += 1
                    else:  # 真实标签模式解析原始ID
                        id_new = int(x[5]) + id_start_index if len(x) >=6 else -1
                    id_cur_max = max(id_new, id_cur_max)  # 更新全局最大ID
                    id_list_image.append(id_new)
            id_list.append(id_list_image)
        else:
            id_list.append([-1])  # 文件不存在时填充-1
    return id_list

def verify_image_label_with_id(args):
    """
        验证图像-标签对的有效性，处理跟踪ID的连续性问题
        Args:
            args: 包含(im_file, lb_file, id_list, prefix)的元组
                id_list: 由cal_cur_max_id生成的ID映射列表

        Returns:
            元组包含处理结果：图像路径、标签数组、图像尺寸、分割信息及各类状态统计

        核心流程：
        1. 图像验证：
            - 检查图像文件完整性，修复损坏的JPEG文件
            - 验证图像尺寸和格式要求
        2. 标签验证：
            - 结合id_list将原始标签转换为含连续ID的新格式
            - 处理标签异常：重复项、数值范围、标准化坐标
        3. ID处理：
            - 使用预先生成的id_list中的ID值
            - 保持跨文件夹的ID连续性
        4. 异常统计：
            - 记录缺失(nm)、找到(nf)、空(ne)、损坏(nc)等状态
     """
    im_file, lb_file, id_list, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    length_with_id = 6
    id_key = ''
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    Image.open(im_file).save(im_file, format='JPEG', subsampling=0, quality=100)  # re-save image
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        # 在不同文件夹下面的跟踪标注需要进行id的累加
        if os.path.isfile(lb_file):
            nf = 1  # label found
            # print(lb_file)
            # print('id_key_tmp', id_key_tmp)
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                classes = []
                box_segments = []
                id_segments = []
                for i, x in enumerate(l):
                    # print('x:', x)
                    classes.append(x[0])
                    box_segments.append(np.array(x[1:5], dtype=np.float32))
                    id_new = id_list[i]
                    id_segments.append(id_new)
                    # if len(x) == length_with_id:
                    #     classes.append(x[0])
                    #     box_segments.append(np.array(x[1:5], dtype=np.float32))
                    #     id_new = id_list[i]
                    #     id_segments.append(id_new)
                    # else: #标注缺少id信息，但又开启了nf选项
                    #     classes.append(x[0])
                    #     box_segments.append(np.array(x[1:5], dtype=np.float32))
                    #     id_segments.append(-1)
                # print(id_key_tmp, id_segments, id_cur_max)
                if len(l):
                    classes = np.array(classes, dtype=np.float32)
                    box_segments = np.array(box_segments)
                    id_segments = np.array(id_segments, dtype=np.float32)
                    l = np.concatenate((classes.reshape(-1, 1), box_segments, id_segments.reshape(-1, 1)), 1)
            nl = len(l)
            if nl:
                assert l.shape[1] == length_with_id, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:5] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:5][l[:, 1:5] > 1]}'
                l = np.unique(l, axis=0)  # remove duplicate rows
                if len(l) < nl:
                    segments = np.unique(segments, axis=0)
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, length_with_id), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, length_with_id), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def verify_image_label(args):
    """
        验证图像-标签对的有效性，处理关键点数据完整性
        Args:
            args: 包含(im_file, lb_file, prefix, num_points)的元组
                num_points: 关键点数量，决定标签列数要求

        Returns:
            元组包含处理结果：图像路径、标签数组、图像尺寸、分割信息及各类状态统计

        核心流程：
        1. 图像验证：
            - 检查图像文件完整性，修复损坏的JPEG文件
            - 验证图像尺寸(>10像素)和格式(支持IMG_FORMATS)
        2. 标签验证：
            - 根据num_points动态计算标签列数要求(length_with_points = 5 + num_points*2)
            - 处理关键点缺失情况：自动补零填充缺失的关键点坐标
        3. 数据处理：
            - 分离类别(class)、边界框(box)、关键点(keypoints)数据
            - 使用np.clip确保关键点坐标在[0,1]范围内
            - 拼接为包含所有信息的二维数组
        4. 数据校验：
            - 验证列数匹配、数值非负、坐标归一化
            - 移除重复标签行
        5. 异常处理：
            - 记录缺失(nm)、找到(nf)、空(ne)、损坏(nc)等状态
            - 特殊处理关键点模式下的标签格式不匹配问题
    """
    im_file, lb_file, prefix, num_points = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    length_with_points = 5 + num_points * 2
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    Image.open(im_file).save(im_file, format='JPEG', subsampling=0, quality=100)  # re-save image
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                classes = []
                box_segments = []
                keypoints_segments = []
                if num_points > 0:
                    for x in l:
                        if len(x) == length_with_points:
                            classes.append(x[0])
                            box_segments.append(np.array(x[1:5], dtype=np.float32))
                            keypoints_segments.append(np.clip(np.array(x[5:length_with_points], dtype=np.float32), 0, 1.0))
                        else: #标注缺少关键点 但是训练时又添加了关键点 补齐关键点
                            classes.append(x[0])
                            box_segments.append(np.array(x[1:5], dtype=np.float32))
                            keypoints_segments.append(np.array([0] * num_points * 2, dtype=np.float32))
                    if len(l):
                        classes = np.array(classes, dtype=np.float32)
                        box_segments = np.array(box_segments)
                        keypoints_segments = np.array(keypoints_segments)
                        l = np.concatenate((classes.reshape(-1, 1), box_segments, keypoints_segments), 1)
                else:
                    for x in l:
                        classes.append(x[0])
                        box_segments.append(np.array(x[1:5], dtype=np.float32))
                    if len(l):
                        classes = np.array(classes, dtype=np.float32)
                        box_segments = np.array(box_segments)
                        l = np.concatenate((classes.reshape(-1, 1), box_segments), 1)

            nl = len(l)
            if nl:
                assert l.shape[1] == length_with_points, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                l = np.unique(l, axis=0)  # remove duplicate rows
                if len(l) < nl:
                    segments = np.unique(segments, axis=0)
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, length_with_points), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, length_with_points), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]

def exif_size(img):
    """
        获取EXIF方向校正后的图像尺寸
        Args:
            img: PIL.Image对象，需包含EXIF信息

        Returns:
            tuple: 校正后的尺寸元组(width, height)

        实现要点：
        1. 读取EXIF方向标记(需外部定义orientation常量)
        2. 处理特殊旋转：
           - 6(270度旋转)和8(90度旋转)时交换宽高
        3. 异常安全：无EXIF或读取失败时返回原始尺寸
        注意：orientation应为EXIF方向标签(通常为274)
    """
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s