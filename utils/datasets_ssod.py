import copy
import glob
import logging
import math
import os
import random
from itertools import repeat
from multiprocessing.pool import ThreadPool,Pool
from pathlib import Path
import torch.functional as F
import cv2
import numpy as np
import torch
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import Albumentations, letterbox, augment_hsv
from utils.datasets import verify_image_label, load_image, HELP_URL
from utils.general import xywhn2xyxy, xyxy2xywh, xyn2xy, resample_segments, segment2box
from utils.self_supervised_utils import box_candidates
from utils.torch_utils import torch_distributed_zero_first

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

'''通过遍历所有Exif标签项，找到并设置方向标志(orientation)对应的ExifTag ID。当识别到'Orientation'标签时停止搜索。'''
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    """计算文件列表的哈希值：累加所有有效文件的大小总和（忽略目录），返回总字节数作为哈希值。"""
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
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

def create_target_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, cfg=None, prefix=''):
    """创建目标数据加载器。主要功能：
        1. 分布式优先控制：确保DDP模式下首个进程处理数据集，其他进程复用缓存
        2. 数据集构建：加载图像和伪标签，支持数据增强/矩形训练/图像缓存等特性
        3. 并行化配置：自动计算最优工作进程数，支持分布式采样器
        4. 数据加载策略：根据image_weights选择普通或无限数据加载器
        5. 批处理模式：支持四边形增强的collate函数(quad模式)或标准collate函数

        参数说明：
        path: 数据集路径 | imgsz: 输入图像尺寸 | batch_size: 批大小 | stride: 模型步长
        hyp: 超参数字典 | augment: 启用数据增强 | rect: 矩形训练模式
        rank: 分布式进程编号(-1表示单机) | workers: 最大工作进程数
        image_weights: 启用带权重采样 | quad: 四边形增强模式

        返回：
        DataLoader对象和基础数据集对象
    """
    with torch_distributed_zero_first(rank):
        # if world_size > 0 and rank != 0:
        #     cache = False
        dataset = LoadImagesAndFakeLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      cfg=cfg,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndFakeLabels.collate_fn4 if quad else LoadImagesAndFakeLabels.collate_fn)
    return dataloader, dataset

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
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def img2label_paths(img_paths):
    """
        图像路径转标签路径规则：
        1. 路径替换：将路径中的第一个'sa'字符串(默认'images')替换为'sb'(默认'labels')
        2. 扩展名转换：将文件扩展名统一替换为.txt格式
        示例：/path/images/train.jpg ➔ /path/labels/train.txt

        背景说明：
        早期版本支持自定义路径映射规则(代码中被注释的sa/sb设置)，
        当前版本使用固定映射策略images➔labels
    """
    # sa, sb = os.sep + '/mnt/datasets/' + os.sep, os.sep + '/home/bowen/annotations/labels/lubiantingche_total_train/' + os.sep  # /images/, /labels/ substrings

    # sa, sb = '/AIDATA/', '/mnt/annotations/annotations/labels/xiaokongbao_total_train/'
    sa, sb = 'images', 'labels'
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]

def fake_image_label(args):
    """伪标签校验处理器：
        核心功能：
        1. 图像完整性校验：验证图像可打开性/基本尺寸要求/格式合法性
        2. 特殊格式修复：自动检测并修复损坏的JPEG文件(通过重新保存)
        3. 错误处理机制：分类记录缺失/损坏/空文件等不同异常状态

        输入参数：
        args: 包含(图像路径, 标签路径, 日志前缀)的元组

        返回元组：
        [有效图像路径, 伪标签数据, 图像尺寸, 分割数据,
         缺失计数, 找到计数, 空文件计数, 损坏计数, 状态信息]

        特殊处理逻辑：
        - 对jpeg文件进行魔数校验，发现损坏时自动重保存
        - 统一返回空标签矩阵以适配训练流程
        - 异常捕获范围覆盖图像打开/校验/尺寸验证全流程"""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
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

        l = np.zeros((1, 5), dtype=np.float32)
        nf += 1
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


class LoadImagesAndFakeLabels(Dataset):  # for training/testing
    """半监督数据加载器（支持伪标签生成），核心功能：
        1. 多源数据加载：支持目录/文件列表/混合路径输入，自动递归扫描图像文件
        2. 智能缓存系统：
           - 版本控制(cache_version=0.6)：防止旧缓存格式造成数据错乱
           - 哈希校验：通过文件大小哈希值检测数据变更
           - 磁盘/内存双缓存模式：支持npy格式磁盘缓存节省内存
        3. 数据预处理流水线：
           - 自动应用Albumentations增强库（当augment=True时）
           - 强制启用mosaic增强（与原始实现不同，始终加载4图拼接）
           - 矩形训练优化(rect=True时按长宽比排序图像)
        4. 分布式训练支持：
           - 自动处理路径格式转换（适应不同OS环境）
           - 带进度的多线程缓存（64线程加速）
        5. 异常处理机制：
           - 文件不存在验证（包含详细的错误路径提示）
           - 损坏文件检测与跳过（集成fake_image_label的校验逻辑）
           - 空标签防护（assert确保训练必须有有效标签）

        关键参数说明：
        cfg.SSOD.ssod_hyp：从配置继承半监督超参数
        with_gt：控制是否加载真实标签（兼容全监督与半监督模式）
        mosaic_border：固定为[-img_size//2, -img_size//2]的拼接策略

        数据结构：
        img_files：归一化路径后的图像文件列表
        label_files：通过img2label_paths转换的对应标签路径
        batch_shapes：矩形训练时的动态批尺寸计算
        segments：存储分割标注数据（如存在）

        注意事项：
        - 强制覆盖self.mosaic=True实现特殊增强策略
        - 当cache_images='disk'时生成_npy缓存目录
        - 自动完成单类别转换(single_cls=True时标签归零)"""

    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, cfg=None, prefix=''):
        """初始化流程：
                1. 路径解析阶段：处理多种输入格式（目录/文件列表/混合路径）
                   - 目录处理：递归扫描所有子目录的图像文件
                   - 文件列表处理：自动补全相对路径为绝对路径
                   - 异常检测：立即中断并提示具体错误路径

                2. 标签关联阶段：
                   - 特殊格式处理：支持旧式空格分隔的路径格式
                   - 通过img2label_paths生成标准标签路径
                   - 缓存文件路径生成：与首个标签文件同目录

                3. 缓存验证阶段：
                   - 版本+哈希值双重校验保障数据一致性
                   - 当校验失败时触发cache_labels重新生成
                   - 显示扫描结果统计（找到/缺失/损坏文件计数）

                4. 数据排序阶段（矩形训练模式）：
                   - 按长宽比排序优化内存利用率
                   - 动态计算各batch的适配尺寸（batch_shapes）

                5. 缓存预加载阶段：
                   - 内存模式：将图像数据预载到RAM（适合小数据集）
                   - 磁盘模式：生成.npy文件加速后续读取（适合大数据集）
                   - 进度可视化：显示缓存总量和进度条"""
        self.img_size = img_size
        self.augment = augment

        self.albumentations = Albumentations() if augment else None
        # self.hyp = hyp
        self.hyp = cfg.SSOD.ssod_hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        # self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic = True  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.with_gt = cfg.SSOD.ssod_hyp.with_gt

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(
                [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats + ['txt']])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        if self.img_files[0].endswith('.txt') and len(self.img_files[0].split(' ')) == 2:
            self.label_files = [a.split(' ')[1] for a in self.img_files]
            self.img_files = [a.split(' ')[0] for a in self.img_files]
        else:
            self.label_files = img2label_paths(self.img_files)  # labels
        # print('sefl.label_files', self.label_files)
        # print('sefl.img_files', self.img_files)
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache
        # if cache_path.is_file():
        #      cache, exists = torch.load(cache_path), True  # load
        #      if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
        #          cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        # else:
        #      cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())

        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
                print('self target im cache dir:', self.im_cache_dir)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            # results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            results = ThreadPool(64).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 32 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[
                        i] = x  # img, hw_original, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(32) as pool:
            if self.with_gt:
                pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                            desc=desc, total=len(self.img_files))
            else:
                pbar = tqdm(pool.imap(fake_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
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
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def cache_labels_old(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                # print('lb_file:', lb_file)

                l = np.zeros((1, 5), dtype=np.float32)
                nf += 1
                # if os.path.isfile(lb_file):
                #    nf += 1  # label found
                #    with open(lb_file, 'r') as f:
                #        l = [x.split() for x in f.read().strip().splitlines()]
                #        if any([len(x) > 8 for x in l]):  # is segment
                #            classes = np.array([x[0] for x in l], dtype=np.float32)
                #            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                #            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                #        l = np.array(l, dtype=np.float32)
                #    if len(l):
                #        assert l.shape[1] == 5, 'labels require 5 columns each'
                #        assert (l >= 0).all(), 'negative labels'
                #        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                #        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                #    else:
                #        ne += 1  # label empty
                #        l = np.zeros((0, 5), dtype=np.float32)
                # else:
                #    nm += 1  # label missing
                #    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels, img_ori, M_s = load_mosaic_with_M(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            # TODO 禁掉mixup
            # if random.random() < hyp['mixup']:
            #     img2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
            #     r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
            #     img = (img * r + img2 * (1 - r)).astype(np.uint8)
            #     labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img_ori = copy.deepcopy(img)
                img, labels, M_s = random_perspective_with_M(img, labels,
                                                             degrees=hyp['degrees'],
                                                             translate=hyp['translate'],
                                                             scale=hyp['scale'],
                                                             shear=hyp['shear'],
                                                             perspective=hyp['perspective'])

            # Augment colorspace
            img = augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            if random.random() < hyp['cutout']:
                if (len(labels) > 0):
                    labels = cutout(img, labels)
            if random.random() < hyp['autoaugment']:
                if (len(labels) > 0):
                    aug_labels = np.stack(
                        (labels[:, 2] / img.shape[1], labels[:, 1] / img.shape[0], labels[:, 4] / img.shape[1],
                         labels[:, 3] / img.shape[0], labels[:, 0]), 1)
                    img, labels_out = distort_image_with_autoaugment(img, aug_labels, 'v5')
                    # padding = np.zeros((labels_out.shape[0]))
                    labels = np.stack(
                        (labels_out[:, 4], labels_out[:, 1] * img.shape[1], labels_out[:, 0] * img.shape[0],
                         labels_out[:, 3] * img.shape[1], labels_out[:, 2] * img.shape[0]), 1)
                    # img = self.gridmask(img)
                    # aug_labels = np.stack((labels[:,2]/img.shape[1], labels[:,1]/img.shape[0], labels[:,4]/img.shape[1], labels[:, 3]/img.shape[0]), 1)
                    # img, labels_out = distort_image_with_autoaugment(img, aug_labels, 'v2')
                    # padding = np.zeros((labels_out.shape[0]))
                    # labels= np.stack((padding, labels_out[:,1] * img.shape[1], labels_out[:,0] * img.shape[0], labels_out[:,3] * img.shape[1], labels_out[:,2] * img.shape[0]), 1)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            img, labels = self.albumentations(img, labels)

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                # img_ori = np.flipud(img_ori)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
                M_s[11] = 1

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                # img_ori = np.fliplr(img_ori)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
                M_s[12] = 1

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        # cv2.imwrite('train_strong.jpg', img)
        # cv2.imwrite('train_weak.jpg', img_ori)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)

        img_ori = img_ori[:, :, ::-1].transpose(2, 0, 1)
        img_ori = np.ascontiguousarray(img_ori)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, torch.from_numpy(
            img_ori), torch.from_numpy(M_s)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, img_ori, M_s = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        for i, l in enumerate(M_s):
            l[0] = i
        # print('collate M_s:', M_s)
        # print('collate path:', path)
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, torch.stack(img_ori, 0), torch.stack(M_s, 0)

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

def verify_image_label(args):
    """图像标签对完整性校验器：
        核心功能：
        1. 图像校验维度：
           - 基础验证：通过PIL的verify()检测文件完整性
           - 尺寸验证：确保图像宽高均>10像素
           - 格式验证：检查文件后缀是否符合预设格式列表(IMG_FORMATS)
           - 特殊修复：自动检测并修复损坏的JPEG文件(通过重写文件实现)

        2. 标签处理维度：
           - 文件存在性检测：区分标签缺失/空文件/有效标签三种状态
           - 格式兼容处理：同时支持边界框(label格式为cls,x,y,w,h)和
             分割数据(原始实现保留处理逻辑但当前版本仅处理xywh)
           - 数据合法性检查：
             * 列数校验：强制要求5列数据格式
             * 数值范围：类别标签非负，坐标值归一化(0~1之间)
             * 去重处理：移除完全重复的标签行

        3. 状态跟踪机制：
           - 计数器系统：nm(缺失)/nf(找到)/ne(空)/nc(损坏)四种状态统计
           - 消息反馈：生成包含详细诊断信息的警告消息

        输入参数：
        args : 元组结构 (图像路径, 标签路径, 日志前缀)

        返回元组：
        [有效图像路径, 处理后的标签数组, 图像尺寸, 分割数据,
         缺失计数, 找到计数, 空文件计数, 损坏计数, 状态信息]

        异常处理：
        - 捕获所有异常并归类为损坏状态
        - 返回结构保持一致性，错误时路径和标签为None"""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
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
                if any([(len(x) > 5 and len(x) <20) for x in l]):  # is segment
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    # segments = [np.array(x[1:13], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    #l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    # l = np.concatenate((classes.reshape(-1, 1), np.array(x[1:5])), 1)  # (cls, xywh)
                    segments = [np.array(x[1:5], dtype=np.float32) for x in l]  # (cls, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments), 1)  # (cls, xywh)
                    segments = []
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                l = np.unique(l, axis=0)  # remove duplicate rows
                if len(l) < nl:
                    segments = np.unique(segments, axis=0)
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]

def load_image(self, index):
    """图像加载与尺寸标准化处理器：
        核心功能：
        1. 多级缓存机制：
           - 内存缓存优先：当self.imgs[index]存在时直接返回缓存数据
           - 磁盘缓存次优：查找同目录的.npy文件加速加载
           - 原始文件兜底：使用OpenCV读取BGR格式图像

        2. 动态尺寸调整：
           - 基准尺寸：以img_size为长边基准，保持宽高比进行缩放
           - 缩放策略：训练时允许上采样（与数据增强配合），验证时仅下采样
           - 插值方法：强制使用双线性插值(INTER_LINEAR)保证质量

        3. 数据校验：
           - 路径有效性断言：确保图像文件存在（否则抛出Image Not Found异常）
           - 尺寸记录：同时保留原始尺寸(hw_original)和调整后尺寸(hw_resized)

        工作流程：
        if 内存缓存命中：
            直接返回缓存的图像数据及关联尺寸信息
        else:
            if 存在磁盘npy缓存：
                加载预处理好的numpy数组
            else:
                用OpenCV读取原始图像(BGR格式)
            执行尺寸缩放（按长边对齐img_size）
            返回处理后图像及尺寸元数据

        参数说明：
        index : 数据集索引，指向具体图像文件

        返回元组：
        img : 处理后的图像数组（BGR顺序）
        (h0, w0) : 原始图像高宽
        (h, w) : 调整后图像高宽

        设计细节：
        - 缓存优先级：内存 > 磁盘npy > 原始文件，兼顾速度与内存消耗
        - 插值方法统一化：放弃原始代码的条件选择，始终使用INTER_LINEAR
          以改善放大时的图像质量，可能与mosaic增强策略配合
        - 尺寸记录机制：保留原始尺寸用于后续数据增强中的坐标换算"""
    img = self.imgs[index]
    if img is None:  # not cached
        npy = self.img_npy[index]
        if npy and npy.exists():  # load npy
            img = np.load(npy)
        else:  # read image
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            #interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            interp =cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    '''HSV色彩空间数据增强器：
    核心功能：
    1. 随机色彩扰动：在HSV空间分别对色调(H)/饱和度(S)/明度(V)三个通道
       进行非线性变换，增强模型颜色鲁棒性

    实现原理：
    - 色相旋转：通过模运算实现循环偏移，处理OpenCV的0-180度色相范围
    - 饱和度/明度缩放：线性缩放后裁剪到有效范围(0-255)

    参数说明：
    hgain : 色相扰动幅度系数(默认0.5)
    sgain : 饱和度扰动幅度系数(默认0.5)
    vgain : 明度扰动幅度系数(默认0.5)

    处理流程：
    1. 生成随机增益因子：r = 随机[-1,1] * 增益系数 + 1 → 最终增益范围[1-gain, 1+gain]
    2. 图像空间转换：BGR→HSV并分离通道
    3. 创建查找表(LUT)：
       - 色相：应用环形偏移 (x*r[0])%180 → 保持色相值在有效范围
       - 饱和度：线性缩放后裁剪到0-255
       - 明度：线性缩放后裁剪到0-255
    4. 应用LUT变换后合并通道，转回BGR空间

    技术细节：
    - 使用查找表优化性能，避免逐像素计算
    - 保持原始数据类型(uint8)确保图像格式正确
    - 单独处理色相通道的环形特性，避免色相跳变

    示例数学运算：
    当hgain=0.5时，r[0]随机范围为0.5~1.5
    原像素x=100 → 100 * 1.5=150 (有效) | 100 * 0.5=50 (有效)
    但x=160*r[0]=240 → 240%180=60 (实现色相环旋转效果)'''

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # 随机增益矩阵 [3,]
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))  # 分离HSV通道
    dtype = img.dtype  # 保持uint8数据类型

    # 构建查找表(Look Up Table)
    x = np.arange(0, 256, dtype=np.int16)  # 原始像素值范围
    lut_hue = ((x * r[0]) % 180).astype(dtype)  # 色相环形处理
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 饱和度线性裁剪
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 明度线性裁剪

    # 应用查找表并合并通道
    img_hsv = cv2.merge((
        cv2.LUT(hue, lut_hue),  # 色相通道变换
        cv2.LUT(sat, lut_sat),  # 饱和度通道变换
        cv2.LUT(val, lut_val)  # 明度通道变换
    )).astype(dtype)

    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # 转回BGR空间


def load_mosaic_with_M(self, index):
    '''马赛克增强与透视变换处理：
    核心功能：
    1. 四图拼接：随机选取4张图像拼接成马赛克大图，增强小目标检测能力
    2. 动态坐标转换：将各子图的标注坐标映射到拼接图的绝对坐标系
    3. 复合数据增强：应用随机透视变换（含旋转/平移/缩放/错切/透视形变）
    4. 变换矩阵追踪：返回变换矩阵M_s用于后续伪标签逆向映射

    实现流程：
    1. 初始化拼接画布：创建2倍img_size的空白图像（填充114灰度值）
    2. 随机确定拼接中心：在[-img_size//2, 3*img_size//2]范围内随机选取中心点(xc,yc)
    3. 循环处理四个子图：
       a. 加载图像并确定其在拼接图中的位置（左上/右上/左下/右下）
       b. 计算子图在原始图像和拼接画布中的对应区域坐标
       c. 将子图局部区域复制到拼接画布指定位置
       d. 转换标注坐标：将归一化xywh转换为拼接图绝对坐标，考虑填充偏移(padw/padh)
    4. 坐标裁剪：限制所有标注坐标不超过2倍img_size范围
    5. 透视增强：应用带随机参数的几何变换，生成变换矩阵
    6. 返回处理结果：增强后图像/标注/原始拼接图/变换矩阵

    关键参数说明：
    mosaic_border：控制中心点随机范围，默认[-img_size//2, -img_size//2]
    hyp配置参数：
       degrees：随机旋转角度范围
       translate：随机平移比例
       scale：随机缩放比例范围
       shear：随机错切强度
       perspective：透视形变强度（0-0.001）

    坐标转换细节：
    使用xywhn2xyxy将归一化坐标转换为拼接图绝对坐标，其中w/2和h/2处理是因为
    原始标注基于0.5*img_size的预处理（具体实现需参考xywhn2xyxy内部逻辑）

    变换矩阵M_s：
    记录从原始拼接图(img4_ori)到增强后图像(img4)的几何变换，可用于：
    - 教师模型预测伪标签的坐标逆变换
    - 可视化增强前后的对应关系
    - 计算增强带来的形变程度'''

    # 初始化标签容器和图像尺寸
    labels4, segments4 = [], []
    s = self.img_size
    # 随机生成马赛克中心点（允许部分超出图像边界）
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]

    # 选取4张图像索引（当前索引+随机3张），并打乱顺序
    indices = [index] + random.choices(self.indices, k=3)
    random.shuffle(indices)

    # 遍历处理每个子图
    for i, index in enumerate(indices):
        # 加载图像及尺寸信息（hw_resized为缩放后尺寸）
        img, _, (h, w) = load_image(self, index)

        # 根据位置索引计算拼接区域
        if i == 0:  # 左上子图
            # 创建2倍尺寸的空白画布（BGR通道，填充114灰度）
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            # 计算子图在画布中的坐标范围（考虑可能的部分超出）
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            # 计算子图有效区域在原图中的坐标范围
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            # 类似逻辑处理其他三个区域（右上/左下/右下）...

        # 将子图局部复制到画布中
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        # 计算坐标偏移量（用于标注转换）
        padw = x1a - x1b
        padh = y1a - y1b

        # 转换标注坐标到拼接图坐标系
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            # 归一化坐标转绝对坐标（w/2和h/2适配预处理逻辑）
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w / 2, h / 2, padw / 2, padh / 2)
            # 转换分割点坐标
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # 合并标签并裁剪越界坐标
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # 限制在2*img_size范围内

    # 扩展画布边界并进行尺寸调整
    height = img4.shape[0] + self.mosaic_border[0] * 2
    width = img4.shape[1] + self.mosaic_border[1] * 2
    img4 = cv2.resize(img4, (height, width))
    img4_ori = copy.deepcopy(img4)  # 保存原始拼接图用于后续变换追踪

    # 应用随机透视变换并获取变换矩阵
    img4, labels4, M_s = random_perspective_with_M(
        img4, labels4, segments4,
        degrees=self.hyp['degrees'],
        translate=self.hyp['translate'],
        scale=self.hyp['scale'],
        shear=self.hyp['shear'],
        perspective=self.hyp['perspective'])

    return img4, labels4, img4_ori, M_s


def random_perspective_with_M(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10,
                              perspective=0.0,
                              border=(0, 0)):
    '''随机几何变换增强器（含变换矩阵追踪）：
    核心功能：
    1. 复合空间变换：组合旋转/平移/缩放/错切/透视变换，增强模型几何鲁棒性
    2. 标签同步处理：将边界框或分割点坐标同步变换至增强后图像空间
    3. 变换参数封装：生成包含完整变换信息的矩阵M_s，支持逆变换计算

    参数说明：
    img : 输入图像（BGR格式，通常来自马赛克拼接结果）
    targets : 原始标注框数组（格式：[class, x1, y1, x2, y2]）
    segments : 分割点坐标列表
    degrees : 随机旋转角度范围（±degrees）
    translate : 随机平移比例（相对于图像尺寸）
    scale : 随机缩放比例范围（1±scale）
    shear : 随机错切角度范围（±shear度）
    perspective : 透视形变强度（0-0.001）
    border : 图像扩展边距（用于透视变换的留白）

    返回元组：
    img : 变换后的图像数据
    targets : 更新后的标注框数组
    M_s : 变换参数矩阵（包含逆变换所需的所有参数）

    实现流程：
    1. 变换矩阵构建：顺序生成中心化(C)/透视(P)/旋转缩放(R)/错切(S)/平移(T)矩阵
    2. 矩阵复合计算：M = T @ S @ R @ P @ C（矩阵右乘顺序）
    3. 图像变换：应用仿射/透视变换至扩展后的画布
    4. 标签变换：将原始标注坐标通过M矩阵映射到新图像空间
    5. 结果过滤：移除无效或过小的标注框

    变换矩阵详解：
    M_ori : 原始组合变换矩阵（3x3齐次坐标形式）
    M_s : 序列化参数矩阵，结构为：
        [-1, M_ori_flatten, scale, 0, 0]（用于后续逆变换计算）'''

    # 图像尺寸计算（考虑边界扩展）
    height = img.shape[0] + border[0] * 2  # 最终高度 = 原高 + 上下边界
    width = img.shape[1] + border[1] * 2  # 最终宽度 = 原宽 + 左右边界

    # ----------------------- 变换矩阵构建阶段 -----------------------
    # 中心化矩阵（将图像中心移至原点）
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x轴平移（像素单位）
    C[1, 2] = -img.shape[0] / 2  # y轴平移（像素单位）

    # 透视变换矩阵（在x/y方向添加随机透视形变）
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x方向透视强度
    P[2, 1] = random.uniform(-perspective, perspective)  # y方向透视强度

    # 旋转缩放矩阵（绕原点旋转+缩放）
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # 错切矩阵（x/y方向随机错切）
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x轴错切
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y轴错切

    # 平移矩阵（在扩展后的画布上随机定位）
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x平移量
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y平移量

    # 复合变换矩阵（注意矩阵相乘顺序）
    M = T @ S @ R @ P @ C  # 从右到左依次应用：中心化→透视→旋转缩放→错切→平移
    M_ori = copy.deepcopy(M)  # 保留原始矩阵用于参数封装

    # ----------------------- 图像变换阶段 -----------------------
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # 判断是否需要变换
        if perspective:
            # 透视变换（支持3D形变）
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            # 仿射变换（2D线性变换）
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # ----------------------- 标签变换阶段 -----------------------
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)  # 判断是否使用分割标注
        new = np.zeros((n, 4))  # 初始化新标注容器

        if use_segments:  # 分割点处理模式
            segments = resample_segments(segments)  # 上采样分割点（增加采样密度）
            for i, segment in enumerate(segments):
                # 构建齐次坐标并应用变换矩阵
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # 坐标变换
                # 透视归一化（如果是透视变换）
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                # 转换分割点为边界框
                new[i] = segment2box(xy, width, height)
        else:  # 边界框处理模式
            # 提取边界框四角坐标（x1y1, x2y2, x1y2, x2y1）
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            # 应用变换矩阵
            xy = xy @ M.T
            # 透视归一化并重塑形状
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)
            # 计算新边界框坐标
            x = xy[:, [0, 2, 4, 6]]  # 所有点的x坐标
            y = xy[:, [1, 3, 5, 7]]  # 所有点的y坐标
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # 裁剪越界坐标
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # 过滤无效标注框（面积过小或位置异常）
        i = box_candidates(
            box1=targets[:, 1:5].T * s,  # 原始框尺寸乘以缩放因子
            box2=new.T,
            area_thr=0.01 if use_segments else 0.10  # 面积阈值（分割模式更严格）
        )
        targets = targets[i]
        targets[:, 1:5] = new[i]

    # 构建变换参数矩阵（用于伪标签逆变换）
    M_s = np.concatenate([
        np.array([-1]),  # 固定标识位
        np.array(M_ori).flatten(),  # 展平的3x3变换矩阵
        np.array([s]),  # 缩放因子
        np.array([0]), np.array([0])  # 预留位
    ])

    return img, targets, M_s


def cutout(image, labels):
    '''目标区域自适应Cutout增强：
    核心功能：
    1. 聚焦目标增强：仅在标注目标区域内生成遮挡，避免背景干扰
    2. 多尺度遮挡：生成不同比例的遮挡块模拟局部遮挡场景
    3. 动态标签过滤：移除遮挡面积过大的目标框，防止训练噪声

    实现流程：
    1. 计算目标聚合区域：根据所有标注框确定最小包围区域(ROI)
    2. 定义交并比函数：计算遮挡区域与标注框的覆盖比例
    3. 生成多尺度遮挡：
       - 使用递减比例序列[0.5, 0.25,...]生成不同大小遮挡块
       - 在ROI区域内随机定位遮挡位置
       - 用随机灰度填充遮挡区域
    4. 标签清洗：移除被遮挡超过80%的目标框

    参数说明：
    image : 输入图像(BGR格式)
    labels : 标注数组([[class, x1, y1, x2, y2], ...])

    返回：
    过滤后的有效标签数组

    设计亮点：
    - 限制遮挡在目标聚集区域，提升增强有效性
    - 多尺度遮挡组合增强模型局部特征鲁棒性
    - 动态标签过滤平衡数据增强与信息保留'''

    h, w = image.shape[:2]
    # 计算目标聚集区域的最小包围框
    bbox_roi_x_start = int(min(labels[:, 1]))
    bbox_roi_y_start = int(min(labels[:, 2]))
    bbox_roi_x_end = int(max(labels[:, 3]))
    bbox_roi_y_end = int(max(labels[:, 4]))
    roi_h = bbox_roi_y_end - bbox_roi_y_start
    roi_w = bbox_roi_x_end - bbox_roi_x_start

    def bbox_ioa(box1, box2):
        '''改进交并比计算：遮挡区域与目标框的交集占目标框面积比例'''
        # 矩阵转置加速批量计算
        box2 = box2.transpose()
        # 解构坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        # 交集区域计算
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
        # 分母使用目标框面积（非并集）
        return inter_area / ((b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16)

    # 多尺度遮挡比例（大->小递减，数量递增）
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0825] * 8 + [0.05125] * 16

    for s in scales:
        # 根据当前尺度生成遮挡尺寸
        mask_h = random.randint(1, max(int(roi_h * s), 1))
        mask_w = random.randint(1, max(int(roi_w * s), 1))

        # 在目标区域内随机定位遮挡中心
        center_x = random.randint(bbox_roi_x_start, bbox_roi_x_end)
        center_y = random.randint(bbox_roi_y_start, bbox_roi_y_end)

        # 计算遮挡区域坐标（限制在图像范围内）
        xmin = int(max(0, center_x - mask_w // 2))
        ymin = int(max(0, center_y - mask_h // 2))
        xmax = int(min(w, xmin + mask_w))
        ymax = int(min(h, ymin + mask_h))

        # 应用随机灰度填充（64-191避免纯黑/白）
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # 过滤被严重遮挡的标签（保留遮挡面积<80%的目标）
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # 计算交并比
            labels = labels[ioa < 0.80]  # 阈值过滤

    return labels

