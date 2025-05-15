import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import colorstr


def check_anchor_order(m):
    """
    检查YOLOv5检测模块m的锚点顺序是否与步长顺序一致，必要时进行纠正

    参数:
        m: YOLOv5模型的Detect()模块

    工作流程:
        1. 计算每个锚点的面积并展平为一维张量
        2. 计算第一个和最后一个锚点面积的差值da
        3. 计算第一个和最后一个步长的差值ds
        4. 比较da和ds的符号:
           - 若符号相同: 锚点和步长顺序一致
           - 若符号不同: 锚点顺序需要翻转
    """
    a = m.anchors.prod(-1).view(-1)  # 计算每个锚点的面积 (w*h)
    da = a[-1] - a[0]  # 锚点面积差: 最后一个锚点面积减第一个锚点面积
    ds = m.stride[-1] - m.stride[0]  # 步长差: 最后一个步长减第一个步长
    if da.sign() != ds.sign():  # 判断锚点和步长的顺序是否一致
        print('Reversing anchor order')  # 打印提示信息
        m.anchors[:] = m.anchors.flip(0)  # 翻转锚点顺序以匹配步长顺序


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """
    检查锚点与数据集的匹配程度，必要时重新计算锚点

    参数:
        dataset: 训练数据集
        model: YOLOv5模型
        thr: 锚点匹配阈值，默认为4.0
        imgsz: 输入图像尺寸，默认为640

    工作流程:
        1. 获取模型的检测头模块
        2. 计算数据集的目标框尺寸并进行随机缩放增强
        3. 定义评估指标函数，计算最佳可能召回率(BPR)和阈值以上锚点比例(AAT)
        4. 评估当前锚点的性能
        5. 如果BPR低于阈值(0.99)，尝试使用K-means重新计算锚点
        6. 比较新旧锚点性能，选择更优的锚点配置
    """
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')
    # m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    m = model.module.head if hasattr(model, 'module') else model.head  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """计算锚点匹配指标"""
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
    if bpr < 0.99:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchors.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    从训练数据集中创建K-means优化的锚点

    参数:
        dataset: 数据yaml文件路径，或已加载的数据集
        n: 锚点数量
        img_size: 训练图像尺寸
        thr: 锚点-标签宽高比阈值，训练超参数hyp['anchor_t']，默认=4.0
        gen: 使用遗传算法优化锚点的迭代代数
        verbose: 是否打印所有结果

    返回:
        k: K-means优化后的锚点

    使用方法:
        from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    thr = 1. / thr  # 转换为比率阈值的倒数
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):  # 计算锚点与目标的匹配指标
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # 宽高比指标
        # x = wh_iou(wh, torch.tensor(k))  # IOU指标
        return x, x.max(1)[0]  # 返回所有匹配值和最佳匹配值

    def anchor_fitness(k):  # 锚点适应度评估函数（用于遗传算法）
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # 只考虑超过阈值的匹配

    def print_results(k):  # 打印锚点评估结果
        k = k[np.argsort(k.prod(1))]  # 按面积从小到大排序
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # 最佳可能召回率，超过阈值的锚点数量
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # 输出锚点值，用于*.cfg文件
        return k

    # 加载数据集
    if isinstance(dataset, str):  # 如果传入的是yaml文件路径
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # 加载数据配置字典
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)  # 加载训练集

    # 获取标签宽高
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # 所有边界框的宽高

    # 过滤异常值
    i = (wh0 < 3.0).any(1).sum()  # 统计过小的边界框数量
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # 过滤掉宽度或高度小于2像素的边界框
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # 随机缩放增强

    # K-means聚类计算
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)  # 用于白化处理的标准差
    k, dist = kmeans(wh / s, n, iter=30)  # 执行K-means聚类（白化后的数据）
    assert len(k) == n, f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s  # 恢复原始比例
    wh = torch.tensor(wh, dtype=torch.float32)  # 转换为PyTorch张量（过滤后）
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # 转换为PyTorch张量（未过滤）
    k = print_results(k)  # 打印初始K-means结果

    # 锚点进化（遗传算法优化）
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # 初始适应度，形状，变异概率，变异强度
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # 进度条
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # 变异直到发生变化（避免重复）
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)  # 生成变异后的锚点
        fg = anchor_fitness(kg)  # 评估变异后的适应度
        if fg > f:  # 如果适应度提高
            f, k = fg, kg.copy()  # 更新最优锚点
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)  # 打印当前最优结果

    return print_results(k)  # 返回最终优化后的锚点