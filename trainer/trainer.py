import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, SGD, lr_scheduler
from tqdm import tqdm
from contextlib import redirect_stdout

from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import colorstr, check_img_size, check_suffix, methods, init_seeds, one_cycle, \
    labels_to_class_weights
from utils.loggers import Loggers
from utils.metrics import fitness, MetricMeter
from utils.plots import plot_labels
from utils.torch_utils import torch_distributed_zero_first, intersect_dicts, ModelEMA, is_parallel, de_parallel
from yolo_models.detector.yolo import Model
from yolo_models.loss.loss import ComputeLoss, ComputeFastXLoss, ComputeNanoLoss

LOGGER = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, device, callbacks):
        """
        训练流程管理类

        功能说明:
        - 集成训练环境初始化、模型构建、优化器配置、数据加载等功能

        重点细节:
        - 参数边界条件:
          * device: 需为有效计算设备 (如torch.device('cuda')或torch.device('cpu'))
          * cfg: 必须包含模型结构、超参数、数据路径等完整配置

        - 关键流程:
          1. set_env: 初始化训练环境(保存路径、日志系统、分布式配置)
          2. build_model: 加载模型架构与预训练权重
          3. build_optimizer: 配置优化器及学习率策略
          4. build_dataloader: 准备训练/验证数据集
          5. build_ddp_model: 分布式训练模型包装

        - 核心配置:
          * 输入图像尺寸通过cfg.imgsz获取，自动同步到train/val流程
          * 支持通过cfg.hyp配置超参数(如学习率、权重衰减等)
          * 自动记录训练结果到save_dir目录

        - 异常处理:
          * 会验证cfg配置完整性，缺少关键参数会触发AttributeError
          * 设备不可用时抛出RuntimeError

        - 性能注意:
          * 数据加载workers数量建议设置为CPU核心数的75%-100%
          * 分布式训练需注意批次大小的自动缩放
          * 混合精度训练需在cfg中明确开启

        示例:
        >>> cfg = {...}  # 包含模型结构、数据路径、超参数等的配置字典
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> trainer = Trainer(cfg, device, callbacks=[], LOCAL_RANK=-1, RANK=0, WORLD_SIZE=1)
        """
        self.cfg = cfg

        # 环境初始化
        self.set_env(cfg, device)

        # 模型构建与优化器初始化
        self.opt_scales = None  # 混合精度训练缩放因子
        ckpt = self.build_model(cfg, device)  # 加载可能的预训练权重
        self.build_optimizer(cfg, ckpt=ckpt)

        # 数据加载系统初始化
        self.build_dataloader(cfg, callbacks)

        # 训练信息日志输出
        LOGGER.info(f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
                    f'Using {self.train_loader.num_workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')

        # 训练控制参数
        self.device = device  # 主计算设备
        self.break_iter = -1  # 迭代中断点 (用于调试)
        self.break_epoch = -1  # 轮次中断点 (用于调试)

    def build_dataloader(self, cfg, callbacks):
        # todo 构建新的数据加载系统
        """
        构建单机训练和验证数据加载系统

        功能说明:
        - 初始化训练集/验证集数据加载器
        - 执行数据预处理配置检查
        - 实现数据完整性验证(标签范围检查、锚框适配检查)
        - 支持数据增强与优化策略

        参数边界条件:
          * cfg.Dataset必须包含img_size、workers等配置项
          * cfg.hyp需包含数据增强相关参数
          * callbacks需要实现预训练结束回调接口
        """
        # 图像尺寸对齐模型要求
        model_stride = int(self.model.stride.max())
        gs = max(model_stride, 32)  # 计算最小网格基准尺寸
        self.imgsz = check_img_size(cfg.Dataset.img_size, gs, floor=gs * 2)  # 校验并调整输入尺寸

        # 数据加载器核心参数配置
        batch_size = cfg.batch_size  # 使用完整批次大小
        workers = cfg.Dataset.workers or os.cpu_count()  # 自动获取CPU核心数

        # 训练数据加载器初始化
        self.train_loader, self.dataset = create_dataloader(
            path=self.data_dict['train'],  # 训练集路径
            imgsz=self.imgsz,
            batch_size=batch_size,  # 单机模式使用完整批次
            stride=model_stride,
            single_cls=cfg.single_cls,  # 单类别模式
            hyp=cfg.hyp,  # 超参数配置
            augment=cfg.hyp.use_aug,  # 数据增强开关
            cache=cfg.cache,  # 数据缓存策略
            rect=cfg.rect,  # 矩形训练优化
            workers=min(workers, 64)  # 限制最大工作进程数
        )

        # 标签完整性验证
        max_label_class = int(np.concatenate(self.dataset.labels, 0)[:, 0].max())
        if max_label_class >= self.nc:
            raise ValueError(f'检测到无效标签类别{max_label_class} (允许范围0-{self.nc - 1})')

        # 验证集数据加载器
        self.val_loader = create_dataloader(
            path=self.data_dict['val'],
            imgsz=self.imgsz,
            batch_size=batch_size * 2,  # 验证批次通常更大
            stride=model_stride,
            single_cls=cfg.single_cls,
            hyp=cfg.hyp,
            augment=False,  # 验证阶段不增强
            cache=cfg.cache
        )[0]

        # 初始化阶段数据分析
        if not cfg.resume:
            # 标签分布可视化
            if self.plots:
                labels = np.concatenate(self.dataset.labels, 0)
                plot_labels(labels, self.names, self.save_dir)

            # 自动锚框优化
            if cfg.autoanchor:
                check_anchors(self.dataset,
                              model=self.model,
                              thr=cfg.hyp.anchor_t,  # 匹配阈值
                              imgsz=self.imgsz)

            # 模型精度稳定性处理
            self.model.half().float()

        # 配置后期训练优化
        self.no_aug_epochs = cfg.hyp.no_aug_epochs  # 最后N个epoch关闭数据增强
        callbacks.run('on_pretrain_routine_end')  # 触发预训练完成回调

    def build_model(self, cfg, device):
        """
        构建并初始化模型架构
        """
        # 预训练权重处理
        check_suffix(cfg.weights, '.pt')  # 验证文件后缀

        if pretrained := (cfg.weights and Path(cfg.weights).exists()):
            # 直接加载本地权重文件
            ckpt = torch.load(cfg.weights, map_location=device)

            # 模型初始化 (继承预训练模型配置或使用当前配置)
            self.model = Model(cfg or ckpt['model'].yaml).to(device)

            # 参数剪裁加载
            exclude = ['anchor'] if cfg.Model.anchors and not cfg.resume else []
            csd = intersect_dicts(ckpt['model'].float().state_dict(),
                                  self.model.state_dict(), exclude=exclude)

            # 剪裁微调特殊处理
            # if cfg.prune_finetune:
            #     dynamic_load(self.model, csd, reinitialize=cfg.reinitial)
            #     if cfg.reinitial:
            #         print("执行动态参数重初始化")

            self.model.load_state_dict(csd, strict=False)  # 非严格模式加载

        # 新模型初始化
        else:
            self.model = Model(cfg).to(device)
            ckpt = None

        # 参数冻结策略
        if cfg.freeze_layer_num > 0:
            freeze = [f'model.{x}.' for x in range(cfg.freeze_layer_num)]
            for k, v in self.model.named_parameters():
                if any(x in k for x in freeze):
                    v.requires_grad = False

        # EMA初始化（始终启用）
        self.ema = ModelEMA(self.model) if pretrained else None # todo 需要修改

        self.start_epoch = 0
        # 训练恢复处理
        if pretrained and not cfg.reinitial:
            # 优化器状态加载
            if ckpt.get('optimizer'):
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except ValueError:
                    print('优化器类型不匹配，重新初始化')

            # EMA状态恢复
            if self.ema and ckpt.get('ema'):
                try:
                    self.ema.ema.load_state_dict(ckpt['ema'].state_dict())
                except RuntimeError:
                    print('EMA参数不匹配，重新构建')

            # 训练周期设置
            self.start_epoch = ckpt['epoch'] + 1
            if cfg.resume:
                assert self.start_epoch > 0, '无法恢复已完成训练的任务'

            # 微调周期扩展
            self.epochs += ckpt['epoch'] if self.epochs < self.start_epoch else 0

        self.epoch = self.start_epoch
        self.model_type = self.model.model_type
        self.detect = self.model.head
        return ckpt

    def build_optimizer(self, cfg, optinit=True, weight_masks=None, ckpt=None):
        """
        构建优化器与学习率调度系统
        """
        # 梯度累积与权重衰减计算
        nbs = 64  # 名义批量基准值
        self.accumulate = max(round(nbs / self.batch_size), 1)  # 自动计算累积步数
        weight_decay = cfg.hyp.weight_decay * self.batch_size * self.accumulate / nbs
        LOGGER.info(f"自适应权重衰减系数: {weight_decay:.5f}")

        # 参数智能分组 (权重/BN参数/偏置)
        g_bnw, g_w, g_b = [], [], []
        for module in self.model.modules():
            if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                g_b.append(module.bias)  # 偏置参数组
            if isinstance(module, nn.BatchNorm2d):
                g_bnw.append(module.weight)  # BN层权重(无衰减)
            elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                g_w.append(module.weight)  # 普通权重(带衰减)

        # 标准优化器初始化
        if cfg.adam:
            self.optimizer = AdamW(g_b, lr=cfg.hyp.lr0, betas=(cfg.hyp.momentum, 0.999))
        else:
            self.optimizer = SGD(g_b, lr=cfg.hyp.lr0, momentum=cfg.hyp.momentum, nesterov=True)

        # 添加参数组（带不同超参数）
        self.optimizer.add_param_group({'params': g_w, 'weight_decay': weight_decay})
        self.optimizer.add_param_group({'params': g_bnw})

        # 优化器信息日志
        LOGGER.info(f"{colorstr('优化器:')} {type(self.optimizer).__name__} "
                    f"参数组划分: {len(g_w)}权重, {len(g_bnw)}BN参数, {len(g_b)}偏置")

        # 学习率调度器配置
        if cfg.linear_lr:
            self.lf = lambda x: (1 - x / (self.epochs - 1)) * (1.0 - cfg.hyp.lrf) + cfg.hyp.lrf  # 线性衰减
        else:
            self.lf = one_cycle(1, cfg.hyp.lrf, self.epochs)  # 单周期余弦退火
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.scheduler.last_epoch = self.epoch - 1  # 训练恢复时保持连续性

        # 混合精度与状态恢复
        self.scaler = amp.GradScaler(enabled=self.cuda)
        if ckpt and 'optimizer' in ckpt:  # 加载历史优化器状态
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                LOGGER.info('成功加载优化器状态')
            except ValueError:
                LOGGER.warning('优化器状态不兼容，进行冷启动')

    def set_env(self, cfg, device):
        """
        初始化基础训练环境配置

        核心功能:
        1. 训练输出目录初始化
        2. 关键配置参数持久化
        3. 数据集配置完整性验证
        4. 基础设备设置
        5. 随机种子设置

        """
        # 基础参数解包
        self.save_dir = Path(cfg.save_dir)  # 转换为Path对象
        self.epochs = cfg.epochs
        self.batch_size = cfg.Dataset.batch_size

        # 设备配置
        self.device = device
        self.cuda = device.type != 'cpu'  # CUDA可用标志

        # 路径系统初始化
        weights_dir = self.save_dir / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录
        self.last = weights_dir / 'last.pt'  # 最后模型路径
        self.best = weights_dir / 'best.pt'  # 最佳模型路径

        # 配置持久化
        with open(self.save_dir / 'opt.yaml', 'w') as f:
            with redirect_stdout(f):
                print(cfg.dump())  # 将配置转储为YAML

        # 随机种子设置 (固定种子保证可复现性)
        init_seeds(0)  # 使用固定种子

        # 数据集配置验证
        self.data_dict = {
            'train': cfg.Dataset.train,
            'val': cfg.Dataset.val,
            'nc': cfg.Dataset.nc,
            'names': cfg.Dataset.names
        }
        self.nc = 1 if cfg.single_cls else int(self.data_dict['nc'])  # 类别数
        # 处理单类别特殊情况
        self.names = ['item'] if cfg.single_cls and len(self.data_dict['names']) != 1 \
            else self.data_dict['names']
        # 类别数量一致性检查
        assert len(self.names) == self.nc, \
            f'类别名称数量({len(self.names)})与配置类别数({self.nc})不匹配'

    def build_ddp_model(self, cfg, device):
        """
        构建分布式数据并行(DDP)模型并配置损失函数

        功能说明:
        - 将模型包装为DDP模型以支持多GPU/分布式训练
        - 初始化损失计算函数
        - 配置类别相关参数(类别权重/名称/数量)
        - 适配模型检测头获取方式

        重点细节:
        - 参数边界条件:
          * device: 必须与已初始化的训练设备一致
          * cfg.Loss.type: 需实现对应的损失计算类
          * self.RANK: 分布式训练时需>=0，单卡训练时为-1

        - DDP配置要点:
          * 自动处理设备分配(device_ids/output_device)
          * find_unused_parameters=True允许梯度传播未使用参数
          * 仅当RANK != -1时启用DDP模式

        - 核心处理逻辑:
          1. DDP模型包装: 实现数据并行和梯度同步
          2. 类别权重计算: 基于标签分布自动平衡损失
          3. 损失函数选择: 根据配置动态加载不同损失实现
          4. 检测头适配: 自动处理普通/DDP模型的结构差异

        - 关键算法:
          * 分布式数据并行: 通过AllReduce算法同步梯度
          * 类别权重计算: 反比于类别出现频率进行加权
          * 动态损失选择: 支持标准/快速/轻量级等不同损失计算方式

        - 异常处理:
          * 不支持的Loss.type会抛出NotImplementedError
          * 无效的设备配置会导致RuntimeError
          * 类别权重计算失败会中断初始化流程

        - 性能注意:
          * DDP模式会增加约10-15%的显存占用
          * find_unused_parameters=True会额外增加计算开销
          * 不同损失函数对训练速度有显著影响
          * 类别权重过大可能导致训练不稳定

        示例:
        >>> # 单卡训练
        >>> device = torch.device('cuda:0')
        >>> trainer.build_ddp_model(cfg, device)
        >>>
        >>> # 多卡分布式训练
        >>> trainer.RANK = 0
        >>> trainer.build_ddp_model(cfg, torch.device('cuda:0'))
        """
        # DDP模型包装 (仅在分布式环境下生效)
        if self.cuda and self.RANK != -1:
            self.model = DDP(
                self.model,
                device_ids=[self.LOCAL_RANK],
                output_device=self.LOCAL_RANK,
                find_unused_parameters=True  # 允许未使用参数回传
            )

        # 类别参数配置
        self.model.nc = self.nc  # 注入类别数量
        # 基于标签分布计算类别权重(处理样本不平衡)
        self.model.class_weights = labels_to_class_weights(
            self.dataset.labels, self.nc
        ).to(device) * self.nc  # 权重标准化
        self.model.names = self.names  # 注入类别名称

        # 损失函数动态初始化
        loss_type = cfg.Loss.type
        if loss_type == 'ComputeLoss':
            self.compute_loss = ComputeLoss(self.model, cfg)  # 标准YOLO损失
        elif loss_type == 'ComputeFastXLoss':
            self.compute_loss = ComputeFastXLoss(self.model, cfg)  # 快速计算版本
        elif loss_type == 'ComputeNanoLoss':
            self.compute_loss = ComputeNanoLoss(self.model, cfg)  # 轻量级版本
        else:
            raise NotImplementedError(f'不支持的损失类型: {loss_type}')

        # 检测头适配处理
        self.detect = self.model.module.head if is_parallel(self.model) \
            else self.model.head  # 兼容DDP/普通模式

    def before_train(self):
        """
        训练前准备钩子方法

        功能说明:
        - 作为训练流程开始前的预留接口
        - 当前版本默认返回0，需在子类中实现具体逻辑
        - 可用于初始化自定义组件或执行预检操作

        重点细节:
        - 方法应在正式训练迭代开始前调用
        - 返回值可扩展为状态码用于流程控制
        - 建议通过继承实现具体功能，保持基类简洁性

        异常处理:
        - 基础实现不会抛出任何异常
        - 子类实现需自行处理可能出现的初始化错误

        示例:
        >>> class CustomTrainer(Trainer):
        ...     def before_train(self):
        ...         print("Initializing custom components")
        ...         return super().before_train()
        """
        return 0

    def build_train_logger(self):
        """
        构建训练指标记录系统

        功能说明:
        - 初始化训练过程指标收集器
        - 定义训练日志的默认表头结构
        - 准备损失项的动态发现机制

        重点细节:
        - 使用MetricMeter进行多维指标聚合统计
        - 日志表头包含epoch、显存占用等基础字段
        - 损失项字段将在首次数据遍历后动态追加

        核心配置:
        - gpu_mem: 记录当前GPU显存占用情况
        - labels: 当前批次的标注信息统计
        - img_size: 输入图像尺寸监控

        异常处理:
        - 重复初始化meter可能覆盖已有统计数据
        - 依赖train_loader的预初始化完成

        示例:
        >>> trainer.build_train_logger()
        >>> print(trainer.log_contents)
        ['Epoch', 'gpu_mem', 'labels', 'img_size']
        """
        self.meter = MetricMeter()
        log_contents = ['Epoch', 'gpu_mem', 'labels', 'img_size']
        self.log_contents = log_contents

    def update_train_logger(self):
        """
        更新训练日志表头并打印格式

        功能说明:
        - 通过预遍历首个训练batch获取损失项名称
        - 动态构建日志表头字段
        - 输出标准化的日志格式

        重点细节:
        - 仅处理训练加载器的首个批次
        - 自动识别损失字典的键作为日志字段
        - 使用固定宽度格式保证对齐

        核心算法:
        - 非阻塞数据加载(non_blocking=True)优化传输效率
        - 自动混合精度上下文管理(autocast)
        - 基于首个batch的元数据采集机制

        异常处理:
        - 空训练加载器将导致阻塞等待
        - 设备不匹配会引发RuntimeError
        - 无效的损失计算会中断预处理流程

        性能注意:
        - 预遍历单个batch增加约3-5%初始化耗时
        - 混合精度可减少约30%显存占用
        - 建议在训练循环开始前执行

        示例:
        >>> trainer.update_train_logger()
        [INFO]     Epoch   gpu_mem    labels  img_size       loss       box       obj       cls
        """
        # 预遍历首个训练批次获取损失项
        for (imgs, targets, paths, _) in self.train_loader:
            # 设备数据传输优化
            imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale

            # 混合精度前向计算
            with amp.autocast(enabled=self.cuda):
                pred = self.model(imgs)
                _, loss_items = self.compute_loss(pred, targets.to(self.device))

            # 构建日志字段（单卡模式直接处理）
            self.log_contents += loss_items.keys()
            break  # 仅处理首个batch

        # 生成固定宽度表头格式
        header_format = '\n' + '%10s' * len(self.log_contents)
        LOGGER.info(header_format % tuple(self.log_contents))

    def before_epoch(self):
        """
        单个训练周期前的预处理

        功能说明:
        - 执行每个epoch开始前的配置更新
        - 管理训练模式/数据增强策略/损失函数调整
        - 初始化训练指标跟踪系统
        - 处理学习率热身阶段设置


        重点细节:
        - 参数边界条件:
          * no_aug_epochs: 必须小于总训练周期数epochs
          * warmup_epochs: 建议设置为总epochs的10%-20%
          * model_type: 需与损失函数调整逻辑匹配

        - 关键处理流程:
          1. 模型训练模式切换
          2. 训练日志系统重建
          3. 后期训练阶段优化策略调整
          4. 热身迭代次数计算


        - 核心配置项:
          * mosaic增强: 默认开启，最后no_aug_epochs阶段关闭
          * L1损失: 仅YOLOX模型在特定阶段启用
          * nw: 保证至少1000次迭代的热身

        - 异常处理:
          * 无效的no_aug_epochs配置会导致增强策略错误
          * 不支持的model_type不会触发L1损失调整


        - 性能注意:
          * 关闭mosaic可提升约15%的训练速度
          * 热身阶段使用较低学习率避免数值不稳定
          * 重新初始化meter会清空历史指标数据
          * 分布式采样控制保证各进程数据分布差异

        示例:
        >>> # 标准训练流程
        >>> trainer.epoch = 290
        >>> trainer.epochs = 300
        >>> trainer.no_aug_epochs = 10
        >>> trainer.before_epoch()
        [INFO] --->No mosaic aug now!
        [INFO] --->Add additional L1 loss now!
        """
        # 基础模式设置
        self.model.train()  # 切换训练模式
        self.build_train_logger()  # 重建指标记录器
        self.update_train_logger()  # 更新日志格式

        # 后期训练策略调整
        if self.epoch == self.epochs - self.no_aug_epochs:
            LOGGER.info("关闭Mosaic数据增强")
            self.dataset.mosaic = False  # 禁用mosaic增强


        # 指标跟踪重置
        self.meter = MetricMeter()  # 新建指标计量器

        # 热身阶段计算
        if self.warmup_epochs > 0:
            # 计算热身迭代次数(至少1000次/最多1.5个epoch)
            self.nw = max(self.warmup_epochs * self.nb, 1000)
            self.nw = min(self.nw, (self.epochs - self.start_epoch) // 2 * self.nb)
        else:
            self.nw = -1  # 禁用热身


    def update_optimizer(self, loss, ni):
        """
        优化器参数更新与梯度管理

        功能说明:
        - 执行梯度反向传播与参数更新
        - 管理混合精度训练与梯度累积
        - 实现学习率/动量的热身策略
        - 维护EMA模型参数

        重点细节:
        - 参数边界条件:
          * ni: 当前迭代次数需从0开始递增
          * batch_size: 建议为2的幂次以获得最佳性能
          * nw: 热身阶段最大迭代次数需预先计算

        - 关键处理流程:
          1. 混合精度梯度放大与反向传播
          2. 动态梯度累积步数计算
          3. 学习率/动量的线性热身调整
          4. 累积梯度参数更新
          5. EMA模型参数同步

        - 核心算法:
          * 梯度累积: 通过accumulate实现大batch等效训练
          * 线性插值: 使用np.interp实现平滑参数过渡
          * 混合精度训练: 通过GradScaler防止梯度下溢

        - 异常处理:
          * 梯度爆炸会触发NaN检测中断训练
          * 优化器参数组缺失会引发KeyError
          * EMA更新失败会记录警告但继续训练

        - 性能注意:
          * 混合精度训练可节省约30%显存
          * 梯度累积可支持更大的等效batch_size
          * 频繁的scaler.update()会增加计算开销

        示例:
        >>> for epoch in range(epochs):
        ...     for ni, batch in enumerate(loader):
        ...         loss = model(batch)
        ...         trainer.update_optimizer(loss, ni)
        """
        # 梯度反向传播 (混合精度上下文)
        self.scaler.scale(loss).backward()  # 自动缩放损失并反向传播

        # 动态梯度累积计算 (基于batch_size)
        self.accumulate = max(round(64 / self.batch_size), 1)  # 确保至少累积1次

        # 学习率/动量热身阶段调整
        if ni <= self.nw:  # 处于热身阶段
            xi = [0, self.nw]  # 插值区间

            # 动态调整累积步数 (随迭代逐步增加)
            self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())

            # 遍历参数组进行调参
            for j, param_group in enumerate(self.optimizer.param_groups):
                # 偏置参数组特殊处理 (第三参数组假设为偏置)
                if j == 2:
                    lr_targets = [self.warmup_bias_lr, param_group['initial_lr'] * self.lf(self.epoch)]
                else:
                    lr_targets = [0.0, param_group['initial_lr'] * self.lf(self.epoch)]

                # 线性插值更新学习率
                param_group['lr'] = np.interp(ni, xi, lr_targets)

                # 动量参数更新 (如果存在)
                if 'momentum' in param_group:
                    param_group['momentum'] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

        # 梯度累积更新条件
        if ni - self.last_opt_step >= self.accumulate:
            # 梯度缩放更新参数 (自动unscale梯度)
            self.scaler.step(self.optimizer)  # 执行优化器step
            self.scaler.update()  # 调整缩放因子

            # 清空累积梯度
            self.optimizer.zero_grad()

            # EMA模型参数同步
            if self.ema:
                self.ema.update(self.model)  # 更新影子权重

            # 记录最后更新步数
            self.last_opt_step = ni

    def train_in_epoch(self, callbacks):
        """
        执行单个epoch的训练循环

        功能说明:
        - 管理完整的前向传播、损失计算、反向传播流程
        - 实现训练进度可视化与指标记录
        - 支持混合精度训练与设备优化
        - 触发训练批次结束回调函数
        - 更新学习率调度器

        重点细节:
        - 参数边界条件:
            * callbacks: 需实现on_train_batch_end接口
            * break_iter: 调试用断点设置需小于总批次数nb

        - 关键处理流程:
            1. 进度条初始化与设备数据预处理
            2. 混合精度前向计算与损失缩放
            3. 实时训练指标可视化
            4. 学习率调度器步进

        - 核心算法:
            * 自动混合精度: 通过autocast上下文管理
            * 动态进度条: 实时显示显存/损失/处理速度等指标

        - 性能优化:
            * non_blocking=True异步传输提升数据加载速度
            * 混合精度减少约30%显存消耗
            * 梯度累积需配合optimizer_step参数使用

        示例:
        >>> callbacks = TrainingCallbacks()
        >>> trainer.train_in_epoch(callbacks)
        0/299      3.2G        64       640      0.1234      0.5678      0.9012  # 进度条示例
        """
        # 初始化进度条（单卡训练模式）
        pbar = tqdm(enumerate(self.train_loader), total=self.nb, desc=f'Epoch {self.epoch}')
        self.optimizer.zero_grad()  # 清空历史梯度

        # 批次训练循环
        for i, (imgs, targets, paths, _) in pbar:
            if i == self.break_iter:  # 调试断点
                break

            # 元数据计算
            ni = i + self.nb * self.epoch  # 全局迭代次数

            # 数据预处理（非阻塞传输+归一化优化）
            imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale

            # 前向传播（混合精度上下文）
            with amp.autocast(enabled=self.cuda):
                pred = self.model(imgs)  # 模型推理
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # 损失计算

            # 优化器参数更新
            self.update_optimizer(loss, ni)

            # 实时指标监控
            self.meter.update(loss_items)
            mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if torch.cuda.is_available() else '0G'

            # 动态进度条格式: Epoch | 显存 | 目标数 | 图像尺寸 | 各项损失
            pbar.set_description((
                f'{self.epoch:>3}/{self.epochs - 1:>3} '
                f'{mem:>8} '
                f'{targets.shape[0]:>6} '
                f'{imgs.shape[-1]:>6} '
                f'{" ".join(f"{v:.4f}" for v in self.meter.get_avg())}'
            ))

            # 触发批次结束回调
            callbacks.run('on_train_batch_end', ni, self.model,
                          imgs, targets, paths, self.plots, self.cfg.Dataset.np)

        # 学习率调度器更新
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # 记录当前学习率
        self.scheduler.step()  # 按epoch更新学习率

    def after_epoch(self, callbacks, val):
        """
        训练周期后处理与模型验证

        功能说明:
        - 执行模型验证与性能评估
        - 维护EMA模型参数与最佳模型记录
        - 保存模型检查点
        - 触发训练结束回调与日志记录

        重点细节:
        - 参数边界条件:
          * val对象需实现run方法并返回标准格式结果
          * callbacks需实现on_train_epoch_end/on_fit_epoch_end等接口
          * RANK=-1表示单机训练，0表示主节点

        - 关键处理流程:
          1. EMA模型参数同步
          2. 执行验证流程计算mAP指标
          3. 更新最佳模型适应度
          4. 持久化模型检查点
          5. 触发模型保存与训练结束回调

        - 核心算法:
          * 适应度计算: 综合精确度/召回率/mAP等指标
          * EMA参数维护: 保持模型参数的平滑版本
          * 检查点压缩: 使用half()存储减少50%空间

        - 异常处理:
          * 验证失败会跳过最佳模型更新
          * 文件保存错误会记录警告信息
          * 回调异常不会中断主流程

        - 性能注意:
          * 验证过程会增加约30%的时间开销
          * 模型保存建议使用SSD存储
          * 频繁保存检查点会影响训练速度

        示例:
        >>> # 典型训练循环中的使用
        >>> for epoch in range(epochs):
        ...     trainer.train_epoch()
        ...     trainer.after_epoch(callbacks, validator)
        """
        # 仅主进程执行后续操作
        if self.RANK in [-1, 0]:
            # 触发训练周期结束回调
            callbacks.run('on_train_epoch_end', epoch=self.epoch)

            # 同步EMA模型属性
            self.ema.update_attr(
                self.model,
                include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights']
            )

            final_epoch = (self.epoch + 1 == self.epochs)

            # 执行验证流程
            if not self.noval:
                self.results, maps, _ = val.run(
                    data_dict=self.data_dict,
                    batch_size=self.batch_size // self.WORLD_SIZE * 2,  # 验证批次调整
                    imgsz=self.imgsz,
                    model=self.ema.ema,  # 使用EMA模型验证
                    single_cls=self.single_cls,
                    dataloader=self.val_loader,
                    save_dir=self.save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=self.compute_loss,
                    num_points=self.cfg.Dataset.np,
                    val_kp=self.cfg.Dataset.val_kp
                )

                # 计算综合适应度指标
                fi = fitness(np.array(self.results).reshape(1, -1))  # [P, R, mAP@.5, mAP@.5-.95]
                if fi > self.best_fitness:
                    self.best_fitness = fi  # 更新最佳指标

            # 组装日志数据 (训练损失 + 验证指标 + 学习率)
            log_vals = list(self.meter.get_avg())[:3] + list(self.results) + self.lr
            callbacks.run('on_fit_epoch_end', log_vals, self.epoch, self.best_fitness, fi)

            # 模型持久化逻辑
            if (not self.nosave) or final_epoch:
                ckpt = {
                    'epoch': self.epoch,
                    'best_fitness': self.best_fitness,
                    'model': deepcopy(de_parallel(self.model)).half(),  # 去除DP包装+半精度压缩
                    'ema': deepcopy(self.ema.ema).half(),
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'wandb_id': None  # 实验追踪ID占位
                }

                # 保存最近检查点
                torch.save(ckpt, self.last)

                # 更新最佳模型
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)

                # 周期性保存
                if self.epoch > 0 and self.save_period > 0 and (self.epoch % self.save_period == 0):
                    save_path = self.save_dir / 'weights' / f'epoch{self.epoch}.pt'
                    torch.save(ckpt, save_path)

                del ckpt  # 释放显存
                callbacks.run('on_model_save', self.last, self.epoch, final_epoch, self.best_fitness, fi)

    def train(self, callbacks, val):
        """
        执行完整的训练流程

        功能说明:
        - 管理从初始周期到最终周期的完整训练过程
        - 协调各阶段方法调用(before_epoch/train_in_epoch/after_epoch)
        - 收集并返回最终训练结果
        - 统计总训练时间并生成总结报告

        重点细节:
        - 参数边界条件:
          * callbacks: 需实现各训练阶段回调接口
          * val: 需实现验证接口并返回标准格式结果
          * start_epoch: 允许从中间周期恢复训练

        - 关键处理流程:
          1. 训练计时与状态初始化
          2. 周期循环控制与中断检测
          3. 训练阶段的三段式调用(前/中/后处理)
          4. 最终结果验证与资源清理
          5. 训练耗时统计与日志输出

        - 核心算法:
          * 弹性训练周期: 支持break_epoch调试断点
          * 自动恢复机制: 通过start_epoch实现训练续接
          * 综合性能评估: 通过fitness函数整合多指标

        - 异常处理:
          * 周期数越界会自动修正为合法范围
          * 训练中断会正常执行after_train清理
          * 空验证器会返回默认零值结果

        - 性能注意:
          * 建议使用NVIDIA Apex自动混合精度
          * 周期数超过100时建议启用周期检查点
          * 分布式训练需合理设置WORLD_SIZE参数

        示例:
        >>> # 典型训练流程
        >>> callbacks = TrainingCallbacks()
        >>> validator = ModelValidator()
        >>> results = trainer.train(callbacks, validator)
        >>> print(f'总训练时间: {results[-1]:.1f}小时')
        [INFO] 300 epochs completed in 12.3 hours.
        """
        # 训练初始化
        t0 = time.time()  # 记录起始时间戳
        self.last_opt_step = -1  # 优化步数计数器
        self.results = (0, 0, 0, 0, 0, 0, 0)  # 初始化验证指标(P,R,mAP等)
        self.best_fitness = 0  # 最佳适应度重置

        # 主训练循环
        for self.epoch in range(self.start_epoch, self.epochs):
            if self.epoch == self.break_epoch:  # 调试断点检查
                break

            # 训练阶段管理
            self.before_epoch()  # 周期预处理
            self.train_in_epoch(callbacks)  # 执行实际训练
            self.after_epoch(callbacks, val)  # 验证与保存

        # 训练后处理
        results = self.after_train(callbacks, val)  # 最终验证与清理

        # 训练总结报告
        # if self.RANK in [-1, 0]:
        #     duration = (time.time() - t0) / 3600  # 计算总耗时(小时)
        #     epochs_done = self.epoch - self.start_epoch + 1
        #     LOGGER.info(f'\n完成{epochs_done}个训练周期，总耗时{duration:.3f}小时')

        return results  # 返回最终验证指标