import copy
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import sklearn.mixture as skm
from utils.general import non_max_suppression_ssod, xyxy2xywh, xywh2xyxy
from utils.plots import plot_images_ssod, plot_images
from utils.self_supervised_utils import online_label_transform

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """执行多GPU间的全收集操作并沿指定维度拼接张量

    功能:
        - 分布式训练时跨进程聚合所有GPU上的相同张量
        - 通过torch.distributed.all_gather实现数据同步
        - 最终输出是沿指定维度拼接的全局张量

    注意:
        - 无梯度计算: 被@torch.no_grad()装饰且使用无梯度all_gather
        - 同步操作: async_op=False确保所有进程同步完成
        - 内存预分配: 预创建ones_like张量会被各进程的实际张量覆盖
        - 维度控制: dim参数决定拼接方向，默认第0维度

    参数:
        tensor: 当前进程需要聚合的输入张量
        dim: 拼接维度，默认为0

    返回:
        拼接后的全局张量(包含所有进程的数据)
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=dim)


class LabelMatch(nn.Module):
    """半监督目标检测中的动态标签匹配与阈值调整模块

    核心功能:
        - 管理类别分数队列实现动态阈值计算
        - 根据历史分数分布调整伪标签采样策略
        - 处理多GPU分布式环境下的数据同步

    重要组件:
        - 分数队列: score_queue存储历史检测分数用于分布统计
        - 动态阈值: 根据队列数据计算高低阈值过滤伪标签
        - 类别平衡: 基于cls_ratio_gt实现类感知采样

    关键细节:
        - 环形队列: queue_ptr实现循环覆盖写入，queue_len=类别数*队列容量
        - 多GPU同步: _dequeue_and_enqueue方法内使用concat_all_gather聚合数据
        - 延迟更新: start_update标志控制队列填满后才开始阈值计算
        - 配置驱动: 从cfg加载NMS参数、采样比例等超参数
        - 缓冲区注册: score_queue和queue_ptr通过register_buffer持久化存储
    """

    def __init__(self, cfg, target_data_len, label_num_per_img, cls_ratio_gt):
        super().__init__()
        # 基础配置参数
        self.nc = cls_ratio_gt.shape[0]  # 类别总数
        self.multi_label = cfg.SSOD.multi_label  # 是否多标签模式
        self.nms_conf_thres = cfg.SSOD.nms_conf_thres  # NMS置信度阈值
        self.nms_iou_thres = cfg.SSOD.nms_iou_thres  # NMS IoU阈值
        self.cls_thr_high = [cfg.SSOD.ignore_thres_high] * self.nc  # 初始高阈值
        self.cls_thr_low = [cfg.SSOD.ignore_thres_low] * self.nc  # 初始低阈值
        self.max_cls_per_img = 10  # 单图最大类别数
        self.boxes_per_image_gt = 7  # 每图基础标注框数量
        self.queue_num = 128  # 队列基础容量
        self.queue_len = self.queue_num * self.max_cls_per_img  # 队列总长度

        # 状态管理与数据存储
        self.register_buffer("score_queue", torch.zeros(self.nc, self.queue_len))  # 分数环形队列
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # 队列写入指针
        self.score_list = [[] for _ in range(self.nc)]  # 临时分数存储
        self.start_update = False  # 队列填满标志

        # 分布参数与统计量
        self.cls_ratio_gt = cls_ratio_gt  # 各类别真实分布比例
        self.cls_num_list = [0 for _ in range(self.nc)]  # 类别计数
        self.target_data_len = target_data_len  # 目标数据总量
        self.anno_num_per_img = label_num_per_img * 3  # 每图标注数量系数

    def _dequeue_and_enqueue(self, score):
        """更新环形分数队列的核心方法

        处理逻辑:
        1. 多GPU数据聚合: 通过concat_all_gather收集所有进程的分数
        2. 环形写入: 根据queue_ptr指针循环覆盖旧数据
        3. 延迟触发: 队列首次填满时设置start_update标志
        4. 指针管理: 写入后更新queue_ptr并取模实现环形索引

        注意:
        - 输入score形状为[类别数, batch_size]
        - 要求队列长度能被batch_size整除以保证完整覆盖
        - 分布式模式下会聚合所有GPU的分数数据
        """
        if dist.is_initialized():
            score = concat_all_gather(score, 1)  # 在维度1拼接多GPU数据
        batch_size = score.shape[1]
        ptr = int(self.queue_ptr)

        # 环形缓冲区写入
        if ptr + batch_size > self.queue_len:
            overflow = ptr + batch_size - self.queue_len
            self.score_queue[:, ptr:] = score[:, :self.queue_len - ptr]
            self.score_queue[:, :overflow] = score[:, -overflow:]
        else:
            self.score_queue[:, ptr:ptr + batch_size] = score

        # 更新指针与状态标志
        if not self.start_update and ptr + batch_size >= self.queue_len:
            self.start_update = True
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_len

    def update(self, labels, n=1, pse_n=1):
        """更新类别分布统计量的辅助方法

        功能:
            - 累计总样本计数和伪标签计数
            - 统计每个类别的出现次数到临时变量

        参数:
            labels: 当前批次的标签数据，形状为[N,6](xyxy+class)
            n: 真实标签计数增量，默认为1
            pse_n: 伪标签计数增量，默认为1

        注意:
            - cls_tmp使用numpy数组存储临时类别计数
            - 标签解析逻辑: 取labels第二列作为类别索引
            - 累计计数器: count记录总样本数，pse_count记录伪标签数
        """
        self.count += n
        self.pse_count += pse_n
        for l in labels:
            self.cls_tmp[int(l[1:2])] += 1

    def gmm_policy(self, scores, given_gt_thr=0.5, policy='high'):
        """基于高斯混合模型的动态阈值计算策略

        核心逻辑:
            1. 数据准备: 样本不足时返回默认阈值
            2. GMM初始化: 用最大最小分数初始化双峰高斯分布
            3. 策略执行:
               - high策略: 选择正类簇中最低分数作为阈值
               - middle策略: 取正类簇的最小分数

        参数:
            scores: 预测框的置信分数，形状为[N,1]
            given_gt_thr: 保底阈值，默认0.5
            policy: 阈值策略，可选'high'/'middle'

        返回:
            pos_thr: 计算得到的动态阈值

        关键技术:
            - 异常处理: 无正类簇时返回给定阈值
            - 分数截断: 最终阈值不低于given_gt_thr
            - 维度处理: 单维分数扩展为二维适配GMM输入
        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]

        means_init = [[np.min(scores)], [np.max(scores)]]
        gmm = skm.GaussianMixture(2, weights_init=[1 / 2, 1 / 2],
                                  means_init=means_init,
                                  precisions_init=[[[1.0]], [[1.0]]])
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)

        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores = gmm.score_samples(scores)
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores)
                pos_indx = (gmm_assignment == 1) & (scores >= scores[indx]).squeeze()
                pos_thr = max(given_gt_thr, float(scores[pos_indx].min()))
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            pos_thr = max(given_gt_thr, float(scores[gmm_assignment == 1].min())) \
                if (gmm_assignment == 1).any() else given_gt_thr
        return pos_thr

    def update_epoch_cls_thr(self, epoch):
        """基于周期分数更新类别置信度阈值的核心方法

        功能流程:
        1. 遍历所有类别，分别处理每个类别的历史分数
        2. 排序分数后根据采样比例确定高低阈值位置
        3. 应用GMM策略动态计算高置信度阈值
        4. 重置周期统计量为下一轮训练做准备

        关键操作:
        - 分数排序: 单类别分数降序排列便于分位数计算
        - 动态调整: 使用resample_high_percent/resample_low_percent控制采样比例
        - GMM融合: 对高阈值采用高斯混合模型策略提升阈值准确性
        - 安全边界: 最终阈值不低于预设的ignore_thres_low/high值

        重要细节:
        - 周期依赖: max_pseudo_label_num计算包含epoch参数实现渐进式调整
        - 空数据处理: 分数列表为空时直接使用预设忽略阈值
        - 日志追踪: 记录每个类别的阈值计算过程用于调试
        - 状态重置: 清空score_list_epoch等临时存储防止数据污染
        """
        for cls in range(self.nc):
            single_cls_score = self.score_list_epoch[cls]
            single_cls_score.sort(reverse=True)
            max_pseudo_label_num = int(self.cls_num_total[cls] / (epoch + 1))

            if not single_cls_score:
                self.cls_thr_high[cls] = self.ignore_thres_high
                self.cls_thr_low[cls] = self.ignore_thres_low
            else:
                pos_loc_high = int(len(single_cls_score) * self.resample_high_percent)
                pos_loc_low = min(max_pseudo_label_num, int(len(single_cls_score) * self.resample_low_percent))
                self.cls_thr_high[cls] = self.gmm_policy(np.array(single_cls_score), policy='high')
                self.cls_thr_low[cls] = max(self.ignore_thres_low, single_cls_score[pos_loc_low])

            LOGGER.info(
                f'Class {cls}: {len(single_cls_score)} samples | HighThr={self.cls_thr_high[cls]:.2f} LowThr={self.cls_thr_low[cls]:.2f}')

        # 重置周期统计量
        self.score_list_epoch = [[] for _ in range(self.nc)]
        self.cls_tmp = np.zeros(self.nc)
        self.count = 0
        self.pse_count = 0

    # def update_cls_thr_old(self):
    #     for cls in range(self.nc):
    #         if len(self.score_list[cls]) >= 1:
    #             self.score_list[cls].sort()
    #             cur = int(len(self.score_list[cls])/2)
    #             s = self.score_list[cls][cur]
    #         else:
    #             s = 0.1
    #         self.cls_thr[cls] = max(0.1, s)
    #     info = ' '.join([f'({v:.2f}-{i})' for i, v in enumerate(self.cls_thr)])
    #     LOGGER.info(f'update score thr (positive): {info}')

    # def update_cls_thr(self):
    #     #等待队列收满才开始更新得分阈值
    #     if self.start_update:
    #         score, _ = self.score_queue.sort(dim=0, descending=True)
    #         boxes_per_img = self.boxes_per_image_gt * self.percent
    #         pos_location = self.queue_num * boxes_per_img * self.cls_ratio_gt
    #         # pos_location = [0] * self.nc
    #         # print('pos_location:', pos_location)
    #         for cls in range(self.nc):
    #             LOGGER.info(f'{pos_location[cls]}:{score[cls, int(pos_location[cls])]}')
    #             self.cls_thr[cls] = max(self.ignore_thres, score[cls, int(pos_location[cls])].item())
    #         # info = ' '.join([f'({v:.2f}-{i})' for i, v in enumerate(self.cls_thr)])
    #         info = ' '.join([f'({v:.2f}-{i})' for i, v in enumerate(score[:, 1])])
    #         LOGGER.info(f'update score thr (positive): {info}')

    # def clean_score_list(self):
    # self.score_list = [[] for _ in range(self.nc)]

    def create_pseudo_label_online_with_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        """
        在线生成伪标签并结合真实标签进行几何变换校正
        主要流程：
        1. 使用改进的非极大值抑制筛选初步预测结果
        2. 基于置信度分数动态维护类别分数池
        3. 对保留的预测框进行几何变换校正
        4. 坐标归一化及镜像翻转补偿

        参数说明：
        out: 模型原始输出
        target_imgs: 增强后的目标图像
        M_s: 增强变换矩阵集合
        target_imgs_ori: 原始目标图像(未增强)
        gt: 真实标签(用于调试)
        RANK: 进程标识

        重点细节：
        - 使用非极大值抑制时配置了特殊参数(num_points, multi_label)
        - 维护score_list_epoch用于动态阈值计算
        - 几何变换包含旋转、缩放、翻转等空间变换
        - 坐标系统经过归一化处理并限制在[0,1]范围
        - 支持上下/左右翻转后的坐标补偿
        - 包含调试可视化功能(需开启debug模式)
        """
        n_img, _, height, width = target_imgs.shape
        lb = []

        # 改进版非极大值抑制，增加关键点数量和多标签处理
        pseudo_out = non_max_suppression_ssod(out, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres,
                                              labels=lb, num_points=self.num_points, multi_label=self.multi_label)

        refine_out = []
        score_list_batch = [[] for _ in range(self.nc)]
        # 置信度分数池维护(按类别)
        for i, o in enumerate(pseudo_out):
            for *box, conf, cls, obj_conf, cls_conf in o.cpu().numpy():
                score_list_batch[int(cls)].append(float(conf))
                refine_out.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, obj_conf, cls_conf])
                self.score_list_epoch[int(cls)].append(float(conf))

        # 分数池填充对齐(max_cls_per_img控制每类最大数量)
        for c in range(len(score_list_batch)):
            if len(score_list_batch[c]) < self.max_cls_per_img:
                score_list_batch[c].extend([0.0] * (self.max_cls_per_img - len(score_list_batch[c])))
                score_list_batch[c].sort(reverse=True)
            else:
                score_list_batch[c].sort(reverse=True)
                score_list_batch[c] = score_list_batch[c][:self.max_cls_per_img]

        # 几何变换处理流程
        refine_out = np.array(refine_out)
        target_out_targets = torch.tensor(refine_out)
        target_shape = target_out_targets.shape
        target_out_targets_perspective = []
        invalid_target_shape = True

        if target_shape[0] > 0 and target_shape[1] > 6:
            # 逐图像进行空间变换校正
            for i, img in enumerate(target_imgs_ori):
                image_targets = refine_out[refine_out[:, 0] == i]
                # 坐标格式转换(xywh->xyxy)
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()
                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])

                # 提取变换参数(3x3矩阵+缩放因子+翻转标志)
                M_select = M_s[M_s[:, 0] == i, :]
                M = M_select[0][1:10].reshape([3, 3]).cpu().numpy()
                s = float(M_select[0][10])
                ud = int(M_select[0][11])
                lr = int(M_select[0][12])

                # 应用在线空间变换
                img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]), M, s)

                # 后处理：坐标归一化+镜像补偿
                if image_targets.shape[0] != 0:
                    image_targets = np.concatenate((np.ones([image_targets.shape[0], 1]) * i, image_targets), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])
                    image_targets[:, [3, 5]] /= height  # y/h
                    image_targets[:, [2, 4]] /= width  # x/w
                    image_targets[:, 2:6] = image_targets[:, 2:6].clip(0, 1)
                    # 翻转补偿逻辑
                    if ud == 1: image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1: image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())

            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))

        # 调试可视化(需满足RANK条件)
        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0:
            invalid_target_shape = False
            if self.debug and RANK in [-1, 0]:
                draw_image = plot_images_ssod(copy.deepcopy(target_imgs), target_out_targets_perspective,
                                              fname='/mnt/bowen/EfficientTeacher/effcient_teacher_pseudo_label_ea.jpg',
                                              names=self.names)
                draw_image = plot_images(copy.deepcopy(target_imgs), gt,
                                         fname='/mnt/bowen/EfficientTeacher/effcient_teacher_gt_ea.jpg',
                                         names=self.names)

        return target_out_targets_perspective, invalid_target_shape