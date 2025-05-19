import torch
import torchvision
import cv2
import numpy as np
from utils.general import xyxy2xywh, xywh2xyxy, non_max_suppression, segment2box, resample_segments
from utils.general import non_max_suppression_ssod
from utils.metrics import box_iou
from utils.plots import plot_images_ssod, plot_images, output_to_target_ssod
import copy

class FairPseudoLabel:
    def __init__(self, cfg):
        """
        公平伪标签生成器
        功能：初始化伪标签生成所需参数，包含NMS阈值、调试模式等配置
        参数说明：
        cfg: 配置文件对象，包含以下关键配置项：
            - SSOD.nms_conf_thres: NMS置信度阈值
            - SSOD.nms_iou_thres: NMS IoU阈值
            - SSOD.debug: 调试模式开关
            - SSOD.multi_label: 多标签模式开关
            - Dataset.names: 类别名称列表
            - Dataset.np: 关键点数量
        重点细节：
        - 同时支持目标检测和关键点检测(num_points参数)
        - 配置参数来源于SSOD(半监督目标检测)和Dataset两个配置模块
        """
        self.nms_conf_thres = cfg.SSOD.nms_conf_thres
        self.nms_iou_thres = cfg.SSOD.nms_iou_thres
        self.debug = cfg.SSOD.debug
        self.multi_label = cfg.SSOD.multi_label
        self.names = cfg.Dataset.names
        self.num_points = cfg.Dataset.np

    def online_label_transform_with_image(self, img, targets, M, s, ud, lr, segments=(), border=(0, 0), perspective=0.0):
        """
        在线图像增强与目标坐标同步变换
        功能：对图像进行空间变换(仿射/透视)并同步调整目标坐标
        参数说明：
        img: 输入图像(Tensor或numpy格式)
        targets: 关联的检测目标(格式需与图像变换匹配)
        M: 3x3变换矩阵
        s: 缩放因子
        ud: 上下翻转标志(1表示翻转)
        lr: 左右翻转标志(1表示翻转)
        segments: 实例分割多边形(暂未使用)
        border: 图像边框填充像素数
        perspective: 透视变换强度系数

        重点细节：
        - 支持Tensor与numpy格式的自动转换
        - 同时处理仿射变换(warpAffine)和透视变换(warpPerspective)
        - 保持图像与目标的空间一致性
        - 翻转操作使用numpy的flipud/fliplr实现
        - 返回标准化后的Tensor格式图像(HWC->CHW, 0-1范围)
        """
        # 格式统一处理(Tensor->numpy)
        if isinstance(img, torch.Tensor):
            img = img.cpu().float().numpy()
            img = img.transpose(1, 2, 0) * 255.0
            img = img.astype(np.uint8)

        # 图像尺寸计算(含边框)
        height = img.shape[0] + border[0] * 2
        width = img.shape[1] + border[1] * 2

        # 空间变换核心逻辑
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # 镜像增强处理
        if ud == 1: img = np.flipud(img)
        if lr == 1: img = np.fliplr(img)

        # 输出格式标准化
        img = torch.from_numpy(img.transpose(2, 0, 1 ) /255.0)

        # 目标坐标变换处理流程
        n = len(targets)
        if n:
            # 判断是否使用分割数据(存在有效分割多边形时启用)
            use_segments = any(x.any() for x in segments)
            new = np.zeros((n, 4))

            # 分割数据处理分支
            if use_segments:
                segments = resample_segments(segments)  # 上采样增加分割点密度
                for i, segment in enumerate(segments):
                    # 构建齐次坐标并进行矩阵变换
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T  # 应用仿射/透视变换矩阵

                    # 透视变换处理：坐标归一化(z分量处理)
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]

                    # 将变换后的分割点转换为边界框
                    new[i] = segment2box(xy, width, height)

            # 边界框处理分支
            else:
                # 构建四角点坐标矩阵(每个框扩展为4个角点)
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # 按x1y1,x2y2,x1y2,x2y1顺序重组坐标

                # 应用空间变换矩阵
                xy = xy @ M.T
                # 透视变换归一化处理并恢复为(n,8)形状
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

                # 计算新边界框坐标
                x = xy[:, [0, 2, 4, 6]]  # 提取所有x坐标
                y = xy[:, [1, 3, 5, 7]]  # 提取所有y坐标
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # 创建最小包围框

                # 坐标裁剪(防止超出图像边界)
                new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)  # x轴范围限制
                new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)  # y轴范围限制

            # 候选框过滤策略
            # 计算原始框面积与变换后面积的比率阈值
            i = box_candidates(
                box1=targets[:, 1:5].T * s,  # 原始框尺寸乘以缩放因子
                box2=new.T,  # 变换后的新框
                area_thr=0.01 if use_segments else 0.10  # 面积变化阈值(分割模式更严格)
            )
            targets = targets[i]  # 筛选有效目标
            targets[:, 1:5] = new[i]  # 更新坐标数据

        return img, targets

    # def create_pseudo_label_online(self, out, target_imgs, M_s, target_imgs_ori, gt=None):
    #     n_img, _, height, width = target_imgs.shape  # batch size, channels, height, width
    #     lb = []
    #     target_out_targets_perspective = []
    #     invalid_target_shape = True

    #     out = non_max_suppression_ssod(out, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres, multi_label=self.multi_label, labels=lb)
    #     out = [out_tensor.detach() for out_tensor in out]
    #     target_out_np = output_to_target_ssod(out)
    #     target_out_targets = torch.tensor(target_out_np)
    #     target_shape = target_out_targets.shape
    #     total_t1 = time_sync()
    #     # print('M:', M)
    #     if(target_shape[0] > 0 and target_shape[1] > 6):
    #         for i, img in enumerate(target_imgs):

    #             image_targets = target_out_targets[target_out_targets[:, 0] == i]
    #             if isinstance(image_targets, torch.Tensor):
    #                 image_targets = image_targets.cpu().numpy()
    #             image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
    #             M_select = M_s[M_s[:, 0] == i, :]  # image targets
    #             M = M_select[0][1:10].reshape([3,3]).cpu().numpy()
    #             s = float(M_select[0][10])
    #             ud = int(M_select[0][11])
    #             lr = int(M_select[0][12])
    #             img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M, s)

    #             if 1:
    #                 image_targets = np.array(image_targets_random)
    #             else:
    #                 image_targets = np.array(image_targets[:, 1:])
    #             if image_targets.shape[0] != 0:
    #                 image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
    #                 image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])  # convert xyxy to xywh
    #                 image_targets[:, [3, 5]] /= height # normalized height 0-1
    #                 image_targets[:, [2, 4]] /= width # normalized width 0-1
    #                 if ud == 1:
    #                     image_targets[:, 3] = 1 - image_targets[:, 3]
    #                 if lr == 1:
    #                     image_targets[:, 2] = 1 - image_targets[:, 2]
    #                 target_out_targets_perspective.extend(image_targets.tolist())
    #         target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))
    #     # if self.RANK in [-1, 0]:
    #         # print('total time cost:', time_sync() - total_t1)

    #     if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
    #         invalid_target_shape = False
    #     return target_out_targets_perspective, target_imgs, invalid_target_shape

    def create_pseudo_label_on_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        """
        基于真实标签生成伪标签的调试模式
        功能：直接使用真实标签作为伪标签，用于算法调试和效果验证

        参数说明：
        out: 模型输出(本函数未实际使用，保留接口兼容)
        target_imgs: 增强后的目标图像
        M_s: 增强变换矩阵集合(未实际使用)
        target_imgs_ori: 原始目标图像(未实际使用)
        gt: 真实标注数据
        RANK: 进程标识符(用于控制调试输出)

        返回：
        target_out_targets_perspective: 带变换的真实标签(直接返回gt)
        target_imgs: 原始输入图像
        invalid_target_shape: 标签有效性标识

        重点细节：
        - 完全依赖真实标签生成伪标签，跳过模型预测处理
        - 包含调试可视化功能，绘制带真实标注的图像
        - 通过RANK参数控制只在主进程保存可视化结果
        - 保持与其他伪标签生成方法相同的输入输出接口
        """
        invalid_target_shape = True
        target_out_targets_perspective = gt  # 直接使用真实标签作为伪标签
        target_shape = target_out_targets_perspective.shape

        # 有效性验证与调试可视化
        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0:
            invalid_target_shape = False
            if self.debug:  # 调试模式专属逻辑
                if RANK in [-1, 0]:  # 主进程执行可视化
                    draw_image = plot_images(copy.deepcopy(target_imgs),
                                             target_out_targets_perspective,
                                             None,
                                             '/mnt/bowen/EfficientTeacher/unbias_teacher_pseudo_label.jpg')

        return target_out_targets_perspective, target_imgs, invalid_target_shape

    def create_pseudo_label_online_with_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        """
        在线生成几何变换增强后的伪标签
        主要流程：
        1. 执行改进版非极大值抑制筛选预测框
        2. 转换预测结果为标准格式
        3. 逐图像进行空间变换校正
        4. 坐标归一化与镜像补偿处理

        参数说明：
        out: 模型原始输出张量
        target_imgs: 数据增强后的图像批次
        M_s: 增强参数矩阵集合[包含变换矩阵/缩放因子/翻转标志]
        target_imgs_ori: 原始未增强图像(用于坐标变换基准)
        gt: 真实标签(仅调试用)
        RANK: 进程标识(-2表示默认)

        重点细节：
        - 使用SSOD专用的NMS实现(non_max_suppression_ssod)
        - 输出格式转换包含置信度分数分解(obj_conf/cls_conf)
        - 变换矩阵M包含3x3空间变换参数
        - 镜像补偿处理ud(上下)/lr(左右)标志位
        - 调试可视化生成对比图：伪标签图与真实标签图
        - 坐标系统始终维持归一化[0,1]范围
        """
        n_img, _, height, width = target_imgs.shape
        lb = []

        # 改进版非极大值抑制(SSOD专用配置)
        out = non_max_suppression_ssod(out, conf_thres=self.nms_conf_thres,
                                       iou_thres=self.nms_iou_thres,
                                       num_points=self.num_points,
                                       multi_label=self.multi_label,
                                       labels=lb)
        out = [out_tensor.detach() for out_tensor in out]  # 解除梯度依赖

        # 格式标准化处理
        target_out_np = output_to_target_ssod(out)  # 包含obj_conf/cls_conf分解
        target_out_targets = torch.tensor(target_out_np)
        target_shape = target_out_targets.shape
        target_out_targets_perspective = []
        invalid_target_shape = True

        # 空间变换主流程
        if target_shape[0] > 0 and target_shape[1] > 6:
            for i, img in enumerate(target_imgs_ori):
                # 提取当前图像对应的预测结果
                image_targets = target_out_np[target_out_np[:, 0] == i]
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()

                # 坐标格式转换(xywh->xyxy)
                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])

                # 解析变换参数(3x3矩阵+缩放+翻转标志)
                M_select = M_s[M_s[:, 0] == i, :]
                M = M_select[0][1:10].reshape([3, 3]).cpu().numpy()
                s = float(M_select[0][10])
                ud = int(M_select[0][11])
                lr = int(M_select[0][12])

                # 执行几何变换(保持图像与标签同步)
                img, image_targets_random = online_label_transform(
                    img, copy.deepcopy(image_targets[:, 1:]), M, s)

                # 后处理流程
                if image_targets.shape[0] != 0:
                    # 重组批次索引并转换坐标格式
                    image_targets = np.concatenate(
                        (np.ones([image_targets.shape[0], 1]) * i,
                         image_targets), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])

                    # 归一化处理
                    image_targets[:, [3, 5]] /= height  # 高度维度(y/height)
                    image_targets[:, [2, 4]] /= width  # 宽度维度(x/width)

                    # 镜像翻转补偿
                    if ud == 1: image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1: image_targets[:, 2] = 1 - image_targets[:, 2]

                    target_out_targets_perspective.extend(image_targets.tolist())

            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))

        # 调试可视化模块
        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0:
            invalid_target_shape = False
            if self.debug and RANK in [-1, 0]:
                # 生成伪标签可视化对比图
                plot_images_ssod(copy.deepcopy(target_imgs),
                                 target_out_targets_perspective,
                                 fname='/mnt/bowen/EfficientTeacher/effcient_teacher_pseudo_label.jpg',
                                 names=self.names)
                # 生成真实标签基准图
                plot_images(copy.deepcopy(target_imgs), gt,
                            fname='/mnt/bowen/EfficientTeacher/effcient_teacher_gt.jpg',
                            names=self.names)

        return target_out_targets_perspective, invalid_target_shape


    def create_pseudo_label_online_with_extra_teachers(self, out, extra_teacher_outs, target_imgs, M_s,
                                                   extra_teacher_class_idxs, RANK):
        """
        多教师模型协同的在线伪标签生成
        功能：集成多个教师模型的预测结果，通过交叉验证提升伪标签质量

        参数说明：
        out: 主模型原始输出
        extra_teacher_outs: 多个辅助教师模型的输出列表
        target_imgs: 增强后的目标图像批次
        M_s: 增强变换参数矩阵集合
        extra_teacher_class_idxs: 各教师模型的类别映射表
        RANK: 进程标识符

        核心流程：
        1. 主模型预测结果初步筛选
        2. 多教师模型预测融合与类别对齐
        3. 跨模型NMS融合去重
        4. 空间变换与坐标校准

        重点细节：
        - 多阶段NMS处理：主模型与各教师模型独立执行NMS
        - 类别索引转换：通过extra_teacher_class_idxs进行跨模型类别对齐
        - 预测框融合策略：合并主模型与所有教师模型的预测结果
        - 跨模型NMS去重：使用torchvision.ops.nms进行最终筛选
        - 空间变换参数复用：使用与单教师版本相同的变换矩阵处理
        - 调试可视化路径与单教师版本保持统一
        """
        n_img, _, height, width = target_imgs.shape
        lb = []
        target_out_targets_perspective = []
        invalid_target_shape = True

        # 主模型预测初步筛选
        out = non_max_suppression(out, conf_thres=self.nms_conf_thres,
                                  iou_thres=self.nms_iou_thres,
                                  labels=lb,
                                  multi_label=self.multi_label)
        out = [out_tensor.detach() for out_tensor in out]

        # 多教师模型预测融合
        for teacher_idx, teacher_out in enumerate(extra_teacher_outs):
            teacher_pseudo_out = non_max_suppression(teacher_out,
                                                     conf_thres=self.nms_conf_thres,
                                                     iou_thres=self.nms_iou_thres,
                                                     labels=lb,
                                                     multi_label=self.multi_label)

            # 批次维度处理
            for i, o in enumerate(out):
                pseudo_out_one_img = teacher_pseudo_out[i]
                if pseudo_out_one_img.shape[0] > 0:
                    # 类别索引转换(根据教师模型映射表)
                    for each in pseudo_out_one_img:
                        origin_class_idx = int(each[5].cpu().item())
                        mapped_idx = extra_teacher_class_idxs[teacher_idx].get(origin_class_idx)
                        if mapped_idx is not None:
                            each[5] = float(mapped_idx)

                # 预测框融合与跨模型NMS
                x = torch.cat([o, pseudo_out_one_img])
                c = x[:, 5:6] * 0  # 类别偏移量(保持原始坐标)
                boxes, scores = x[:, :4] + c, x[:, 4]  # 合并坐标与置信度
                index = torchvision.ops.nms(boxes, scores, self.nms_iou_thres)
                out[i] = x[index]  # 更新当前批次预测结果

        # 后续处理流程(与单教师版本相同)
        target_out_np = output_to_target_ssod(out)
        target_out_targets = torch.tensor(target_out_np)
        target_shape = target_out_targets.shape

        if target_shape[0] > 0 and target_shape[1] > 6:
            for i, img in enumerate(target_imgs):
                # 坐标变换流程(与基础版相同)
                image_targets = target_out_targets[target_out_targets[:, 0] == i]
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()

                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
                M_select = M_s[M_s[:, 0] == i, :]
                M = M_select[0][1:10].reshape([3, 3]).cpu().numpy()
                s = float(M_select[0][10])
                ud = int(M_select[0][11])
                lr = int(M_select[0][12])

                img, image_targets_random = online_label_transform(img,
                                                                   copy.deepcopy(image_targets[:, 1:]),
                                                                   M, s)
                # 坐标归一化与镜像补偿
                if image_targets.shape[0] != 0:
                    image_targets = np.concatenate((np.ones([image_targets.shape[0], 1]) * i, image_targets), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])
                    image_targets[:, [3, 5]] /= height  # y轴归一化
                    image_targets[:, [2, 4]] /= width  # x轴归一化
                    # 镜像翻转补偿
                    if ud == 1: image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1: image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())

            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))

        # 调试可视化(复用单教师版本路径)
        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0:
            invalid_target_shape = False
            if self.debug and RANK in [-1, 0]:
                plot_images_ssod(copy.deepcopy(target_imgs),
                                 target_out_targets_perspective,
                                 fname='/mnt/bowen/EfficientTeacher/effcient_teacher_pseudo_label.jpg',
                                 names=self.names)

        return target_out_targets_perspective, target_imgs, invalid_target_shape


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    """
    数据增强候选框有效性筛选器
    功能：验证增强后的检测框是否符合几何约束条件

    参数说明：
    box1: 增强前原始框坐标(4,n)格式[x1,y1,x2,y2]
    box2: 增强后目标框坐标(4,n)
    wh_thr: 宽高最小像素阈值(默认2px)
    ar_thr: 宽高比最大阈值(防止极端比例)
    area_thr: 面积保留最小比率(过滤严重缩放的框)
    eps: 极小值防止除零错误

    筛选条件(同步满足)：
    1. 增强后宽高>wh_thr
    2. 宽高比<ar_thr
    3. 面积变化率>area_thr
    4. 排除无效/过度形变的框

    返回：符合要求的候选框布尔掩码(n,)
    """
    # 原始框宽高计算
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    # 增强后宽高计算
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    # 宽高比约束(取w/h与h/w的最大值)
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))

    # 四重条件联合判断
    return (w2 > wh_thr) & (h2 > wh_thr) & \
        (w2 * h2 / (w1 * h1 + eps) > area_thr) & \
        (ar < ar_thr)


def online_label_transform(img, targets, M, s, segments=(), border=(0, 0), perspective=0.0):
    """
    在线数据增强与标签同步变换
    功能：对图像进行空间变换时，同步调整目标框或分割标签的坐标

    参数：
    img: 输入图像张量(CHW格式)
    targets: 目标标注矩阵[样本数, (类别+坐标)]
    M: 3x3空间变换矩阵
    s: 缩放因子(用于候选框过滤)
    segments: 分割多边形列表(暂未使用)
    border: 图像填充尺寸(上下,左右)
    perspective: 透视变换强度

    处理流程：
    1. 计算变换后图像尺寸(含填充)
    2. 根据标注类型(框/分割)执行坐标变换
    3. 应用变换矩阵并计算新坐标
    4. 过滤无效目标(尺寸过小/形变过大)

    关键细节：
    - 支持仿射/透视两种变换模式
    - 边界框处理采用四角点扩展法保证精度
    - 分割标签处理包含重采样提升精度
    - 使用齐次坐标进行矩阵运算
    - 通过box_candidates实现自适应过滤
    - 返回与增强后图像匹配的新坐标
    """
    height = img.shape[1] + border[0] * 2  # 计算含填充的总高度(HWC转CHW后的维度)
    width = img.shape[2] + border[1] * 2  # 计算含填充的总宽度

    # 坐标变换主逻辑
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))

        # 分割标签处理分支
        if use_segments:
            segments = resample_segments(segments)  # 上采样增加分割点密度
            for i, segment in enumerate(segments):
                # 构建齐次坐标矩阵(N,3)
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                # 应用变换矩阵(M^T)
                xy = xy @ M.T
                # 透视变换归一化处理
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                # 生成新边界框
                new[i] = segment2box(xy, width, height)

        # 边界框处理分支
        else:
            # 构建四角点坐标矩阵(每个框扩展为4个角点)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # 按x1y1,x2y2,x1y2,x2y1顺序重组

            # 应用空间变换矩阵
            xy = xy @ M.T
            # 透视处理并恢复形状(n,8)
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

            # 计算最小包围框
            x = xy[:, [0, 2, 4, 6]]  # 所有x坐标
            y = xy[:, [1, 3, 5, 7]]  # 所有y坐标
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # 边界裁剪(防止坐标越界)
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)  # x方向限制
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)  # y方向限制

        # 候选框过滤(基于面积变化率等指标)
        i = box_candidates(box1=targets[:, 1:5].T * s,
                           box2=new.T,
                           area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]  # 保留有效目标
        targets[:, 1:5] = new[i]  # 更新坐标数据

    return img, targets

def check_pseudo_label_with_gt(detections, labels, iouv=torch.tensor([0.5]), ignore_thres_low=None, ignore_thres_high=None, batch_size=1):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    tp_num = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    fp_cls_num = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    fp_loc_num = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    gt = labels[:, 2:6]
    # print('labels shape:', labels.shape)
    if ignore_thres_low is not None:
        # print('thres_high:', ignore_thres_high)
        # print('thres_low:', ignore_thres_low)
        detections, uc_pseudo = select_targets(detections, ignore_thres_low, ignore_thres_high)
        # pseudo = detections[detections[:, 6] > ignore_thres]
        # print('pseudo:', detections)
        # print('uc_pseudo:', uc_pseudo)
        # detections = torch.cat((detections, uc_pseudo))
        detections = uc_pseudo
        pseudo = detections[:, 2:6]
        pseudo_label_num = pseudo.shape[0] / batch_size
        # pseudo_label_num = detections.shape[0]
        gt_label_num = gt.shape[0]/batch_size
    else:
        pseudo = detections[:, 2:6]
        pseudo_label_num = detections.shape[0]
        pseudo_label_num /= batch_size
        gt_label_num = gt.shape[0]/batch_size

    gt *= torch.tensor([640, 640] * 2)
    pseudo *= torch.tensor([640, 640] * 2)
    gt = xywh2xyxy(gt) + labels[:, 0:1] * torch.tensor([640, 640] * 2)
    pseudo = xywh2xyxy(pseudo) + detections[:, 0][:, None] * torch.tensor([640, 640] * 2)
    # print(gt)
    # print(pseudo)
    # print(labels[:, 0:1][:,None].shape)
    iou = box_iou(gt, pseudo)
    # print('iou:', iou)
    # print('iou shape', iou.shape)
    correct_class = labels[:, 1:2] == detections[:, 1]
    correct_image = labels[:, 0:1] == detections[:, 0]
    # print('correct_class:', correct_class)

    for i in range(len(iouv)):
        # print('correct_image:', correct_image)
        # print('correct_class:', correct_class)
        # print('iou:', (iou < iouv[i]) & (iou > torch.tensor(0.1)))
        tp_x = torch.where((iou >= iouv[i]) & correct_class & correct_image)  # IoU > threshold and classes match
        fp_cls_x = torch.where((iou >= iouv[i]) & ~correct_class & correct_image)  # IoU > threshold and classes match
        # fp_loc_x = torch.where((iou < iouv[i]) & (iou > torch.tensor(0.1)) & correct_class & correct_image)  # IoU > threshold and classes match
        fp_loc_x = torch.where((iou < iouv[i]) & (iou > torch.tensor(0.01)) & correct_image)  # IoU > threshold and classes match
        # print('fp_loc_x:', fp_loc_x.shape)
        # print('fp_cls_x:', fp_cls_x.shape)
        # print(iou.shape, correct_image.shape, correct_class.shape)
        # fp_both_x = torch.where((iou < iouv[i]) & correct_image & ~correct_class)  # IoU > threshold and classes match
        # print('x:', x)
        # print(torch.where(iou >= iouv[i]))
        if tp_x[0].shape[0]:
            matches = torch.cat((torch.stack(tp_x, 1), iou[tp_x[0], tp_x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if tp_x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            tp_num[matches[:, 1].astype(int), i] = True
        if fp_cls_x[0].shape[0]:
            matches = torch.cat((torch.stack(fp_cls_x, 1), iou[fp_cls_x[0], fp_cls_x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if fp_cls_x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            fp_cls_num[matches[:, 1].astype(int), i] = True
        if fp_loc_x[0].shape[0]:
            matches = torch.cat((torch.stack(fp_loc_x, 1), iou[fp_loc_x[0], fp_loc_x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if fp_loc_x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            fp_loc_num[matches[:, 1].astype(int), i] = True
    if detections.shape[0] == 0:
        tp_rate = 0
        fp_cls_rate = 0
        fp_loc_rate = 0
    else:
        tp_rate = np.sum(tp_num, 0) * 1.0/ detections.shape[0]
        fp_cls_rate = np.sum(fp_cls_num, 0) * 1.0/ detections.shape[0]
        fp_loc_rate = np.sum(fp_loc_num, 0) * 1.0/ detections.shape[0]
    # print('tp_rate:', tp_rate, tp_num)
    # print('fp_cls_rate:', fp_cls_rate, np.sum(fp_cls_num, 0))
    # print('fp_loc_rate:', fp_loc_rate, fp_loc_num, detections.shape[0])
    # iou_recall_rate = np.sum(correct, 0) * 1.0/ labels.shape[0]
    # print('correct:', np.sum(correct, 0))
    # if ignore_thres_low is not None:
    # hit_rate = detections.shape[0] * 1.0 / labels.shape[0]
    # print('correct shape:', correct.shape)
    # print(np.sum(correct, 1))
    # print(detections.shape[0])
    return tp_rate, fp_cls_rate, fp_loc_rate, pseudo_label_num, gt_label_num

def check_pseudo_label(detections, ignore_thres_low=None, ignore_thres_high=None, batch_size=1):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # print('ignore_thres_high:', ignore_thres_high, ' ignore_thres_low:', ignore_thres_low)
    reliable_pseudo, uc_pseudo = select_targets(detections, ignore_thres_low, ignore_thres_high)
    reliable_num = reliable_pseudo.shape[0]/batch_size
    uncertain_num = uc_pseudo.shape[0]/batch_size
    denorm = reliable_num + uncertain_num
    if denorm == 0:
        precision_rate = 0
    else:
        precision_rate = reliable_num/ denorm
    if detections.shape[0] == 0:
        recall_rate = 0
    else:
        recall_rate = (reliable_num + uncertain_num) * batch_size / detections.shape[0]
    return precision_rate, recall_rate, reliable_num + uncertain_num, reliable_num

def select_targets(targets, ignore_thres_low, ignore_thres_high):
    device = targets.device
    certain_targets = []
    uncertain_targets = []
    for t in targets:
        # 伪标签得分大于相应类别的阈值,标记为正样本
        if t[6] >= ignore_thres_high[int(t[1])]:
            certain_targets.append(np.array(t.cpu()))
        # 伪标签在0.1到阈值去之前的，标记为忽略样本
        elif t[6] >= ignore_thres_low[int(t[1])]:
            uncertain_targets.append(np.array(t.cpu()))

    certain_targets = np.array(certain_targets).astype(np.float32)
    certain_targets = torch.from_numpy(certain_targets).contiguous()
    certain_targets = certain_targets.to(device)
    if certain_targets.shape[0] == 0:
        certain_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

    uncertain_targets = np.array(uncertain_targets).astype(np.float32)
    uncertain_targets = torch.from_numpy(uncertain_targets).contiguous()
    uncertain_targets = uncertain_targets.to(device)
    if uncertain_targets.shape[0] == 0:
        uncertain_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)
    return certain_targets, uncertain_targets