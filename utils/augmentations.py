import logging
import math
import random

import cv2
import numpy as np

from utils.general import colorstr, check_version
from utils.metrics import bbox_ioa
from utils.self_supervised_utils import box_candidates


class Albumentations:
    """
    YOLOv5的Albumentations增强类（可选，仅在安装了相应包时使用）

    功能:
        初始化图像增强流水线并应用随机变换
        支持边界框变换并保持标注一致性

    使用方法:
        transforms = Albumentations()
        image, labels = transforms(image, labels)
    """
    def __init__(self):
        """初始化Albumentations图像增强流水线"""
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3')  # 版本检查，确保兼容性

            # 定义增强变换组合
            self.transform = A.Compose([
                A.Blur(p=0.01),                  # 模糊处理，概率1%
                A.MedianBlur(p=0.01),            # 中值模糊，概率1%
                A.ToGray(p=0.01),                # 转换为灰度图，概率1%
                A.CLAHE(p=0.01),                 # 对比度受限的自适应直方图均衡化，概率1%
                A.RandomBrightnessContrast(p=0.0), # 随机亮度对比度调整，概率0%（不启用）
                A.RandomGamma(p=0.0),            # 随机伽马校正，概率0%（不启用）
                A.ImageCompression(quality_lower=75, p=0.0)], # 图像压缩，质量下限75，概率0%（不启用）
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) # YOLO格式边界框参数

            # 打印已启用的增强变换
            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # 未安装包，跳过
            pass
        except Exception as e:
            logging.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        """
        应用增强变换

        参数:
            im: 输入图像
            labels: 标签数组，格式为[class, x_center, y_center, width, height]
            p: 应用变换的概率

        返回:
            im: 变换后的图像
            labels: 变换后的标签
        """
        if self.transform and random.random() < p:
            # 执行增强变换
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # 应用变换
            # 提取变换后的图像和标签
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    HSV颜色空间数据增强

    参数:
        im: 输入图像(BGR格式 numpy数组)
        hgain: 色调增益系数(0表示不增强)
        sgain: 饱和度增益系数(0表示不增强)
        vgain: 明度增益系数(0表示不增强)

    处理逻辑:
        1. 生成随机增益系数，范围为[1-gain, 1+gain]
        2. 将图像从BGR颜色空间转换为HSV
        3. 分别对色调、饱和度、明度通道应用查找表变换:
           - 色调: 循环取模(0-180范围)
           - 饱和度/明度: 裁剪到0-255范围
        4. 转换回BGR颜色空间并直接修改输入图像
    """
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # 生成随机增益系数（范围[1-gain, 1+gain]）
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))  # 转换至HSV颜色空间并拆分通道
        dtype = im.dtype  # 保存原始数据类型（用于后续转换）

        x = np.arange(0, 256, dtype=r.dtype)  # 生成0-255的索引数组
        lut_hue = ((x * r[0]) % 180).astype(dtype)  # 色调通道循环取模（HSV色调范围0-180）
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 饱和度通道裁剪到0-255范围
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 明度通道裁剪到0-255范围

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))  # 应用查找表变换
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # 转换回BGR颜色空间并直接修改原图像


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    图像缩放与填充（Letterbox变换），满足步长倍数约束

    参数:
        im: 输入图像(numpy数组，shape为[H, W, C])
        new_shape: 目标尺寸，可为整数或元组(int, int)
        color: 填充边框颜色(RGB格式)
        auto: 是否自动计算最小填充（使填充后尺寸为stride倍数）
        scaleFill: 是否拉伸图像至完全覆盖目标尺寸（可能变形）
        scaleup: 是否允许放大图像（为False时仅缩小）
        stride: 步长约束（用于auto模式下的尺寸对齐）

    返回:
        im: 处理后的图像
        ratio: 缩放比例（宽, 高）
        pad: 边框填充量（宽方向, 高方向）

    处理逻辑:
        1. 计算缩放比例，优先保持原始宽高比
        2. 根据scaleup参数限制缩放比例（不放大或自由缩放）
        3. 计算原始缩放后的尺寸(new_unpad)和边框填充量(dw, dh)
        4. auto模式下将填充量对齐到stride倍数
        5. scaleFill模式下禁用边框填充，直接拉伸图像
        6. 执行图像缩放和边框填充
    """
    shape = im.shape[:2]  # 当前图像尺寸[高, 宽]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例（取宽高比最小值，保持原始比例）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 禁止放大图像（仅缩小或保持原尺寸）
        r = min(r, 1.0)

    ratio = r, r  # 宽高缩放比例（等比例缩放）
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 原始缩放后的尺寸（未填充）
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高方向需填充的总像素数

    if auto:  # 自动模式：使填充后的尺寸为stride整数倍
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 取模运算实现最小非负填充
    elif scaleFill:  # 拉伸模式：直接填满目标尺寸（可能变形）
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 非等比例缩放比例

    dw /= 2  # 左右边框填充量（总填充量均分两侧）
    dh /= 2  # 上下边框填充量（总填充量均分两侧）

    if shape[::-1] != new_unpad:  # 需要执行缩放（当原始尺寸与缩放后尺寸不一致时）
        # 使用双线性插值调整图像大小（保持平滑）
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        # im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_NEAREST)  # 可切换为最近邻插值

    # 计算实际填充像素数（四舍五入处理浮点误差）
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # 添加边框（BORDER_CONSTANT表示纯色填充）
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def random_perspective_keypoints(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10,
                                 perspective=0.0, num_points=0, border=(0, 0)):
    """
    🔥 带关键点处理的随机几何变换增强

    参数:
        img: 输入图像(HWC格式)
        targets: 目标数据[类别,x1,y1,x2,y2,关键点坐标...]
        segments: 可选分割多边形
        degrees: 随机旋转角度范围
        translate: 平移比例范围
        perspective: 透视变换强度(0-1)
        num_points: 关键点数量(0表示无)
        border: 图像边界的填充像素

    核心处理:
        1. 构造组合变换矩阵(M = T·S·R·P·C)
        2. 应用仿射/透视变换到图像
        3. 动态计算目标框与关键点新坐标
        4. 关键点有效性校验与数据修复

    关键细节:
        📌 矩阵变换顺序: 平移->剪切->旋转->透视->中心化
        📌 分割/框两种处理模式：
           - 使用4点时为分割模式(每个框4角点)
           - 使用num_points时处理关键点坐标
        📌 关键点碰边检测(check_board=True时激活)：
           - 出界坐标标记为-1
           - 全无效关键点恢复原始数据
        📌 使用0.1阈值过滤无效box：
           - 保留面积变化合理的目标
           - 防止过小的错误检测框

    数据格式:
        targets结构:
        [class, x_min, y_min, x_max, y_max, x1,y1,...xn,yn]
        关键点顺序需与原始标注严格对应
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy, x0, y0, x1, y2, x2, y2, x3, y3]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped
    check_board = False
    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if num_points == 0:  # warp segments
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        else:  # warp boxes
            xy = np.ones((n * (4 + num_points), 3))
            # xy = np.ones((n * 8, 3))
            index = [1, 2, 3, 4, 1, 4, 3, 2]
            index_landmark = list(range(5, 5 + num_points * 2))
            index.extend(index_landmark)
            xy[:, :2] = targets[:, index].reshape(n * (4 + num_points), 2)
            # xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11, 12]].reshape(n * 8, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 2 * (4 + num_points))  # rescale
            else:  # affine
                # xy = xy[:, :2].reshape(n, 2*8)
                xy = xy[:, :2].reshape(n, 2 * (4 + num_points))
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            landmarks_index = list(range(8, 8 + num_points * 2))
            # landmarks = xy[:, [8, 9, 10, 11, 12, 13, 14, 15]]
            landmarks = xy[:, landmarks_index]
            # mask = np.array(targets[:, 5:] > 0, dtype=np.int32)

            # 检查进行数据增强后的关键点标注是否有碰边行为，如果有，该标注无效
            if check_board:
                mask = np.array(landmarks > 0, dtype=np.int32)
                landmarks = landmarks * mask
                landmarks = landmarks + mask - 1
                # landmarks[lmk_non_valid_index] = targets[lmk_non_valid_index, 5:]

                # landmarks = np.where(landmarks < 0, -1, landmarks)
                landmarks[:, [0, 2, 4, 6]] = np.where(landmarks[:, [0, 2, 4, 6]] > width, -1,
                                                      landmarks[:, [0, 2, 4, 6]])
                landmarks[:, [1, 3, 5, 7]] = np.where(landmarks[:, [1, 3, 5, 7]] > height, -1,
                                                      landmarks[:, [1, 3, 5, 7]])
                # landmarks_tmp = landmarks.copy()
                for ind, landmark in enumerate(landmarks):
                    if -1 in landmark:
                        landmarks[ind] = np.ones((1, num_points * 2)) * -1

            # landmarks[:, 0] = np.where(landmarks[:, 1] == -1, -1, landmarks[:, 0])
            # landmarks[:, 1] = np.where(landmarks[:, 0] == -1, -1, landmarks[:, 1])

            # landmarks[:, 2] = np.where(landmarks[:, 3] == -1, -1, landmarks[:, 2])
            # landmarks[:, 3] = np.where(landmarks[:, 2] == -1, -1, landmarks[:, 3])

            # landmarks[:, 4] = np.where(landmarks[:, 5] == -1, -1, landmarks[:, 4])
            # landmarks[:, 5] = np.where(landmarks[:, 4] == -1, -1, landmarks[:, 5])

            # landmarks[:, 6] = np.where(landmarks[:, 7] == -1, -1, landmarks[:, 6])
            # landmarks[:, 7] = np.where(landmarks[:, 6] == -1, -1, landmarks[:, 7])
            ori_targets = targets.copy()

            # 检查8个关键点是否都为-1, 表述在进行数据增强之前的标注就已经无效
            targets[:, 5:] = landmarks
            non_valid_index = ((ori_targets[:, 5:] == -1).sum(1) == num_points * 2)  # [false true false] example
            targets[non_valid_index, 5:] = ori_targets[non_valid_index, 5:]

            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        # print('before:', targets)
        targets[:, 1:5] = new[i]
        # print('after:', targets)

    return img, targets

def copy_paste(im, labels, segments, p=0.5):
    """🎭 Copy-Paste数据增强(水平翻转版)，提升小目标检测能力

    参数:
        im: 原始图像(numpy数组)
        labels: 目标标签数组[class,x1,y1,x2,y2]
        segments: 分割多边形坐标列表
        p: 执行概率(默认0.5)

    核心逻辑:
        1. 按概率随机选取部分实例
        2. 生成水平镜像副本
        3. 计算IOA避免目标过度重叠
        4. 通过位运算合成新图像

    关键细节:
        🔄 镜像翻转处理：
           - 计算水平翻转后的x坐标：w - x
           - 同步调整边界框坐标顺序
        🎯 遮挡控制：
           - 30%的IOA阈值过滤重叠目标
           - 保持原始标签数据完整性
        🖌️ 图像合成：
           - 使用cv2轮廓绘制生成掩膜
           - 位操作实现像素级融合

    注意:
        - 原图会被直接修改(in-place操作)
        - segments会追加新的多边形坐标
        - 要求输入labels为numpy数组格式
    """
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def mixup(im, labels, im2, labels2):
    """🎨 MixUp数据增强，融合两张图像及标签

    参数:
        im/labels: 主图像及标签
        im2/labels2: 混合图像及标签
    核心:
        - 使用β分布(α=β=32)生成混合比r
        - 线性叠加图像: im*r + im2*(1-r)
        - 合并标签数据
    特点:
        - 增强模型鲁棒性和泛化能力
        - 保留两张图像的全部标注信息
        - 混合比例偏向中间值(β分布特性)
    """
    r = np.random.beta(32.0, 32.0)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels

