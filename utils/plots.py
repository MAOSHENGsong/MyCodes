import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.general import user_config_dir, is_ascii, is_chinese, xywh2xyxy

# Settings
CONFIG_DIR = user_config_dir()  # Ultralytics设置目录
RANK = int(os.getenv('RANK', -1))  # 分布式训练中的进程排名，-1表示非分布式
matplotlib.rc('font', **{'size': 11})  # 设置matplotlib字体大小为11
matplotlib.use('Agg')  # 使用非交互式后端，仅写入文件


class Colors:
    """Ultralytics颜色调色板，支持20种高对比度颜色循环使用
    功能特性：
    - 预定义调色板：基于品牌风格的20种HEX颜色
    - BGR/RGB格式转换：适配OpenCV和PIL的图像处理需求
    - 索引循环机制：超出颜色数量时自动循环取色

    颜色来源：https://ultralytics.com/
    """

    def __init__(self):
        # 原始HEX颜色取自matplotlib的TABLEAU_COLORS，调整为品牌专属配色
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]  # 转换为RGB元组列表
        self.n = len(self.palette)  # 调色板颜色总数

    def __call__(self, i, bgr=False):
        """通过索引获取颜色，自动循环选择
        参数说明：
        i: 颜色索引(超出数量时自动取模)
        bgr: 是否返回BGR格式(默认RGB)
        返回：(R,G,B)或(B,G,R)格式的颜色元组
        """
        c = self.palette[int(i) % self.n]  # 循环取色机制
        return (c[2], c[1], c[0]) if bgr else c  # OpenCV需BGR格式时转换通道顺序

    @staticmethod
    def hex2rgb(h):
        """将HEX颜色码转换为RGB元组(PIL兼容格式)
        参数示例：#FF3838 → (255,56,56)
        实现原理：按两位一组解析十六进制字符串
        """
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()


def check_font(font='Arial.ttf', size=10):
    """字体加载器，自动下载缺失字体文件到配置目录

    核心功能：
    - 多路径查找：优先本地路径，其次配置目录(CONFIG_DIR)
    - 自动下载机制：从Ultralytics CDN获取缺失字体
    - 异常处理：下载失败时回退到系统默认字体名称

    实现要点：
    - 使用torch.hub下载保障断点续传
    - 下载后持久化存储，避免重复下载
    - 适配可视化库的字体加载接口(PIL.ImageFont)
    """
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)  # 优先本地路径
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception as e:  # 下载缺失字体
        url = "https://ultralytics.com/assets/" + font.name
        print(f'Downloading {url} to {font}...')
        torch.hub.download_url_to_file(url, str(font), progress=False)  # 无进度条下载
        return ImageFont.truetype(str(font), size)


class Annotator:
    """YOLOv5标注工具，支持训练/验证马赛克图、检测结果可视化及透视变换辅助标注
    核心功能：
    - 双渲染引擎：自动选择PIL(适合文本)或OpenCV(适合几何图形)
    - 多语言支持：自动检测中文字符切换字体
    - 自适应样式：线宽/字体大小随图像尺寸动态调整
    """
    if RANK in (-1, 0):
        check_font()  # 主进程预下载字体文件

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        """初始化标注器
        参数说明：
        im: 输入图像(PIL.Image或numpy数组)
        line_width: 标注线宽，None时自动计算
        font_size: 字体大小，None时基于图像尺寸计算
        font: 字体文件路径，中文自动切换'Arial.Unicode.ttf'
        pil: 强制使用PIL渲染
        example: 示例文本用于检测字符集
        """
        assert im.data.contiguous, '图像内存不连续，请使用np.ascontiguousarray(im)处理输入'

        # 渲染引擎选择逻辑：含中文/非ASCII字符时强制PIL
        self.pil = pil or not is_ascii(example) or is_chinese(example)

        if self.pil:  # PIL模式(文本渲染优化)
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            # 中文字体特殊处理
            self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                   size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # OpenCV模式(几何变换优化)
            self.im = im
            h, w, _ = self.im.shape
            # 透视变换控制点(用于3D标注效果)
            self.pts_1 = np.float32([[0.0 * w, 0], [0.5 * w, 0], [0.6 * w, 1.0 * h], [0, 1.0 * h]])
            self.pts_2 = np.float32([[0, 0], [0.33 * w, 0], [0.33 * w, 1.0 * h], [0, 1.0 * h]])
            self.M = cv2.getPerspectiveTransform(self.pts_1, self.pts_2)  # 计算变换矩阵

            # 创建扩展画布用于辅助标注
            self.frame_draw = np.zeros((h, int(w + w / 2), 3), dtype=np.uint8)
            self.frame_draw[:h, :w, :] = self.im

            # 生成网格辅助线(100x100像素间隔)
            mesh_width = 100
            mesh_height = 100
            for i in range(int(w / mesh_width)):
                cv2.line(self.frame_draw[:, w:, :], (0, mesh_height * i), (w, mesh_height * i), (255, 255, 255), 1)
            for i in range(int(h / mesh_height)):
                cv2.line(self.frame_draw[:, w:, :], (mesh_width * i, 0), (mesh_width * i, h), (255, 255, 255), 1)

            self.mask = np.zeros(self.im.shape, dtype='uint8')  # 创建标注掩模

        # 自适应线宽计算：基于图像对角线长度的0.3%
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """在原始图像上绘制边界框及标签，支持PIL/OpenCV双渲染模式

        参数说明：
        box: 边界框坐标(xyxy格式)
        label: 标签文本，空字符串时不显示
        color: 框线颜色(RGB)
        txt_color: 文本颜色(RGB)

        实现细节：
        - PIL模式：绘制矩形框后计算文本区域，自动判断标签显示位置(框内/外)
        - OpenCV模式：使用抗锯齿线型，动态调整文本背景框大小
        - 标签位置优化：优先显示在框外顶部，空间不足时显示在框内底部
        """
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # 绘制矩形框
            if label:
                w, h = self.font.getsize(label)  # 获取文本尺寸
                outside = box[1] - h >= 0  # 判断框上方空间是否足够
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)  # 文本背景框
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # OpenCV模式
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)  # 抗锯齿矩形
            if label:
                tf = max(self.lw - 1, 1)  # 字体厚度
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # 获取文本尺寸
                outside = p1[1] - h - 3 >= 0  # 判断上方空间
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # 填充文本背景
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                            self.lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    def bev_box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """在鸟瞰图区域绘制边界框及标签，专用于3D透视变换可视化

        实现差异：
        - 绘制目标为frame_draw画布而非原始图像
        - 配合透视变换矩阵M实现空间投影效果
        - 右侧扩展画布区域用于显示变换后视图

        注：基础绘制逻辑与box_label一致，仅操作画布不同
        """
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = box[1] - h >= 0
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # OpenCV模式
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.frame_draw, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                outside = p1[1] - h - 3 >= 0
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.frame_draw, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.frame_draw, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                            self.lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    def polygon_label(self, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, label='', color=(128, 128, 128),
                      txt_color=(255, 255, 255)):
        """绘制四边形标注框及标签，支持关键点标注可视化
        核心功能：
        - 输入四角坐标(Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        - 自动处理浮点坐标转整数，适配不同数据源
        - 无效坐标过滤：全零坐标跳过绘制
        """
        if self.pil or not is_ascii(label):
            polygon = [int(x_tl), int(y_tl), int(x_tr), int(y_tr), int(x_br), int(y_br), int(x_bl), int(y_bl)]
            self.draw.polygon(polygon, outline=color)  # PIL多边形绘制
        else:
            polygon = np.array([[int(x_tl), int(y_tl)], [int(x_tr), int(y_tr)],
                                [int(x_br), int(y_br)], [int(x_bl), int(y_bl)]], np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            if np.mean(polygon) == 0:  # 过滤无效标注
                return
            cv2.polylines(self.im, [polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)

    def polygon_label_3d(self, xyxy, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, label='', color=(128, 128, 128),
                         txt_color=(255, 255, 255)):
        """3D立方体标注可视化，通过上下四边形+垂直线模拟立体效果
        特殊实现：
        - 根据目标框y坐标计算立方体高度
        - 绘制顶部四边形和连接立柱线
        - 适配张量输入自动转换CPU数值
        应用场景：自动驾驶中3D边界框的2D投影可视化
        """
        if self.pil or not is_ascii(label):
            polygon = [x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl]
            self.draw.polygon(polygon, outline=color)
        else:
            bbox_y_lt = xyxy[1]
            height = min(y_tl - bbox_y_lt, y_tr - bbox_y_lt)  # 基于框顶计算立方体高度

            # 底部四边形
            polygon = np.array([[x_tl.cpu(), y_tl.cpu()], [x_tr.cpu(), y_tr.cpu()],
                                [x_br.cpu(), y_br.cpu()], [x_bl.cpu(), y_bl.cpu()]], np.int32)
            if np.mean(polygon) == 0:
                return
            polygon = polygon.reshape((-1, 1, 2))

            # 顶部四边形(通过高度偏移生成)
            upper_polygon = np.array([[x_tl.cpu(), (y_tl - height).cpu()], [x_tr.cpu(), (y_tr - height).cpu()],
                                      [x_br.cpu(), (y_br - height).cpu()], [x_bl.cpu(), (y_bl - height).cpu()]],
                                     np.int32)
            upper_polygon = upper_polygon.reshape((-1, 1, 2))

            # 绘制立方体结构
            cv2.polylines(self.im, [polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.polylines(self.im, [upper_polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            # 绘制立柱连接线
            cv2.line(self.im, np.array([x_tl.cpu(), (y_tl - height).cpu()], np.int32),
                     np.array([x_tl.cpu(), y_tl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.im, np.array([x_tr.cpu(), (y_tr - height).cpu()], np.int32),
                     np.array([x_tr.cpu(), y_tr.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.im, np.array([x_br.cpu(), (y_br - height).cpu()], np.int32),
                     np.array([x_br.cpu(), y_br.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.im, np.array([x_bl.cpu(), (y_bl - height).cpu()], np.int32),
                     np.array([x_bl.cpu(), y_bl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)

    def polygon_label_3d_8points(self, xyxy, points, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """3D立方体8关键点标注可视化，带角点编号显示
        核心功能：
        - 输入为8个3D点投影后的2D坐标(x0,y0)...(x7,y7)
        - 绘制底部/顶部四边形及连接立柱线
        - 在每个角点显示编号(0-7)，用于视觉校验关键点定位

        实现细节：
        - 点坐标自动从张量转换至CPU数值
        - 使用抗锯齿线型提升视觉效果
        - 过滤全零无效标注避免干扰图像
        """
        x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7 = points
        if self.pil or not is_ascii(label):
            polygon = points
            self.draw.polygon(polygon, outline=color)
        else:
            # 基础立方体结构绘制(同label方法)
            polygon = np.array(
                [[x_4.cpu(), y_4.cpu()], [x_5.cpu(), y_5.cpu()], [x_6.cpu(), y_6.cpu()], [x_7.cpu(), y_7.cpu()]],
                np.int32)
            if np.mean(polygon) == 0:
                return
            polygon = polygon.reshape((-1, 1, 2))
            upper_polygon = np.array(
                [[x_0.cpu(), y_0.cpu()], [x_1.cpu(), y_1.cpu()], [x_2.cpu(), y_2.cpu()], [x_3.cpu(), y_3.cpu()]],
                np.int32)
            upper_polygon = upper_polygon.reshape((-1, 1, 2))

            # 生成各侧面多边形
            left_polygon = np.array(
                [[x_0.cpu(), y_0.cpu()], [x_3.cpu(), y_3.cpu()], [x_7.cpu(), y_7.cpu()], [x_4.cpu(), y_4.cpu()]],
                np.int32).reshape((-1, 1, 2))  # 左侧面
            right_polygon = np.array(
                [[x_1.cpu(), y_1.cpu()], [x_2.cpu(), y_2.cpu()], [x_6.cpu(), y_6.cpu()], [x_5.cpu(), y_5.cpu()]],
                np.int32).reshape((-1, 1, 2))  # 右侧面
            front_polygon = np.array(
                [[x_3.cpu(), y_3.cpu()], [x_2.cpu(), y_2.cpu()], [x_6.cpu(), y_6.cpu()], [x_7.cpu(), y_7.cpu()]],
                np.int32).reshape((-1, 1, 2))  # 前面
            back_polygon = np.array(
                [[x_0.cpu(), y_0.cpu()], [x_1.cpu(), y_1.cpu()], [x_5.cpu(), y_5.cpu()], [x_4.cpu(), y_4.cpu()]],
                np.int32).reshape((-1, 1, 2))  # 后面

            cv2.fillPoly(self.mask, [polygon], color)
            cv2.fillPoly(self.mask, [upper_polygon], color)
            cv2.fillPoly(self.mask, [left_polygon], color)
            # cv2.fillPoly(mask, [right_polygon], color)
            cv2.fillPoly(self.mask, [front_polygon], color)
            cv2.fillPoly(self.mask, [back_polygon], color)

    def draw_mask(self):
        """将半透明掩模叠加到原始图像上，实现遮挡可视化效果"""
        self.im = cv2.addWeighted(self.im, 1.0, self.mask, 0.5, 0)

    def polygon_label_4points(self, xyxy, points, color, labelcls, rect=True):
        """四边形关键点标注：绘制4个带编号的定位点及可选外接矩形"""
        txt_color = (0, 0, 255)
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for i in range(4):
            cv2.putText(self.im, str(i), (int(points[2 * i]) - 2, int(points[2 * i + 1]) - 2), 0, min(self.lw / 3, 1),
                        txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.circle(self.im, (int(points[2 * i]), int(points[2 * i + 1])), 3, color, -1)
        if rect:
            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)

    def order_points_old(self, pts):
        """四点排序：将无序四边形顶点排序为(左上,右上,右下,左下)"""
        rect = np.zeros((4, 2), dtype="int")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上点(坐标和最小)
        rect[2] = pts[np.argmax(s)]  # 右下点(坐标和最大)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上点(坐标差最小)
        rect[3] = pts[np.argmax(diff)]  # 左下点(坐标差最大)
        return rect

    def rotate_bev(self, img, xy, degrees):
        """鸟瞰图旋转变换：绕图像中心旋转并同步更新坐标点位置"""
        height, width = img.shape[:2]
        R = np.eye(3)
        a = -(90 + degrees) if degrees < -90 else -(90 + degrees)  # 角度归一化处理
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=1)
        img = cv2.warpAffine(img, R[:2], dsize=(width, height), borderValue=(0, 0, 0))

        # 坐标变换矩阵运算
        xy_tmp = np.ones((4, 3))
        xy_tmp[:, :2] = np.array(xy).reshape(4, 2)
        xy_tmp = (xy_tmp @ R.T)[:, :2]  # 应用旋转矩阵

        # 坐标边界裁剪
        xy_tmp[:, 0] = np.clip(xy_tmp[:, 0], 0, width)
        xy_tmp[:, 1] = np.clip(xy_tmp[:, 1], 0, height)
        return xy_tmp.astype(np.float32)

    def vector_space_label(self, xyxy, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, bev_angle, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        """
        三维空间标注引擎：实现3D立方体可视化与鸟瞰图投影转换
        核心功能：
        - 输入底面四边形坐标，自动构建3D立方体结构
        - 应用透视变换矩阵生成鸟瞰图投影
        - 计算最小外接旋转矩形并标注方向

        关键处理流程：
        1. 立方体构建：通过底面四边形推导顶部四边形位置，绘制立柱线形成3D结构
        2. 方向指示：计算底面中心点与顶边中点连线表示物体朝向
        3. 透视投影：使用预定义变换矩阵M将底面坐标映射到鸟瞰图空间
        4. 鸟瞰图绘制：在右侧扩展画布显示旋转矩形及方向标签

        特殊坐标处理：
        - 张量坐标转换：自动将GPU张量转为CPU数值，适配OpenCV操作
        - 鸟瞰图偏移：投影后x坐标右移原图宽度，实现并排显示
        - 无效数据过滤：全零坐标自动跳过绘制流程
        """
        if self.pil or not is_ascii(label):
            polygon = [x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl]
            self.draw.polygon(polygon, outline=color)  # box
        else:  # cv2
            bbox_y_lt = xyxy[1]
            height = min(y_tl - bbox_y_lt, y_tr - bbox_y_lt)
            # p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            polygon = np.array([[x_tl.cpu(), y_tl.cpu()], [x_tr.cpu(), y_tr.cpu()], [x_br.cpu(), y_br.cpu()], [x_bl.cpu(), y_bl.cpu()]], np.int32)
            if np.mean(polygon) == 0:
                # cv2.rectangle(self.frame_draw, p1, p2, color, -1, cv2.LINE_AA)  # filled
                return 0
            # polygon = np.array([[x_tl.cpu(), y_tl.cpu()], [x_br.cpu(), y_br.cpu()], [x_tr.cpu(), y_tr.cpu()], [x_bl.cpu(), y_bl.cpu()]], np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            upper_polygon = np.array([[x_tl.cpu(), (y_tl - height).cpu()], [x_tr.cpu(), (y_tr - height).cpu()], [x_br.cpu(), (y_br - height).cpu()], [x_bl.cpu(), (y_bl - height).cpu()]], np.int32)
            upper_polygon = upper_polygon.reshape((-1 , 1, 2))
            polygon_center = np.mean(polygon.reshape((-1, 2)), 0)
            # print(polygon_center)
            orientate_point = np.array([((x_tl + x_tr)/ 2).cpu(), ((y_tl + y_tr) /2).cpu()])
            cv2.polylines(self.frame_draw, [polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.polylines(self.frame_draw, [upper_polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.frame_draw, np.array([x_tl.cpu(), (y_tl - height).cpu()], np.int32), np.array([x_tl.cpu(), y_tl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.frame_draw, np.array([x_tr.cpu(), (y_tr - height).cpu()], np.int32), np.array([x_tr.cpu(), y_tr.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.frame_draw, np.array([x_br.cpu(), (y_br - height).cpu()], np.int32), np.array([x_br.cpu(), y_br.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.frame_draw, np.array([x_bl.cpu(), (y_bl - height).cpu()], np.int32), np.array([x_bl.cpu(), y_bl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.frame_draw, np.array(polygon_center, np.int32), np.array(orientate_point, np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.frame_draw, label, np.array([x_tl.cpu(), (y_tl - height - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.putText(self.frame_draw, '1', np.array([x_tr.cpu(), (y_tr - height - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.putText(self.frame_draw, '2', np.array([x_br.cpu(), (y_br - height - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.putText(self.frame_draw, '3', np.array([x_bl.cpu(), (y_bl - height - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            polygon = polygon.astype(np.float32)

            # pts_1 = np.float32([[0.0 * w, 0], [0.5 * w, 0], [0.6 * w, 1.0 * h], [0, 1.0 * h]])
            # pts_2 = np.float32([[0, 0], [0.33 * w, 0], [0.33 * w, 1.0 * h], [0, 1.0 * h]])
            # pts_1 = self.rotate_bev(self.frame_draw[:, :self.im.shape[1], :], self.pts_1, bev_angle)
            # print(self.pts_1)
            # print(pts_1)
            # self.M = cv2.getPerspectiveTransform(self.pts_1, self.pts_2)
            birds_eye_polygon = cv2.perspectiveTransform(polygon, self.M)
            # print('polygon:', polygon)
            # print('birds eye polygon:', birds_eye_polygon)
            birds_eye_polygon[:, :, 0] = birds_eye_polygon[:, :, 0] * 0.5 + self.im.shape[1]
            birds_eye_polygon[:, :, 1] = birds_eye_polygon[:, :, 1]
            birds_eye_polygon = birds_eye_polygon.astype(np.int32)
            rect = cv2.minAreaRect(birds_eye_polygon)
            # x, y, w, h, a = rect[0][0], rect[0][1], rect[1][0], rect[1][0], rect[2]
            # rect_refine = ((x, y), (w, h), a)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # box = self.order_points_old(box)
            box = box.reshape((-1, 1, 2))
            cv2.polylines(self.frame_draw, [box], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            # bev_angle = 1
            # print(self.im.shape, '|', h, '|', w)
            # frame_draw[:h, :w, :] = self.im
            # self.im = cv2.resize(frame_draw, (w, h))
            # self.im = frame_draw
            # cv2.polylines(self.im, [upper_polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.line(self.im, np.array([x_tl.cpu(), (y_tl - height).cpu()], np.int32), np.array([x_tl.cpu(), y_tl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.line(self.im, np.array([x_tr.cpu(), (y_tr - height).cpu()], np.int32), np.array([x_tr.cpu(), y_tr.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.line(self.im, np.array([x_br.cpu(), (y_br - height).cpu()], np.int32), np.array([x_br.cpu(), y_br.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.line(self.im, np.array([x_bl.cpu(), (y_bl - height).cpu()], np.int32), np.array([x_bl.cpu(), y_bl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.frame_draw, label, np.array([box[0][0][0], box[0][0][1] - 2], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.putText(self.im, '1', np.array([x_tr.cpu(), (y_tr - height - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.putText(self.im, '2', np.array([x_br.cpu(), (y_br - height - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            # cv2.putText(self.im, '3', np.array([x_bl.cpu(), (y_bl - height - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """在图像上绘制矩形框(PIL专用方法)"""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        """添加文本标注(PIL专用方法)，自动调整文本基线位置"""
        w, h = self.font.getsize(text)  # 获取文本尺寸
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        """返回PIL.Image对象转换的numpy数组格式的标注结果"""
        return np.asarray(self.im)

    def bev_result(self):
        """返回鸟瞰图扩展画布的numpy数组结果，包含原始图像与BEV可视化"""
        return np.asarray(self.frame_draw)

def plot_images_ssod(images, targets, paths=None, fname='images.jpg', num_points=0, names=None,  max_size=1920, max_subplots=16):
    """
        批量图像网格标注工具，支持目标检测框及置信度可视化
        核心功能：
        - 输入批处理图像张量，自动拼接为网格马赛克图
        - 绘制边界框、类别标签及三重置信度(obj_conf/cls_conf/conf)
        - 自适应缩放保证输出图像不超过最大尺寸限制

        关键处理流程：
        1. 输入张量转换：将PyTorch张量转为numpy数组并进行反归一化
        2. 网格布局计算：根据batch_size和max_subplots确定行列数
        3. 图像拼接：将多张图像排列为ns x ns的网格布局
        4. 智能缩放：基于max_size约束进行整体尺寸调整
        5. 标注绘制：使用Annotator工具添加边框及多维度置信信息

        参数说明：
        max_size: 输出图像最大边长(保持纵横比)
        max_subplots: 最大子图数量限制(16=4x4网格)
        num_points: 关键点数量(当前版本未激活)
        names: 类别名称映射字典
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255.0  # de-normalise (optional)
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    # print('scale:', scale)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    # fs = 25
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        # print('targets:', targets)
        if len(targets) > 0:
            # print('targets:', targets, '==================================')
            ti = targets[targets[:, 0] == i]  # image targets
            boxes = xywh2xyxy(ti[:, 2:6]).T
            # keypoints = ti[:, -8:]
            classes = ti[:, 1].astype('int')
            # labels = ti.shape[1] == 14  # labels if no conf column
            # labels = ti.shape[1] == (6 + num_points * 2)  # labels if no conf column
            labels = False
            conf = ti[:, 6]  # check for confidence presence (label vs pred)
            obj_conf = ti[:, 7]
            cls_conf = ti[:, 8]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y

            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.1:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f} {obj_conf[j]:.1f} {cls_conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)


def plot_images(images, targets, paths=None, fname='images.jpg', num_points=0, names=None, max_size=1920,
                max_subplots=16):
    """
    绘制带标注框的图像网格

    参数:
        images: 输入图像张量/数组(支持torch.Tensor或numpy格式)
        targets: 目标检测数据，包含类别和边界框信息
        paths: 图像路径列表(用于文件名标注)
        fname: 输出文件名
        num_points: 关键点数量(当前版本未启用)
        names: 类别名称字典
        max_size: 输出图像最大尺寸
        max_subplots: 最大子图数量

    重点细节:
        1. 自动处理张量到numpy的转换和反归一化(当像素值<=1时)
        2. 创建正方形排列的图像马赛克，自动计算子图布局
        3. 边界框坐标转换(xywh->xyxy)和尺度适配
        4. 支持两种标注模式：带置信度分数和不带置信度的纯标签
        5. 自动调整字体大小和线宽，适配不同分辨率
        6. 边界框颜色根据类别自动分配
        7. 文件名显示时自动截断前40个字符
    """
    if isinstance(images, torch.Tensor): images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1: images *= 255.0  # 反归一化(当输入为浮点归一化图像时)
    bs, _, h, w = images.shape  # 获取batch大小和图像尺寸
    bs = min(bs, max_subplots)  # 限制子图数量
    ns = np.ceil(bs ** 0.5)  # 计算网格尺寸(平方根取整)

    # 创建马赛克画布
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        if i == max_subplots: break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 计算子图位置
        mosaic[y:y + h, x:x + w, :] = im.transpose(1, 2, 0)  # 填充图像到网格

    # 图像缩放(保持宽高比)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h, w = math.ceil(scale * h), math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # 初始化标注工具
    fs = int((h + w) * ns * 0.01)  # 自动计算字体大小
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)

    # 遍历所有子图进行标注
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 当前子图原点坐标
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # 绘制白边

        # 添加文件名标注
        if paths: annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))

        # 处理目标数据
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # 筛选当前图像的目标
            boxes = xywh2xyxy(ti[:, 2:6]).T  # 转换坐标格式(w,h -> x1,y1,x2,y2)
            classes = ti[:, 1].astype('int')  # 获取类别索引
            labels = ti.shape[1] == (6 + num_points * 2)  # 判断是否为标签模式(非预测结果)
            conf = None if labels else ti[:, 6]  # 置信度分数(预测模式时存在)

            # 调整边界框坐标尺度
            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # 处理归一化坐标
                    boxes[[0, 2]] *= w;
                    boxes[[1, 3]] *= h
                elif scale < 1:
                    boxes *= scale  # 绝对坐标缩放
                boxes[[0, 2]] += x;
                boxes[[1, 3]] += y  # 转换到马赛克坐标系

            # 绘制边界框和标签
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)  # 根据类别获取颜色
                cls = names[cls] if names else cls  # 转换类别索引为名称
                if labels or conf[j] > 0.1:  # 过滤低置信度预测
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)

    annotator.im.save(fname)  # 保存最终结果


def plot_labels(labels, names=(), save_dir=Path('')):
    """
    可视化数据集标签分布及边界框信息

    参数:
        labels: 包含类别和边界框的标签数据(numpy数组)
        names: 类别名称列表(可选)
        save_dir: 输出目录路径

    重点细节:
        1. 生成4种可视化图表：类别分布直方图、边界框位置热力图、尺寸分布图、样本框展示
        2. 自动处理边界框坐标转换(xywh->xyxy)和归一化
        3. 使用双后端策略：svg加速绘图，最后切换回非交互式Agg模式
        4. 颜色映射根据类别自动生成
        5. 对超过30个类别的情况优化显示方式
        6. 边界框示例图使用固定2000x2000画布展示前1000个样本
    """
    # 处理输入数据
    c, b = labels[:, 0], labels[:, 1:5].transpose()  # 分解类别和边界框数据
    nc = int(c.max() + 1)  # 计算类别总数
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])  # 创建DataFrame

    # 生成seaborn相关矩阵图
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # 配置matplotlib
    matplotlib.use('svg')  # 使用svg后端加速绘图
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()

    # 类别分布直方图
    hist = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('实例数量')
    if 0 < len(names) < 30:  # 合理类别数时显示名称
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('类别分布')

    # 生成边界框位置和尺寸分布图
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)  # 位置热力图
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)  # 尺寸散点图

    # 边界框示例图(中心点对齐)
    labels[:, 1:3] = 0.5  # 中心点归一化
    labels[:, 1:5] = xywh2xyxy(labels[:, 1:5]) * 2000  # 转换坐标并放大到2000x2000画布
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)  # 创建白底画布
    # 绘制前1000个边界框
    for cls, *box in labels[:1000, :5]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))
    ax[1].imshow(img)
    ax[1].axis('off')

    # 美化图表样式
    for a in ax:
        for s in ['top', 'right', 'left', 'bottom']:
            a.spines[s].set_visible(False)  # 隐藏坐标轴边框

    # 保存并清理
    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')  # 切换回非GUI后端
    plt.close()