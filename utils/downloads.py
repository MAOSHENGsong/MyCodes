import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    """安全下载文件，支持多URL回退和完整性校验

    功能特性：
    - 自动重试机制（通过curl实现）
    - 下载中断后支持断点续传
    - 文件完整性检查（最小字节数验证）
    - 自动清理不完整下载文件

    参数：
    file      : 下载文件保存路径
    url       : 首选下载URL
    url2      : 备用下载URL（可选）
    min_bytes : 文件最小有效字节数（默认1字节）
    error_msg : 自定义错误提示信息

    返回：无，直接保存文件到指定路径
    """

    file = Path(file)  # 转换为Path对象（跨平台路径处理）
    assert_msg = f"文件 '{file}' 下载失败或大小 < {min_bytes}字节"  # 基础错误信息

    # 首选URL下载尝试 -----------------------------------------------------
    try:
        print(f'正在从 {url} 下载到 {file}...')
        torch.hub.download_url_to_file(url, str(file))  # 使用PyTorch官方下载方法

        # 验证文件完整性（存在性 + 大小检查）
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg

    # 备用URL回退机制 -----------------------------------------------------
    except Exception as e:
        file.unlink(missing_ok=True)  # 删除不完整文件（如果存在）
        print(f'错误: {e}\n尝试备用URL: {url2 or url}...')

        # 使用curl命令下载（支持重试/断点续传）
        # -L 跟随重定向 --retry 3 重试次数 -C - 自动续传
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")

        # 最终校验 -----------------------------------------------------------
    finally:
        # 最终检查（处理前两步可能的失败）
        if not file.exists() or file.stat().st_size < min_bytes:
            file.unlink(missing_ok=True)  # 清理无效文件
            print(f"错误: {assert_msg}\n{error_msg}")
        print('')  # 空行分隔日志

def attempt_download(file, repo='AlibabaResearch/efficientteacher'):
    """智能文件下载器，支持多种来源获取缺失文件

    功能特性：
    - 自动检测本地文件是否存在
    - 支持直接URL下载
    - 支持GitHub Release资源下载
    - 内置多CDN回退机制
    - 自动创建目录结构
    - 处理特殊字符编码

    参数：
    file : str 目标文件路径（可含URL）
    repo : str GitHub仓库地址（默认阿里云EfficientTeacher仓库）

    返回：
    str 下载后的本地文件路径
    """

    file = Path(str(file).strip().replace("'", ''))  # 标准化路径处理

    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name  # 解码特殊字符（如%20→空格）

        # 直接URL下载处理 ------------------------------------------------
        if str(file).startswith(('http:/', 'https:/')):  # 识别URL格式
            url = str(file).replace(':/', '://')  # 修复Pathlib的URL解析问题
            name = name.split('?')[0]  # 去除URL参数（如认证token）
            safe_download(file=name, url=url, min_bytes=1E5)  # 执行安全下载
            return name

        # GitHub资源下载处理 ---------------------------------------------
        file.parent.mkdir(parents=True, exist_ok=True)  # 递归创建父目录

        try:  # 尝试通过GitHub API获取最新发布信息
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()
            assets = [x['name'] for x in response['assets']]  # 资源文件名列表
            tag = response['tag_name']  # 最新发布版本标签（如v1.0）
        except Exception:  # API请求失败时的备用方案
            assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt']  # 常见模型文件名
            try:  # 尝试获取git标签
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except Exception:
                tag = 'v6.0'  # 保底版本标签

        if name in assets:  # 校验文件是否在发布资源中
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',  # 主CDN
                # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # 备用CDN
                min_bytes=1E5,  # 最小文件大小验证（100KB）
                error_msg=f'{file}缺失，请从https://github.com/{repo}/releases/下载'
            )

    return str(file)  # 返回兼容路径字符串