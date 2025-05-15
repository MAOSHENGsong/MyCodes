import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    """安全下载文件，支持备选URL和完整性检查

    参数:
        file: 文件保存路径
        url: 首选下载URL
        url2: 备用下载URL(可选)
        min_bytes: 最小文件字节数(默认1E0)，小于此值视为无效
        error_msg: 自定义错误信息(可选)

    逻辑:
        1. 优先使用torch的下载方法
        2. 失败后切换curl命令下载(支持断点续传)
        3. 始终检查文件完整性，删除不完整文件

    异常处理:
        - 下载失败自动删除不完整文件
        - 最终检查不通过时显示自定义错误信息
        - 使用--retry 3实现curl重试机制
        - -C - 参数支持断点续传

    注意:
        - 同时处理本地路径的Path对象转换
        - 使用文件大小作为完整性校验标准
        - 删除操作使用missing_ok避免文件不存在的报错
    """
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print('')

def attempt_download(file, repo='AlibabaResearch/efficientteacher'):  # from utils.downloads import *; attempt_download()
    """🔄 智能文件下载器，自动处理GitHub资源获取

        参数:
            file: 目标文件路径/URL
            repo: GitHub仓库地址(默认AlibabaResearch/efficientteacher)

        核心功能:
            1. 本地文件存在时直接返回路径
            2. 支持直接下载HTTP/HTTPS文件
            3. GitHub release资源自动探测：
               - 通过API获取最新版本资源
               - 备用机制获取git标签版本
               - 多平台下载源支持

        特殊处理:
            - URL解码文件名处理特殊字符
            - 自动创建父级目录
            - 处理GitHub API限流/失败的备用方案
            - 内置经典YOLO模型文件名作为备用资源列表

        异常处理:
            - 使用subprocess获取git标签时的错误捕获
            - GitHub API请求失败自动切换本地git版本号
            - 最终版本号获取失败使用硬编码v6.0

        注意:
            - 优先使用github.com的release下载源
            - 包含注释掉的google storage备用下载源
            - 通过min_bytes=1E5(约100KB)验证文件有效性
        """
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            name = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            safe_download(file=name, url=url, min_bytes=1E5)
            return name

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
            assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response['tag_name']  # i.e. 'v1.0'
        except:  # fallback plan
            assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except:
                tag = 'v6.0'  # current release

        if name in assets:
            safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

    return str(file)