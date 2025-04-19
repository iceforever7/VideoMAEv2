import io
import cv2
import numpy as np
from decord import VideoReader, cpu
from functools import partial

try:
    from petrel_client.client import Client
    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False

# 将内部函数移到外部作为顶级函数
def _video_loader(video_path, client=None):
    """加载视频函数，作为顶级函数以支持序列化"""
    if client is not None and 's3:' in video_path:
        video_path = io.BytesIO(client.get(video_path))

    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    return vr

def get_video_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    # 使用partial创建可序列化的函数对象
    return partial(_video_loader, client=_client)

# 同样修改图像加载器
def _image_loader(frame_path, client=None):
    """加载图像函数，作为顶级函数以支持序列化"""
    if client is not None and 's3:' in frame_path:
        img_bytes = client.get(frame_path)
    else:
        with open(frame_path, 'rb') as f:
            img_bytes = f.read()

    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img

def get_image_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    # 使用partial创建可序列化的函数对象
    return partial(_image_loader, client=_client)
