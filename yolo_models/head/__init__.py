import copy
from .yolov11_head import *

def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.Model.Head.name
    if name == 'YoloV11':
        return Detect(head_cfg)
    else:
        return NotImplementedError
