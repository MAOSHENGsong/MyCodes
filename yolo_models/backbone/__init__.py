import copy
from .yolov11_backbone import *

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    # name = backbone_cfg.pop("name")
    name = backbone_cfg.Model.Backbone.name
    if name == "YoloV11":
        return YoloV11BackBone(backbone_cfg)
    else:
        raise NotImplementedError