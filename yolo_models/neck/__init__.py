import copy
from yolo_models.neck.yolov11_neck import YoloV11Neck

def build_neck(cfg,nas_arch=None,in_channels=None):
    fpn_cfg = copy.deepcopy(cfg)
    name = cfg.Model.Neck.name
    if name == "YoloV11":
        return YoloV11Neck(fpn_cfg)
    else:
        raise NotImplementedError