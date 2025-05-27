import copy
from yolo_models.neck.yolov11_neck import YOLOv11Neck

def build_neck(cfg,nas_arch=None,in_channels=None):
    fpn_cfg = copy.deepcopy(cfg)
    name = cfg.Model.Neck.name
    if name == "YoLov11":
        return YOLOv11Neck(fpn_cfg)
    else:
        raise NotImplementedError