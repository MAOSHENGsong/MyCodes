import torch
import torch.distributed as dist
from trainer.trainer import Trainer
from trainer.ssod_trainer import SSODTrainer
from configs.defaults import get_cfg
from utils.general import increment_path, print_args,  set_logging
from pathlib import Path
import os
import argparse
from utils.callbacks import Callbacks
from utils.torch_utils import select_device
import sys
import val

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # EfficientTeacher root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def setup(cfg):
    cfg.save_dir = str(increment_path(Path(cfg.project) / cfg.name, exist_ok=cfg.exist_ok))
    device = select_device(cfg.device)
    return device

def main(opt, callbacks=Callbacks()):
    cfg = get_cfg()
    cfg.merge_from_file(opt.cfg)
    cfg.merge_from_list(opt.opts)

    device = setup(cfg)
    cfg.freeze()
    if cfg.SSOD.train_domain:
        trainer = SSODTrainer(cfg, device, callbacks)
    else:
        trainer = Trainer(cfg, device, callbacks)
    trainer.train(callbacks, val)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)