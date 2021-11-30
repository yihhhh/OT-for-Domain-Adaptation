import numpy as np
import os
import sys
import yaml
import csv
from easydict import EasyDict
import wandb

import torch


def wandb_init(project_name, group_id, exp_id):
    wandb.init(project=project_name, entity='yihan', group=group_id, name=exp_id)

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        dict = yaml.load(f, Loader=yaml.FullLoader)
        try:
            wandb.config.update(dict)
        except:
            print("wandb not initiated.")
        return EasyDict(dict)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def save_ckpt(model, ckpt_pth):
    torch.save(model.state_dict(), ckpt_pth)

def load_ckpt(model, ckpt_pth):
    model.load_state_dict(torch.load(ckpt_pth))

def l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix holding all ||.||_2 distances."""
    xTy = 2 * x.matmul(y.transpose(0, 1))
    x2 = torch.sum(x ** 2, dim=1)[:, None]
    y2 = torch.sum(y ** 2, dim=1)[None, :]
    K = x2 + y2 - xTy
    return K
