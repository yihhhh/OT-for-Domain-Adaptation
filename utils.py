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

class IterationBasedBatchSampler:
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, src_batch_sampler, dst_batch_sampler, batch_size, num_iterations, start_iter=0):
        self.src_batch_sampler = src_batch_sampler
        self.dst_batch_sampler = dst_batch_sampler
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        for i in range(self.num_iterations):
            src_batch, dst_batch = [], []
            for s in self.src_batch_sampler:
                # print("src shape: ", self.src_batch_sampler.data_source[s].shape)
                src_batch.append(self.src_batch_sampler.data_source[s])
                if self.batch_size == len(src_batch):
                    break
            for d in self.dst_batch_sampler:
                # print("dst shape: ", self.dst_batch_sampler.data_source[d].shape)
                dst_batch.append(self.dst_batch_sampler.data_source[d])
                if self.batch_size == len(dst_batch):
                    break
            yield torch.stack(src_batch, dim=0), torch.stack(dst_batch, dim=0)

    def __len__(self):
        return self.num_iterations
