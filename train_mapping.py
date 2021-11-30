import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torch.optim import Adam

import utils
from data import ImageDataset
from models.ot_model import OTPlan
from models.mapping import Mapping

def cli_main(config_file='config', group_id='group', exp_id='exp', mode='formal'):
    print("Reading configurations ...")
    # utils.wandb_init("11785-project", group_id, exp_id)
    args = utils.load_config(os.path.join('./configs', '{}.yml'.format(config_file)))
    ot_args = args.ot_plan
    map_args = args.mapping
    print(group_id, exp_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data ...")
    ot_src_sampler, ot_dst_sampler = load_data()
    map_src_sampler, map_dst_sampler = load_data()

    ot_planner = OTPlan(source_type='continuous', target_type='continuous',
                        target_dim=args.target_dim, source_dim=args.source_dim,
                        regularization=ot_args.regularization, alpha=ot_args.alpha,
                        device=device)
    if ot_args.load_model:
        ot_planner.load_model(ot_args.load_name)
    if ot_args.train_model:
        ot_optimizer = Adam(ot_planner.parameters(), amsgrad=True, lr=ot_args.lr)
        train_ot_plan(ot_args, ot_src_sampler, ot_dst_sampler, ot_planner, ot_optimizer, device=device)
        ot_planner.save_model(ot_args.save_name)

    mapping = Mapping(ot_planner, dim=args.target_dim)
    if map_args.load_model:
        mapping.load_model(map_args.load_name)
    if map_args.train_model:
        map_optimizer = Adam(mapping.parameters(), amsgrad=True, lr=map_args.lr)
        train_mapping(map_args, map_src_sampler, map_dst_sampler, mapping, map_optimizer, device=device)
        mapping.save_model(map_args.save_name)

    print('Train finished!')


def load_data():
    src_dataset = ImageDataset('./dataset', 'mnist')
    dst_dataset = ImageDataset('./dataset', 'usps')
    print("src dataset len: ", len(src_dataset), " dst dataset len: ", len(dst_dataset))
    src_sampler = RandomSampler(src_dataset, replacement=True)
    dst_sampler = RandomSampler(dst_dataset, replacement=True)
    # src_loader = DataLoader(src_dataset, batch_size=None, sampler=BatchSampler(RandomSampler(src_dataset), batch_size=args.batch_size, drop_last=False))
    # dst_loader = DataLoader(dst_dataset, batch_size=None, sampler=BatchSampler(RandomSampler(dst_dataset), batch_size=args.batch_size, drop_last=False))
    return src_sampler, dst_sampler


def train_ot_plan(args, source_sampler, target_sampler, ot_planner, optimizer, device):
    print("Learning OT Plan ...")
    best_loss = np.inf
    iter_sampler = utils.IterationBasedBatchSampler(source_sampler, target_sampler, args.batch_size, args.epoch)
    with tqdm(enumerate(iter_sampler)) as t:
        for i, batch in t:
            optimizer.zero_grad()
            source_data = batch[0].to(device)
            target_data = batch[1].to(device)
            loss = ot_planner.loss(source_data, target_data)
            loss.backward()
            optimizer.step()

            loss_log = loss.cpu().detach().item()

            t.set_description("train OT plan")
            t.set_postfix(loss=loss_log, lr=optimizer.param_groups[0]['lr'])

            # wandb.log({"ot_planner/train loss": loss_log})

def train_mapping(args, source_sampler, target_sampler, mapping, optimizer, device):
    print("Learning Mapping ...")
    iter_sampler = utils.IterationBasedBatchSampler(source_sampler, target_sampler, args.batch_size, args.epoch)
    with tqdm(enumerate(iter_sampler)) as t:
        for i, batch in t:
            optimizer.zero_grad()
            source_data = batch[0].to(device)
            target_data = batch[1].to(device)
            loss = mapping.loss(source_data, target_data)
            loss.backward()
            optimizer.step()

            loss_log = loss.cpu().detach().item()

            t.set_description("train OT plan")
            t.set_postfix(loss=loss_log, lr=optimizer.param_groups[0]['lr'])

            # wandb.log({"mapping/train loss": loss_log})

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--group_id', type=str, default='trial')
    parser.add_argument('--exp_id', type=str, default='trial')
    parser.add_argument('--mode', type=str, default='formal')
    args = parser.parse_args()
    cli_main(args.config, args.group_id, args.exp_id, args.mode)
