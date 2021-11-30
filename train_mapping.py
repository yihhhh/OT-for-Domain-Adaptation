import numpy as np
import pickle
from tqdm import tqdm

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
    args = utils.get_cfg(config_file)
    ot_args = args.ot_plan
    map_args = args.mapping
    print(group_id, exp_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data ...")
    ot_src_loader, ot_dst_loader, src_len, dst_len = load_data(ot_args)
    map_src_loader, map_dst_loader, _, _ = load_data(map_args)

    ot_planner = OTPlan(source_type='continuous', target_type='continuous',
                     target_dim=args.target_dim, source_dim=args.source_dim)
    if ot_args.load_model:
        ot_planner.load_model(ot_args.load_name)
    if ot_args.train_model:
        ot_optimizer = Adam(ot_planner.parameters(), amsgrad=True, lr=ot_args.lr)
        train_ot_plan(ot_args, ot_src_loader, ot_dst_loader, ot_planner, ot_optimizer, device=device)
        ot_planner.save_model(ot_args.save_name)

    mapping = Mapping(ot_planner, dim=args.target_dim)
    if map_args.load_model:
        mapping.load_model(map_args.load_name)
    if map_args.train_model:
        map_optimizer = Adam(mapping.parameters(), amsgrad=True, lr=map_args.lr)
        train_mapping(map_args, map_src_loader, map_dst_loader, mapping, map_optimizer, device=device)
        mapping.save_model(map_args.save_name)

    print('Train finished!')


def load_data(args):
    src_dataset = ImageDataset('./dataset', 'mnist')
    dst_dataset = ImageDataset('./dataset', 'usps')
    src_loader = DataLoader(src_dataset, batch_size=None, sampler=BatchSampler(RandomSampler(src_dataset), batch_size=args.batch_size))
    dst_loader = DataLoader(dst_dataset, batch_size=None, sampler=BatchSampler(RandomSampler(dst_dataset), batch_size=args.batch_size))
    return src_loader, dst_loader, len(src_dataset), len(dst_dataset)


def train_ot_plan(args, source_loader, target_loader, ot_planner, optimizer, device):
    print("Learning OT Plan ...")
    with tqdm(range(args.epoch)) as t:
        for i in t:
            optimizer.zero_grad()
            source_data = source_loader.next().to(device)
            target_data = target_loader.next().to(device)
            loss = ot_planner.loss(source_data, target_data)
            loss.backward()
            optimizer.step()

            loss_log = loss.cpu().detach().item()

            t.set_description("train OT plan")
            t.set_postfix(loss=loss_log, lr=optimizer.param_groups[0]['lr'])

            # wandb.log({"ot_planner/train loss": loss_log})

def train_mapping(args, source_loader, target_loader, mapping, optimizer, device):
    print("Learning Mapping ...")
    with tqdm(range(args.epoch)) as t:
        for i in t:
            optimizer.zero_grad()
            source_data = source_loader.next().to(device)
            target_data = target_loader.next().to(device)
            loss = mapping.loss(source_data, target_data)
            loss.backward()
            optimizer.step()

            loss_log = loss.cpu().detach().item()

            t.set_description("train OT plan")
            t.set_postfix(loss=loss_log, lr=optimizer.param_groups[0]['lr'])

            # wandb.log({"mapping/train loss": loss_log})