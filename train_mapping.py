import os
import numpy as np
import wandb
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.utils.data import RandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

import utils
from dataset import MappingDataset
from models.ot_model import OTPlan
from models.mapping import Mapping

def cli_main(config_file='config', group_id='group', exp_id='exp', mode='formal'):
    print("Reading configurations ...")
    utils.wandb_init("11785-project", group_id, exp_id)
    args = utils.load_config(os.path.join('./configs', '{}.yml'.format(config_file)))
    ot_args = args.ot_plan
    map_args = args.mapping
    save_dir = os.path.join('./mapping_checkpoints', group_id, exp_id)
    if map_args.train_model or ot_args.train_model:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    print('save dir: ', save_dir)
    print(group_id, exp_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_split = np.load('./dataset/mapping/label_split_mnist.npy', allow_pickle=True).item()
    dst_split = np.load('./dataset/mapping/label_split_usps.npy', allow_pickle=True).item()

    print("Loading data ...")
    ot_src_sampler, ot_dst_sampler = load_data()
    map_src_sampler, map_dst_sampler = load_data()

    ot_planner = OTPlan(source_type='continuous', target_type='continuous',
                        target_dim=args.target_dim, source_dim=args.source_dim,
                        regularization=ot_args.regularization, alpha=ot_args.alpha,
                        hiddens=ot_args.hidden_size, device=device)
    if ot_args.load_model:
        ot_planner.load_model(ot_args.load_name)
    if ot_args.train_model:
        ot_optimizer = Adam(ot_planner.parameters(), amsgrad=True, lr=ot_args.lr)
        ot_lr_scheduler = StepLR(ot_optimizer, step_size=ot_args.step_size, gamma=0.1)
        train_ot_plan(src_split, dst_split, ot_args, ot_src_sampler, ot_dst_sampler, ot_planner, ot_optimizer, ot_lr_scheduler, device=device)
        ot_planner.save_model(save_dir)

    mapping = Mapping(ot_planner, dim=args.target_dim, hidden_size=map_args.hidden_size, device=device)
    if map_args.load_model:
        mapping.load_model(map_args.load_name)
    if map_args.train_model:
        map_optimizer = Adam(mapping.parameters(), amsgrad=True, lr=map_args.lr)
        map_lr_scheduler = StepLR(map_optimizer, step_size=map_args.step_size, gamma=0.1)
        train_mapping(map_args, map_src_sampler, map_dst_sampler, mapping, map_optimizer, map_lr_scheduler, device=device)
        mapping.save_model(save_dir)

    print('Train finished!')


def load_data():
    src_dataset = MappingDataset('./dataset/mapping', 'mnist')
    dst_dataset = MappingDataset('./dataset/mapping', 'usps')
    print("src dataset len: ", len(src_dataset), " dst dataset len: ", len(dst_dataset))
    src_sampler = RandomSampler(src_dataset, replacement=True)
    dst_sampler = RandomSampler(dst_dataset, replacement=True)
    return src_sampler, dst_sampler


def train_ot_plan(src_split, dst_split, args, source_sampler, target_sampler, ot_planner, optimizer, scheduler, device):
    print("Learning OT Plan ...")
    # best_loss = np.inf
    # src_bin = np.zeros(10)
    # dst_bin = np.zeros(10)
    iter_sampler = utils.IterationBasedBatchSampler(source_sampler, target_sampler, args.batch_size, args.epoch)
    with tqdm(enumerate(iter_sampler)) as t:
        for i, batch in t:
            # src_bin += utils.bin_index(src_split, batch[2])
            # dst_bin += utils.bin_index(dst_split, batch[3])
            optimizer.zero_grad()
            source_data = batch[0].to(device)
            target_data = batch[1].to(device)
            # source_label = batch[2].to(device)
            # target_label = batch[3].to(device)
            loss = ot_planner.loss(source_data, target_data)
            loss.backward()
            optimizer.step()

            loss_log = loss.cpu().detach().item()

            t.set_description("train OT plan")
            t.set_postfix(loss=loss_log, lr=optimizer.param_groups[0]['lr'])

            wandb.log({"ot_planner/train loss": loss_log})
            wandb.log({"ot_planner/learning rate": optimizer.param_groups[0]['lr']})

            scheduler.step()

    # print(src_bin)
    # print(dst_bin)

def train_mapping(args, source_sampler, target_sampler, mapping, optimizer, scheduler, device):
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

            wandb.log({"mapping/train loss": loss_log})
            wandb.log({"mapping/learning rate": optimizer.param_groups[0]['lr']})

            scheduler.step()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--group_id', type=str, default='trial')
    parser.add_argument('--exp_id', type=str, default='trial')
    parser.add_argument('--mode', type=str, default='formal')
    args = parser.parse_args()
    cli_main(args.config, args.group_id, args.exp_id, args.mode)
