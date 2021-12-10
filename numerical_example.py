import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
from torch.utils.data import RandomSampler
from torch.optim import Adam
from sklearn import datasets

import utils
from dataset import NumericalDataset
from models.ot_model import OTPlan
from models.mapping import Mapping

def cli_main(type='moon'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data ...")
    src_data, dst_data, src_label, dst_label = generate_data(type, 1000, 0.1, 0.5)
    ot_src_sampler, ot_dst_sampler = load_data(src_data, dst_data, src_label, dst_label)
    map_src_sampler, map_dst_sampler = load_data(src_data, dst_data, src_label, dst_label)

    ot_planner = OTPlan(source_type='continuous', target_type='continuous',
                        target_dim=2, source_dim=2,
                        regularization='l2', alpha=0.001,
                        hiddens=[128, 256, 256, 128], device=device)

    ot_optimizer = Adam(ot_planner.parameters(), amsgrad=True, lr=0.0001)
    train_ot_plan(ot_src_sampler, ot_dst_sampler, ot_planner, ot_optimizer, device=device)

    mapping = Mapping(ot_planner, dim=2, hidden_size=[256, 512, 256], device=device)
    map_optimizer = Adam(mapping.parameters(), amsgrad=True, lr=0.0001)
    train_mapping(map_src_sampler, map_dst_sampler, mapping, map_optimizer, device=device)

    print('Train finished!')

    mapped_src_data = mapping(torch.FloatTensor(src_data).to(device)).detach().cpu().numpy()

    plt.scatter(src_data[:, 0], src_data[:, 1], label='source data', color='g', alpha=0.5)
    plt.scatter(dst_data[:, 0], dst_data[:, 1], label='target data', color='b', alpha=0.5)
    plt.scatter(mapped_src_data[:, 0], mapped_src_data[:, 1], label='mapped source data', color='r', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./figs/double_{}_mapping.png'.format(type))


def generate_data(type, n_samples, noise, factor=0.5):
    if type == 'moon':
        x, y = datasets.make_moons(n_samples=n_samples, noise=noise)
    else:
        x, y = datasets.make_circles(n_samples=n_samples, factor=factor, noise=noise)

    src_idx = np.where(y == 0)[0]
    dst_idx = np.where(y != 0)[0]
    src_data = x[src_idx, :]
    dst_data = x[dst_idx, :]

    return src_data, dst_data, np.zeros(src_data.shape[0]), np.ones(dst_data.shape[0])


def load_data(src_data, dst_data, src_label, dst_label):
    src_dataset = NumericalDataset(src_data, src_label)
    dst_dataset = NumericalDataset(dst_data, dst_label)
    print("src dataset len: ", len(src_dataset), " dst dataset len: ", len(dst_dataset))
    src_sampler = RandomSampler(src_dataset, replacement=True)
    dst_sampler = RandomSampler(dst_dataset, replacement=True)
    return src_sampler, dst_sampler

def train_ot_plan(source_sampler, target_sampler, ot_planner, optimizer, device):
    print("Learning OT Plan ...")

    iter_sampler = utils.IterationBasedBatchSampler(source_sampler, target_sampler, 64, 1000)
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


def train_mapping(source_sampler, target_sampler, mapping, optimizer, device):
    print("Learning Mapping ...")
    iter_sampler = utils.IterationBasedBatchSampler(source_sampler, target_sampler, 64, 1000)
    with tqdm(enumerate(iter_sampler)) as t:
        for i, batch in t:
            optimizer.zero_grad()
            source_data = batch[0].to(device)
            target_data = batch[1].to(device)
            loss = mapping.loss(source_data, target_data)
            loss.backward()
            optimizer.step()

            loss_log = loss.cpu().detach().item()

            t.set_description("train Mapping")
            t.set_postfix(loss=loss_log, lr=optimizer.param_groups[0]['lr'])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, default='moon')
    args = parser.parse_args()
    cli_main(args.type)
