import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
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

    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data ...")
    src_dataset = ImageDataset('./dataset', 'mnist')
    dst_dataset = ImageDataset('./dataset', 'usps')
    src_split = np.load('./dataset/label_split_mnist.npy', allow_pickle=True).item()
    dst_split = np.load('./dataset/label_split_usps.npy', allow_pickle=True).item()

    ot_planner = OTPlan(source_type='continuous', target_type='continuous',
                        target_dim=args.target_dim, source_dim=args.source_dim,
                        regularization=ot_args.regularization, alpha=ot_args.alpha,
                        device=device)
    ot_planner.load_model(ot_args.load_name)

    mapping = Mapping(ot_planner, dim=args.target_dim)
    mapping.load_model(map_args.load_name)

    all_batch_src_data, all_batch_dst_data = [], []
    for i in range(10):
        print(i)
        src_idx_range = src_split[str(i)]
        dst_idx_range = dst_split[str(i)]
        batch_src_data, batch_dst_data = [], []
        for j in range(src_idx_range[0], src_idx_range[1]):
            batch_src_data.append(src_dataset[j])
        for j in range(dst_idx_range[0], dst_idx_range[1]):
            batch_dst_data.append(dst_dataset[j])
        batch_src_data = torch.stack(batch_src_data, dim=0)
        batch_dst_data = torch.stack(batch_dst_data, dim=0)
        all_batch_src_data.append(batch_src_data)
        all_batch_dst_data.append(batch_dst_data)

        mapped_src_data = mapping(batch_src_data.to(device))

        mean_batch_src_data = torch.mean(batch_src_data, dim=0).detach().cpu().numpy().reshape(16, 16)
        mean_mapped_src_data = torch.mean(mapped_src_data, dim=0).detach().cpu().numpy().reshape(16, 16)
        mean_batch_dst_data = torch.mean(batch_dst_data, dim=0).detach().cpu().numpy().reshape(16, 16)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(mean_batch_src_data)
        plt.title('source mean')

        plt.subplot(1, 3, 2)
        plt.imshow(mean_mapped_src_data)
        plt.title('mapped source mean')

        plt.subplot(1, 3, 3)
        plt.imshow(mean_batch_dst_data)
        plt.title('target mean')

        plt.savefig('./figs/mapping_{}.png'.format(i))
        plt.clf()

    for i in range(10):
        for j in range(10):
            plan = ot_planner(all_batch_src_data[i].to(device), all_batch_dst_data[j].to(device))
            plt.subplot(10, 10, 10*i+j+1)
            sns_plot = sns.heatmap(plan.detach().cpu().numpy())
            plt.title("{0}-{1}".format(i, j))

    plt.savefig('./figs/ot_plan.png')


    print('Train finished!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--group_id', type=str, default='trial')
    parser.add_argument('--exp_id', type=str, default='trial')
    parser.add_argument('--mode', type=str, default='formal')
    args = parser.parse_args()
    cli_main(args.config, args.group_id, args.exp_id, args.mode)
