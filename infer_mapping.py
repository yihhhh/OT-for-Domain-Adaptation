import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torch.optim import Adam

import utils
from dataset import MappingDataset
from models.ot_model import OTPlan
from models.mapping import Mapping

def cli_main(config_file='config', group_id='group', exp_id='exp', mode='formal'):
    print("Reading configurations ...")
    # utils.wandb_init("11785-project", group_id, exp_id)
    args = utils.load_config(os.path.join('./configs', '{}.yml'.format(config_file)))
    ot_args = args.ot_plan
    map_args = args.mapping
    save_dir = os.path.join('./mapping_checkpoints', group_id, exp_id)
    print(group_id, exp_id)

    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data ...")
    src_dataset = MappingDataset('./dataset/mapping', 'mnist')
    dst_dataset = MappingDataset('./dataset/mapping', 'usps')
    src_split = np.load('./dataset/label_split_mnist.npy', allow_pickle=True).item()
    dst_split = np.load('./dataset/label_split_usps.npy', allow_pickle=True).item()

    ot_planner = OTPlan(source_type='continuous', target_type='continuous',
                        target_dim=args.target_dim, source_dim=args.source_dim,
                        regularization=ot_args.regularization, alpha=ot_args.alpha,
                        device=device)
    ot_planner.load_model(save_dir)

    mapping = Mapping(ot_planner, dim=args.target_dim, device=device)
    mapping.load_model(save_dir)

    compute_sim = torch.nn.CosineSimilarity(dim=1)
    all_batch_src_data, all_batch_dst_data = [], []
    all_mapped_sim, all_sim = [], []
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

        mean_batch_src_data = torch.mean(batch_src_data, dim=0).detach().cpu().numpy().reshape(1, -1)
        mean_mapped_src_data = torch.mean(mapped_src_data, dim=0).detach().cpu().numpy().reshape(1, -1)
        mean_batch_dst_data = torch.mean(batch_dst_data, dim=0).detach().cpu().numpy().reshape(1, -1)

        sim = compute_sim(torch.Tensor(mean_batch_src_data), torch.Tensor(mean_batch_dst_data)).numpy()[0]
        mapped_sim = compute_sim(torch.Tensor(mean_mapped_src_data), torch.Tensor(mean_batch_dst_data)).numpy()[0]
        all_sim.append(sim)
        all_mapped_sim.append(mapped_sim)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(mean_batch_src_data.reshape(16, 16))
        plt.title('source mean\ncos similarity={:.2f}'.format(sim))

        plt.subplot(1, 3, 2)
        plt.imshow(mean_mapped_src_data.reshape(16, 16))
        plt.title('mapped source mean\ncos similarity={:.2f}'.format(mapped_sim))

        plt.subplot(1, 3, 3)
        plt.imshow(mean_batch_dst_data.reshape(16, 16))
        plt.title('target mean')

        plt.savefig('./figs/mapping_{}.png'.format(i))
        plt.clf()

    path = "./figs/cos_sim.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = [group_id, exp_id] + all_mapped_sim
        csv_write.writerow(data_row)

    # for i in range(10):
    #     for j in range(10):
    #         plan = ot_planner(all_batch_src_data[i].to(device), all_batch_dst_data[j].to(device))
    #         plan = plan.detach().cpu().numpy()
    #         np.save('./figs/ot_plan_{0}-{1}.npy'.format(i, j), plan)
        #     plt.subplot(2, 5, j+1)
        #     sns_plot = sns.heatmap(plan)
        #     plt.title("{0}-{1}".format(i, j))
        # plt.savefig('./figs/ot_plan_{0}-{1}.png'.format(i, j))

    print('Train finished!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--group_id', type=str, default='trial')
    parser.add_argument('--exp_id', type=str, default='trial')
    parser.add_argument('--mode', type=str, default='formal')
    args = parser.parse_args()
    cli_main(args.config, args.group_id, args.exp_id, args.mode)
