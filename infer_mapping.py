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

def cli_main(group_id='group', exp_id='exp'):
    print("Reading configurations ...")
    save_dir = os.path.join('./exp/mapping_checkpoints', group_id, exp_id)
    print(group_id, exp_id)

    fig_dir = './figs/{0}-{1}'.format(group_id, exp_id)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data ...")
    src_dataset = MappingDataset('./dataset/mapping', 'mnist')
    dst_dataset = MappingDataset('./dataset/mapping', 'usps')
    # src_split = np.load('./dataset/mapping/label_split_mnist.npy', allow_pickle=True).item()
    # dst_split = np.load('./dataset/mapping/label_split_usps.npy', allow_pickle=True).item()
    # print(src_split)
    # print(dst_split)

    ot_planner = OTPlan(source_type='continuous', target_type='continuous',
                        target_dim=256, source_dim=256,
                        regularization='l2', alpha=0.0006,
                        device=device)
    ot_planner.load_model(save_dir)

    mapping = Mapping(ot_planner, dim=256, hidden_size=[], device=device)
    mapping.load_model(save_dir)

    compute_sim = torch.nn.CosineSimilarity(dim=1)
    all_batch_src_data = torch.load("./dataset/mapping/all_batched_src_data.pt")
    all_batch_dst_data = torch.load("./dataset/mapping/all_batched_dst_data.pt")
    all_mapped_sim, all_sim = [], []
    for i in range(10):
        print(i)
        # src_idx_range = src_split[str(i)]
        # dst_idx_range = dst_split[str(i)]
        # batch_src_data, batch_dst_data = [], []
        # for j in range(src_idx_range[0], src_idx_range[1]):
        #     batch_src_data.append(src_dataset[j][0])
        # for j in range(dst_idx_range[0], dst_idx_range[1]):
        #     batch_dst_data.append(dst_dataset[j][0])
        # batch_src_data = torch.stack(batch_src_data, dim=0)
        # batch_dst_data = torch.stack(batch_dst_data, dim=0)
        # all_batch_src_data.append(batch_src_data)
        # all_batch_dst_data.append(batch_dst_data)
        batch_src_data = all_batch_src_data[i]
        batch_dst_data = all_batch_dst_data[i]

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
        plt.axis('off')
        # plt.title('source mean\ncos similarity={:.2f}'.format(sim))

        plt.subplot(1, 3, 2)
        plt.imshow(mean_mapped_src_data.reshape(16, 16))
        plt.axis('off')
        # plt.title('mapped source mean\ncos similarity={:.2f}'.format(mapped_sim))

        plt.subplot(1, 3, 3)
        plt.imshow(mean_batch_dst_data.reshape(16, 16))
        plt.axis('off')
        # plt.title('target mean')

        plt.tight_layout()
        plt.savefig('./figs/{0}-{1}/mapping_{2}.png'.format(group_id, exp_id, i))
        plt.clf()

    path = "./figs/cos_sim+.csv"
    if not os.path.isfile(path):
        with open(path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = ['group_id', 'exp_id'] + [str(i) for i in range(10)]
            csv_write.writerow(data_row)
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = [group_id, exp_id] + all_mapped_sim
        csv_write.writerow(data_row)

    # all_plan = []
    # for i in range(10):
    #     single_src_plan = []
    #     for j in range(10):
    #         plan = ot_planner(all_batch_src_data[i].to(device), all_batch_dst_data[j].to(device))
    #         single_src_plan.append(plan.detach().cpu().numpy())
    #     single_src_plan = np.concatenate(single_src_plan, axis=1)
    #     all_plan.append(single_src_plan)
    # all_plan_np = np.concatenate(all_plan, axis=0)
    # # np.save('./figs/{0}-{1}/ot_plan.npy'.format(group_id, exp_id), all_plan_np)
    # sns_plot = sns.heatmap(all_plan_np)
    # plt.xlabel('Dataset: USPS')
    # plt.ylabel('Dataset: MNIST')
    # plt.savefig('./figs/{0}-{1}/ot_plan.png'.format(group_id, exp_id))

    # torch.save(all_batch_src_data, "./dataset/mapping/all_batched_src_data.pt")
    # torch.save(all_batch_dst_data, "./dataset/mapping/all_batched_dst_data.pt")
    print('Finished!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--group_id', type=str, default='trial')
    parser.add_argument('--exp_id', type=str, default='trial')
    args = parser.parse_args()
    cli_main(args.group_id, args.exp_id)
