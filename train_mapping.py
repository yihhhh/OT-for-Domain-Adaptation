import os
import csv
import pickle
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.utils.data import RandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from sklearn.neighbors import KNeighborsClassifier

import utils
from dataset import MappingDataset
from models.ot_model import OTPlan
from models.mapping import Mapping

def cli_main(config='config', group_id='group', exp_id='exp'):
    print("Reading configurations ...")
    args = utils.load_config(config)
    ot_args = args.ot_plan
    map_args = args.mapping
    save_dir = os.path.join('./mapping_checkpoints', group_id, exp_id)
    if map_args.train_model or ot_args.train_model:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    print('save dir: ', save_dir)
    print(group_id, exp_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        train_ot_plan(ot_args, ot_src_sampler, ot_dst_sampler, ot_planner, ot_optimizer, ot_lr_scheduler, device=device)
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

    print("Loading data ...")
    train_data, train_label = read_data(root_dir='./dataset', dataset_name='mnist', split='train')
    valid_data, valid_label = read_data(root_dir='./dataset', dataset_name='mnist', split='test')
    test_data, test_label = read_data(root_dir='./dataset', dataset_name='usps', split='test')

    mapped_train_data = mapping(torch.FloatTensor(train_data).to(device))
    mapped_train_data = mapped_train_data.detach().cpu().numpy()

    mapped_valid_data = mapping(torch.FloatTensor(valid_data).to(device))
    mapped_valid_data = mapped_valid_data.detach().cpu().numpy()

    print("Train KNN with mapping ...")
    knn_with_mapping = KNeighborsClassifier(n_neighbors=1)
    knn_with_mapping.fit(mapped_train_data, train_label)
    print("Validation ...")
    valid_pred = knn_with_mapping.predict(mapped_valid_data)
    valid_acc = (valid_pred.reshape(-1) == valid_label.reshape(-1)).sum() / valid_pred.shape[0]
    print("Testing ...")
    test_pred = knn_with_mapping.predict(test_data)
    test_acc = (test_pred.reshape(-1) == test_label.reshape(-1)).sum() / test_pred.shape[0]
    print("trained with mapping -- valid acc: {0}, test acc: {1}".format(valid_acc, test_acc))

    file_path = "./figs/knn_exp.csv"
    if not os.path.isfile(file_path):
        with open(file_path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = ['map', 'group_id', 'exp_id', 'alpha', 'valid_acc', 'test_acc']
            csv_write.writerow(data_row)
    with open(file_path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = [map, group_id, exp_id] + [ot_args.alpha, valid_acc, test_acc]
        csv_write.writerow(data_row)


def load_data():
    src_dataset = MappingDataset('./dataset/mapping', 'mnist')
    dst_dataset = MappingDataset('./dataset/mapping', 'usps')
    print("src dataset len: ", len(src_dataset), " dst dataset len: ", len(dst_dataset))
    src_sampler = RandomSampler(src_dataset, replacement=True)
    dst_sampler = RandomSampler(dst_dataset, replacement=True)
    return src_sampler, dst_sampler

def read_data(root_dir='./dataset', dataset_name='mnist', split='train'):
    assert dataset_name in ['mnist', 'usps']
    if dataset_name == 'mnist':
        mean_value = 33.318421449829934
        std_value = 78.56748998339798
    elif dataset_name == 'usps':
        mean_value = 62.95362165255109
        std_value = 76.21333201287572

    # load data and label
    file_data = os.path.join(root_dir, dataset_name, '{}_data.pkl'.format(split))
    with open(file_data, 'rb') as f:
        data = pickle.load(f)
        data = data.reshape((data.shape[0], -1))
        data = (data - mean_value) / std_value

    file_label = os.path.join(root_dir, dataset_name, '{}_label.pkl'.format(split))
    with open(file_label, 'rb') as f:
        labels = pickle.load(f)

    return data, labels

def train_ot_plan(args, source_sampler, target_sampler, ot_planner, optimizer, scheduler, device):
    print("Learning OT Plan ...")

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

            scheduler.step()


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

            t.set_description("train Mapping")
            t.set_postfix(loss=loss_log, lr=optimizer.param_groups[0]['lr'])

            scheduler.step()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--group_id', type=str, default='trial')
    parser.add_argument('--exp_id', type=str, default='trial')
    args = parser.parse_args()
    cli_main(config=args.config, group_id=args.group_id, exp_id=args.exp_id)
