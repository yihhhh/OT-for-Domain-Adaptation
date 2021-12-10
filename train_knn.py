import os
import csv
import numpy as np
from argparse import ArgumentParser
import pickle

import torch
from sklearn.neighbors import KNeighborsClassifier

from models.mapping import Mapping
from models.ot_model import OTPlan
import utils


def cli_main(record=False, map=False, group_id='', exp_id='', config=''):
    print("Loading data ...")
    train_data, train_label = load_data(root_dir='./dataset', dataset_name='mnist', split='train')
    valid_data, valid_label = load_data(root_dir='./dataset', dataset_name='mnist', split='test')
    test_data, test_label = load_data(root_dir='./dataset', dataset_name='usps', split='test')

    if not map:
        print("Train KNN without mapping ...")
        knn_wo_mapping = KNeighborsClassifier(n_neighbors=1)
        knn_wo_mapping.fit(train_data, train_label)
        print("Validation ...")
        valid_pred = knn_wo_mapping.predict(valid_data)
        valid_acc = (valid_pred.reshape(-1) == valid_label.reshape(-1)).sum() / valid_pred.shape[0]
        print("Testing ...")
        test_pred = knn_wo_mapping.predict(test_data)
        test_acc = (test_pred.reshape(-1) == test_label.reshape(-1)).sum()/test_pred.shape[0]
        print("trained without mapping -- valid acc: {0}, test acc: {1}".format(valid_acc, test_acc))

    if map:
        args = utils.load_config(config)
        map_args = args.mapping

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ot_planner = OTPlan(source_type='continuous', target_type='continuous',
                            target_dim=256, source_dim=256,
                            regularization='l2', alpha=0.0006,
                            device=device)
        ot_planner.load_model(os.path.join('./exp/mapping_checkpoints', group_id, exp_id))
        mapping = Mapping(ot_plan=ot_planner, dim=16, hidden_size=map_args.hidden_size, device=device)
        mapping.load_model(os.path.join('./exp/mapping_checkpoints', group_id, exp_id))
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

        test_acc_digits = []
        for i in range(10):
            idx = np.where(test_label == i)[0]
            test_label_selected = test_label[idx]
            test_data_selected = test_data[idx, :]
            test_pred_selected = knn_with_mapping.predict(test_data_selected)
            test_acc_digits.append((test_pred_selected.reshape(-1) == test_label_selected.reshape(-1)).sum() / test_label_selected.shape[0])
        print(test_acc_digits)

    if record:
        file_path = "./figs/knn_exp.csv"
        if not os.path.isfile(file_path):
            with open(file_path, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = ['map', 'group_id', 'exp_id', 'valid_acc', 'test_acc'] + [str(i) for i in range(10)]
                csv_write.writerow(data_row)
        with open(file_path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [map, group_id, exp_id] + [valid_acc, test_acc] + test_acc_digits
            csv_write.writerow(data_row)

    print("Finished!")

def load_data(root_dir='./dataset', dataset_name='mnist', split='train'):
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--map', action='store_true', default=False)
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--group_id', type=str, default='')
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument('--config', type=str, default='')
    args = parser.parse_args()
    cli_main(record=args.record, map=args.map, group_id=args.group_id, exp_id=args.exp_id, config=args.config)