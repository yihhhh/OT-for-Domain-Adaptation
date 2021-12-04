import numpy as np
import os
from argparse import ArgumentParser
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier

import utils
from dataset import ClassificationDataset
from models.mapping import Mapping


def cli_main(map=False):
    print("Loading data ...")
    train_data, train_label = load_data(root_dir='./dataset', dataset_name='mnist', split='train')

    valid_data, valid_label = load_data(root_dir='./dataset', dataset_name='mnist', split='test')

    test_data, test_label = load_data(root_dir='./dataset', dataset_name='usps', split='test')

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapping = Mapping(ot_plan=None, dim=16, device=device)
    mapping.load_model('./mapping_checkpoints/linear+normalize/a=0.000001')
    # mapping.load_model('./mapping_checkpoints/entropy/a=5')
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
    args = parser.parse_args()
    cli_main(map=False)