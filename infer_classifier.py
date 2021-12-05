import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from models.cnn import CNN, FeatureLayer
from dataset import ClassificationDataset
from models.mapping import Mapping


def cli_main(config_file='config'):

    print("Reading configurations ...")
    args = utils.load_config(config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data ...")
    test_set = ClassificationDataset(root_dir='./dataset', dataset_name='usps', split='test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print("Initializing NN ...")
    feature_nn = FeatureLayer().to(device)
    model = CNN(feature_nn).to(device)

    print("Loading checkpoint file: {} ...".format(args.ckpt_pth))
    utils.load_ckpt(model, args.ckpt_pth)
    model.eval()

    mapping_func = None
    if args.use_map:
        mapping_func = Mapping(ot_plan=None, dim=16, device=device)
        mapping_func.load_model(args.mapping_ckpt)

    loss_fn = nn.NLLLoss()

    print("Testing ...")
    infer_loss, infer_acc = 0, 0
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device).unsqueeze(1), y.to(device)
        data_shape = x.shape

        if map and mapping_func is not None:
            x_src = x.reshape(data_shape[0], -1)
            x_dst = mapping_func(x_src.to(device)).detach()
            x = x_dst.reshape(data_shape)

        with torch.no_grad():
            output = model(x)
            loss = loss_fn(output, y.view(-1))
            infer_loss += loss.item() / y.shape[0]
            infer_acc += (output.argmax(1) == y.view(-1)).sum().item() / y.shape[0]

    print("testing acc: {0}, testing loss: {1}".format(infer_acc / len(test_loader), infer_loss / len(test_loader)))

    print("Finished!")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    args = parser.parse_args()
    cli_main(args.config)