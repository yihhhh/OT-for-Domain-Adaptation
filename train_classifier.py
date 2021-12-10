import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from models.cnn import CNN, FeatureLayer
from dataset import ClassificationDataset
from models.mapping import Mapping


def cli_main(config_file='config', group_id='group', exp_id='exp'):

    print("Reading configurations ...")
    args = utils.load_config(config_file)
    print(exp_id)
    save_dir = './classifier_checkpoints/{0}/{1}'.format(group_id, exp_id)
    if args.save_ckpt:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data ...")
    train_loader, valid_loader = load_data(args)

    print("Initializing NN ...")
    feature_nn = FeatureLayer().to(device)

    if args.pretrain:
        utils.load_ckpt(feature_nn, args.pretrain_model)
        feature_nn.eval()

    model = CNN(feature_nn).to(device)
    if args.load_ckpt:
        print("Loading checkpoint file: {} ...".format(args.ckpt_pth))
        utils.load_ckpt(model, args.ckpt_pth)
        model.eval()

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=0, eps=1e-08, verbose=False)

    print("Training ...")
    best_valid_loss = np.inf
    best_valid_acc = 0
    train_loss_lst, train_acc_lst, valid_loss_lst, valid_acc_lst = [], [], [], []
    for i in range(args.n_epoch):

        trainset_loss, trainset_acc = train(model, loss_fn, optimizer, train_loader, device)
        validset_loss, validset_acc = inference(model, loss_fn, valid_loader, device)

        lr_scheduler.step(-validset_acc)

        train_loss_lst.append(trainset_loss)
        train_acc_lst.append(trainset_acc)
        valid_loss_lst.append(validset_loss)
        valid_acc_lst.append(validset_acc)
        if best_valid_acc < validset_acc:
            best_valid_loss = validset_loss
            best_valid_acc = validset_acc
            if args.save_ckpt:
                save_file = os.path.join(save_dir, 'valid_acc=' + '{}.pt'.format(round(validset_acc, 6)))
                utils.save_ckpt(model, save_file)

        print("Epoch {0}/{1} -- lr : {5}\ntrain loss: {2}, valid loss: {3}, best valid loss: {4}".format(i+1, args.n_epoch, trainset_loss, validset_loss, best_valid_loss, optimizer.param_groups[0]['lr']))
        print("train acc: {2}, valid acc: {3}, best valid acc: {4}".format(i+1, args.n_epoch, trainset_acc, validset_acc, best_valid_acc))

    print("Finished!")


def load_data(args):
    train_set = ClassificationDataset(root_dir='./dataset', dataset_name='usps', split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set = ClassificationDataset(root_dir='./dataset', dataset_name='usps', split='test')
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader


def train(model, loss_fn, optimizer, training_data, device):
    model.train()

    train_loss = 0
    train_acc = 0

    with tqdm(enumerate(training_data)) as t:
        for i, (x, y) in t:
            x, y = x.to(device).unsqueeze(1), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y.view(-1))
            loss.backward()
            optimizer.step()

            cur_loss = loss.item() / y.shape[0]
            train_loss += cur_loss

            cur_acc = (output.argmax(1) == y.view(-1)).sum().item() / y.shape[0]
            train_acc += cur_acc

            t.set_description("train")
            t.set_postfix(loss=cur_loss, acc=cur_acc, lr=optimizer.param_groups[0]['lr'])

    return train_loss / len(training_data), train_acc / len(training_data)


def inference(model, loss_fn, testing_data, device):
    model.eval()

    infer_loss = 0
    infer_acc = 0

    for i, (x, y) in enumerate(testing_data):
        x, y = x.to(device).unsqueeze(1), y.to(device)
        with torch.no_grad():
            output = model(x)
            loss = loss_fn(output, y.view(-1))
            infer_loss += loss.item()/y.shape[0]
            infer_acc += (output.argmax(1) == y.view(-1)).sum().item()/y.shape[0]

    return infer_loss / len(testing_data), infer_acc / len(testing_data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--group_id', type=str, default='classifier')
    parser.add_argument('--exp_id', type=str, default='trial')
    args = parser.parse_args()
    cli_main(args.config, args.group_id, args.exp_id)