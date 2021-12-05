import os
import pickle
import numpy as np
from scipy import interpolate

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class NumericalDataset(Dataset):
    def __init__(self, data, label):
        self.all_data = data
        self.all_label = label

    def __getitem__(self, item):
        data = self.all_data[item, :]
        data = torch.FloatTensor(data)

        label = self.all_label[item]
        label = torch.LongTensor([label])
        return data, label

    def __len__(self):
        return self.all_data.shape[0]

class MappingDataset(Dataset):
    def __init__(self, root_dir="./dataset", dataset_name='mnist'):
        self.root = root_dir
        assert dataset_name in ['mnist', 'usps']
        if dataset_name == 'mnist':
            file_name = 'mnist_dspd'
            mean_value = 33.318421449829934
            std_value = 78.56748998339798
        elif dataset_name == 'usps':
            file_name = 'usps'
            mean_value = 62.95362165255109
            std_value = 76.21333201287572

        all_data = []
        all_label = []
        for i in range(10):
            with open(os.path.join(self.root, '{0}_{1}.pkl'.format(file_name, i)), 'rb') as f:
                data = pickle.load(f)
                for d in data:
                    d = d.reshape(-1).astype(float)
                    d = (d - mean_value)/std_value
                    all_data.append(d)
                    all_label.append(i)
        self.all_data = all_data
        self.all_label = all_label

    def __getitem__(self, item):
        data = self.all_data[item]
        data = torch.FloatTensor(data)

        label = self.all_label[item]
        label = torch.LongTensor([label])
        return data, label

    def __len__(self):
        return len(self.all_data)


class ClassificationDataset(Dataset):
    def __init__(self, root_dir="./dataset", dataset_name='mnist', split='train'):
        self.root = root_dir
        assert dataset_name in ['mnist', 'usps']
        if dataset_name == 'mnist':
            mean_value = 33.318421449829934
            std_value = 78.56748998339798
        elif dataset_name == 'usps':
            mean_value = 62.95362165255109
            std_value = 76.21333201287572

        # load data and label
        file_data = os.path.join(self.root, dataset_name, '{}_data.pkl'.format(split))
        with open(file_data, 'rb') as f:
            data = pickle.load(f)
            data = (data - mean_value) / std_value

        file_label = os.path.join(self.root, dataset_name, '{}_label.pkl'.format(split))
        with open(file_label, 'rb') as f:
            labels = pickle.load(f)

        self.data = data
        self.labels = labels
        # print(data.shape, labels.shape)

    def __getitem__(self, item):
        data = torch.FloatTensor(self.data[item, :, :])
        label = torch.LongTensor([self.labels[item]])
        return data, label

    def __len__(self):
        return self.data.shape[0]

def process_data(dataset_np, src_size, tgt_size):
    data_processed = np.empty([dataset_np.shape[0], tgt_size, tgt_size])
    for i in range(dataset_np.shape[0]):
        sample = dataset_np[i, :, :]
        sample_itprtd = downsample(sample, src_size, tgt_size)
        data_processed[i, :, :] = sample_itprtd
    return data_processed

def downsample(sample, src_size, tgt_size):
    x_i_low = -5
    x_i_high = 5
    x1_src = np.linspace(x_i_low, x_i_high, num=src_size)
    x2_src = np.linspace(x_i_low, x_i_high, num=src_size)

    x1_tgt = np.linspace(x_i_low, x_i_high, num=tgt_size)
    x2_tgt = np.linspace(x_i_low, x_i_high, num=tgt_size)
    f_inter = interpolate.interp2d(x1_src, x2_src, sample, kind='cubic')
    sample_itprtd = f_inter(x1_tgt, x2_tgt)
    return sample_itprtd

def save_pkl(file_name, data):
    file_open = open(file_name, 'wb')
    pickle.dump(data, file_open)
    file_open.close()
    print("Saved ", file_name, " data shape: ", data.shape)

# if __name__=="__main__":
    # dset = MappingDataset(root_dir="./dataset", dataset_name='mnist')
    # dset = MappingDataset(root_dir="./dataset", dataset_name='usps')

    # # process train data
    # mnist_dataset = datasets.MNIST(root='./dataset/mnist', train=True,
    #                                transform=transforms, download=True)
    # mnist_np = process_data(mnist_dataset.data.numpy(), 28, 16)
    # save_pkl("./dataset/mnist/train_data.pkl", mnist_np)
    #
    # mnist_label_np = mnist_dataset.targets.numpy()
    # save_pkl("./dataset/mnist/train_label.pkl", mnist_label_np)
    #
    # usps_dataset = datasets.USPS(root='./dataset/usps', train=False,
    #                              transform=transforms, download=True)
    # usps_np = usps_dataset.data
    # save_pkl("./dataset/usps/train_data.pkl", usps_np)
    #
    # usps_label_np = np.asarray(usps_dataset.targets)
    # save_pkl("./dataset/usps/train_label.pkl", usps_label_np)
    #
    # # process test data
    # mnist_dataset = datasets.MNIST(root='./dataset/mnist', train=False,
    #                                transform=transforms, download=True)
    # mnist_np = process_data(mnist_dataset.data.numpy(), 28, 16)
    # save_pkl("./dataset/mnist/test_data.pkl", mnist_np)
    #
    # mnist_label_np = mnist_dataset.targets.numpy()
    # save_pkl("./dataset/mnist/test_label.pkl", mnist_label_np)
    #
    # usps_dataset = datasets.USPS(root='./dataset/usps', train=False,
    #                              transform=transforms, download=True)
    # usps_np = usps_dataset.data
    # save_pkl("./dataset/usps/test_data.pkl", usps_np)
    #
    # usps_label_np = np.asarray(usps_dataset.targets)
    # save_pkl("./dataset/usps/test_label.pkl", usps_label_np)
