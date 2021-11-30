import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir="./dataset", dataset_name='mnist'):
        self.root = root_dir
        assert dataset_name in ['mnist', 'usps']
        if dataset_name == 'mnist':
            file_name = 'mnist_dspd'
        elif dataset_name == 'usps':
            file_name = 'usps'

        all_data = []
        for i in range(3):
            with open(os.path.join(self.root, '{0}_{1}.pkl'.format(file_name, i)), 'rb') as f:
                data = pickle.load(f)
                for d in data:
                    all_data.append(d.reshape(-1).astype(float))

        self.all_data = all_data

    def __getitem__(self, item):
        data = self.all_data[item]
        data = torch.FloatTensor(data)
        return data

    def __len__(self):
        return len(self.all_data)

