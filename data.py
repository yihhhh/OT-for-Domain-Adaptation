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

        data_split = {}
        all_data = []
        for i in range(10):
            with open(os.path.join(self.root, '{0}_{1}.pkl'.format(file_name, i)), 'rb') as f:
                data = pickle.load(f)
                # start_idx = len(all_data)
                for d in data:
                    all_data.append(d.reshape(-1).astype(float))
                # end_idx = len(all_data)
                # data_split[str(i)] = [start_idx, end_idx]

        # np.save(os.path.join(self.root, 'label_split_{}.npy'.format(dataset_name)), data_split)

        self.all_data = all_data

    def __getitem__(self, item):
        data = self.all_data[item]
        data = torch.FloatTensor(data)
        return data

    def __len__(self):
        return len(self.all_data)

if __name__=="__main__":
    dset = ImageDataset(root_dir="./dataset", dataset_name='mnist')
    dset = ImageDataset(root_dir="./dataset", dataset_name='usps')