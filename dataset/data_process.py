'''
The experiment of
Scalable Dual OT
("Optimal Coupling is all you need for good representations"

("Large" Scale dataset testing)

Process the mnist and usps dataset for testing

    Notice:
        For interpolation, the range of domain (x_i) is [-5, 5]
'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import pickle
import os
from os import path
import random


def process_data(dataset_np, label_np, name, interpred=True):
    for label_selected in range(10):
        print('Now label_selected src is', label_selected)
        data_to_save_temp = []
        for i in range(label_np.shape[0]):
            label = label_np[i]
            if label == label_selected:
                sample_temp = dataset_np[i, :, :]
                if interpred:
                    x_i_low = -5
                    x_i_high = 5
                    src_shape = 28
                    tgt_shape = 16
                    x1_src = np.linspace(x_i_low, x_i_high, num=src_shape)
                    x2_src = np.linspace(x_i_low, x_i_high, num=src_shape)

                    x1_tgt = np.linspace(x_i_low, x_i_high, num=tgt_shape)
                    x2_tgt = np.linspace(x_i_low, x_i_high, num=tgt_shape)
                    f_inter = interpolate.interp2d(x1_src, x2_src, sample_temp, kind='cubic')
                    sample_itprtd = f_inter(x1_tgt, x2_tgt)
                else:
                    sample_itprtd = sample_temp
                data_to_save_temp.append(sample_itprtd)

        print("len(data_to_save_temp) =", len(data_to_save_temp))
        file_name = "dataset" + "/" + name + str(label_selected) + '.pkl'
        file_open = open(file_name, 'wb')
        pickle.dump(data_to_save_temp, file_open)
        file_open.close()
        print("Saved ", file_name)
    return 1


def get_mixed_data(num, name, shuffle=True, cat_list=None, **kwargs):
    """
    :param num: number of sample in each pattern
    :param name: "mnist_dspd_" or "usps_"
    :param shuffle: True of False
    :return:
    """

    all_data_list = []

    if cat_list == None:
        cat_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in cat_list:
        address = "dataset/" + name + str(i) + ".pkl"
        file_read = open(address, 'rb')
        data_list = pickle.load(file_read)
        # print("data.shape =", data[0].shape)

        if shuffle:
            random.shuffle(data_list)

        all_data_list.extend(data_list[:num])

    return all_data_list


def get_mixed_data_pro(num_per_mix, name, cat_list=None, shuffle=True):

    all_data_list= []
    if cat_list == None:
        cat_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in cat_list:
        address = "dataset/" + name + str(i) + ".pkl"
        file_read = open(address, 'rb')
        data_list = pickle.load(file_read)
        # print("data.shape =", data[0].shape)

        if shuffle:
            random.shuffle(data_list)

        all_data_list.extend(data_list[:num_per_mix])
    return

if __name__ == "__main__":


    # Notice: Load mnist dataset
    mnist_dataset = datasets.MNIST(root='../data/mnist', train=True,
                                   transform=transforms, download=True)
    mnist_np = mnist_dataset.data.numpy()
    mnist_label_np = mnist_dataset.targets.numpy()
    print('mnist_np.shape =', mnist_np.shape)   # (60000, 28, 28)
    print('mnist_label_np.shape =', mnist_label_np.shape)

    # Notice: Load usps dataset
    usps_dataset = datasets.USPS(root='../data/usps', train=True,
                                 transform=transforms, download=True)
    usps_np = usps_dataset.data
    usps_label_np = np.asarray(usps_dataset.targets)
    print('usps_np.shape =', usps_np.shape)     # (7291, 16, 16)
    print('usps_label_np.shape =', usps_label_np.shape)


    # Notice: First check the label

    print("mnist_label_np =", mnist_label_np)

    process_data(dataset_np=mnist_np, label_np=mnist_label_np,
                       name="mnist_dspd_", interpred=True)

    process_data(dataset_np=usps_np, label_np=usps_label_np,
                 name="usps_", interpred=False)

    # Notice: Read the saved data and
    num_per_pattern = 100
    data_name = "mnist_dspd_"
    mnist_test_list = get_mixed_data(num=num_per_pattern, name=data_name, shuffle=True)

    mnist_test_np = np.asarray(mnist_test_list)
    print("mnist_test_np.shape =", mnist_test_np.shape)