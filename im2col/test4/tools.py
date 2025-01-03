import numpy as np


def split_data(data, label, split_size):
    data_num = int(data.shape[0])
    train_num = int(data_num * split_size)
    train_data = data[:train_num - 1]
    val_data = data[train_num:]
    train_label = label[:train_num - 1]
    val_label = label[train_num:]

    return train_data, val_data, train_label, val_label


# 归一化
def normalize(data):
    data = data / 255

    return data


# one-hot
def one_hot(num, label):
    label = np.eye(num)[label]

    return label
