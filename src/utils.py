import torch
import numpy as np
import random


def y_to_torch(y_list, shape=None):
    y_np = np.array(y_list)
    if shape is not None:
        y_np = y_np.reshape(shape)
    y_torch = torch.from_numpy(y_np).float()
    return y_torch


def random_splits(indices, test_size, valid_size):
    n = len(indices)
    np.random.shuffle(indices)
    split = int(np.floor(test_size * n))
    train_and_valid_idx, test_idx = indices[split:], indices[:split]
    n_tv = len(train_and_valid_idx)
    split = int(np.floor(valid_size * n_tv))
    train_idx, valid_idx = train_and_valid_idx[split:], train_and_valid_idx[:split]

    return train_idx, valid_idx, test_idx


def shuffle_lists(*lists):
    l = list(zip(*lists))
    random.shuffle(l)
    return zip(*l)


def assign(lt, ls):
    if lt is None:
        lt = ls
    else:
        lt += ls
    return lt
