import torch
import numpy as np
import random
import tqdm
from src.constants import Constants as c
import matplotlib.pyplot as plt


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


def pbar(iterable=None, **kwargs):
    if c.use_ray:
        return iterable
    else:
        return tqdm.tqdm(iterable=iterable, **kwargs)


def print_ray_overview(result, prefix):
    dfs = result.trial_dataframes
    if len(dfs) > 0:
        dfs_list = list(dfs.values())
        first_data_frame = dfs_list[0]
        if 'accuracy' in first_data_frame.columns:
            plt.figure()
            ax = None  # This plots everything on the same plot
            for d in dfs_list:
                if 'accuracy' in d.columns:
                    ax = d.accuracy.plot(ax=ax, legend=False)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            plt.savefig(f'overview-accuracy-{prefix}.png')
            plt.show()
        if 'loss' in first_data_frame.columns:
            plt.figure()
            ax = None  # This plots everything on the same plot
            for d in dfs_list:
                if 'loss' in d.columns:
                    ax = d.loss.plot(ax=ax, legend=False)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("loss")
            plt.savefig(f'overview-loss-{prefix}.png')
            plt.show()