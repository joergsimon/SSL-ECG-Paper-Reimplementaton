import torch
import numpy as np
import random
import tqdm
from src.constants import Constants as c
import matplotlib.pyplot as plt
from collections import OrderedDict


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

def save_load_state_dict(model, state_dict):
    """
    If DataParallel is used, this wrapps your model, and changes your state dict. In case you load the model without
    the data parallel, or the other way around, save the model without data parallel and load it with, the load fails.
    Also loading a model saved on pgu on cpu fails. This method encapsulates all these problems...

    :param model: the nn.Module model to load
    :param state_dict: the dictionary with the weights
    :return: an model initialized with the values of the state dict
    """
    try:
        model.load_state_dict(state_dict)
        return model
    except RuntimeError as e:
        print('loading failed, probably different wrapping on save and load.\nOriginal Error:')
        print(e)
        print('try to resolve...')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.starts_with('module'):
                name = k[7:]  # remove `module.`
            else:
                name = f'module.{k}'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model
