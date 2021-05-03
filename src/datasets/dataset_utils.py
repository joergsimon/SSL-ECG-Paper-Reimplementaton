import numpy as np
import pickle
import os


def get_max_len(data, window_len):
    max_len = (len(data) // window_len) * window_len
    return max_len


def make_windows_list(data, max_len, window_len):
    d = data[:max_len]
    n_windows = len(data) // window_len
    #print(len(data), len(d), n_windows, max_len, window_len)
    mat = d.reshape((n_windows, window_len))
    w = list(mat)
    return w


def normalize(data, data_mean, data_std):
    data_scaled = (data - data_mean) / data_std
    return data_scaled


def get_mean_std(all_data):
    all_data = np.sort(all_data)
    data_mean = np.mean(all_data)
    data_std = np.std(all_data[np.int(0.025 * len(all_data)): np.int(0.975 * len(all_data))])
    return data_mean, data_std


def save_windows_to_cache(path_to_cache, windows, window_labels):
    print(f'saving {len(windows)}  and {len(window_labels)} to cache')
    for i, (w, l) in enumerate(zip(windows, window_labels)):
        with open(f'{path_to_cache}window-{i}.data.npy', 'wb') as f:
            np.save(f, w)
        with open(f'{path_to_cache}window-{i}.label.npy', 'wb') as f:
            pickle.dump(l, f)


def cache_is_empty(path_to_cache):
    is_empty = len(os.listdir(path_to_cache)) == 0
    return is_empty