import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sciio

import src.datasets.dataset_utils as du
from src.datasets.torch_helper import ECGCachedWindowsDataset
from src.constants import Constants as c


class AmigosConstants:
    glob_to_original_data: str = "amigos/data_original/**/*.mat"
    glob_to_preprocessed_data: str = "amigos/data-preprocessed/**/*.mat"
    path_to_cache: str = c.cache_base_path + "amigos/"
    path_to_selfassement: str = "amigos/SelfAsessment.xlsx"
    window_size: int = 2560


def iterate_amigos(basepath: str):
    amigos_raw_files = glob.glob(basepath + AmigosConstants.glob_to_original_data)
    for idx, file in enumerate(amigos_raw_files):
        person_id = int(file[-6:-4])
        yield idx, file, person_id


def iterate_data(ecg_d):
    for d_idx in range(len(ecg_d) - 4):  # for now skip the last 4 long videos
        d = ecg_d[d_idx]
        d_ts, d_left_ecg, d_right_ecg, d_accel_x, d_accel_y, d_accel_z = \
            d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5]
        yield d_idx, d_ts, d_left_ecg, d_right_ecg, d_accel_x, d_accel_y, d_accel_z


def load_ecg(file):
    data = sciio.loadmat(file)
    ecg_d = data['ECG_DATA'][0]
    return ecg_d


def load_ecg_windows(basepath: str):

    def make_windows(d_left_ecg, d_right_ecg):
        ws = AmigosConstants.window_size
        max_len = du.get_max_len(d_left_ecg, ws)
        w1 = du.make_windows_list(d_left_ecg, max_len, ws)
        w2 = du.make_windows_list(d_right_ecg, max_len, ws)
        return w1, w2

    def make_labels(idx, assesment_pers, num_windows):
        video_labels = assesment_pers[assesment_pers['Rep_Index'] == idx + 1]
        valance = video_labels['valence'].values[0]
        arousal = video_labels['arousal'].values[0]
        wl = [(valance, arousal)] * num_windows
        return wl

    def collect_all_ecg_data(ecg_d):
        data = []
        for d_idx, _, d_left_ecg, d_right_ecg, _, _, _ in iterate_data(ecg_d):
            data += [d_left_ecg, d_right_ecg]
        all_ecg = np.concatenate(data)
        return all_ecg

    windows = []
    window_labels = []
    selfassesment = pd.read_excel(basepath + AmigosConstants.path_to_selfassement, engine='openpyxl')
    print(selfassesment)
    for idx, file, person_id in iterate_amigos(basepath):
        print(file)
        assesment_pers = selfassesment[selfassesment['UserID'] == person_id]
        ecg_d = load_ecg(file)

        all_ecg = collect_all_ecg_data(ecg_d)
        data_mean, data_std = du.get_mean_std(all_ecg)

        for d_idx, _, d_left_ecg, d_right_ecg, _, _, _ in iterate_data(ecg_d):
            d_left_ecg = du.normalize(d_left_ecg, data_mean, data_std)
            d_right_ecg = du.normalize(d_right_ecg, data_mean, data_std)

            w1, w2 = make_windows(d_left_ecg, d_right_ecg)
            wl = make_labels(d_idx, assesment_pers, len(w1)+len(w2))
            windows += w1
            windows += w2
            window_labels += wl

    return windows, window_labels


class ECGAmigosCachedWindowsDataset(ECGCachedWindowsDataset):

    def __init__(self, basepath: str):
        super(ECGAmigosCachedWindowsDataset, self).__init__(basepath, AmigosConstants.path_to_cache, load_ecg_windows)

    @property
    def target_size(self):
        return 10

    def get_item(self, idx):
        # else we assume it is a single index so:
        with open(f'{AmigosConstants.path_to_cache}window-{idx}.data.npy', 'rb') as f:
            sample = np.load(f)
        with open(f'{AmigosConstants.path_to_cache}window-{idx}.label.npy', 'rb') as f:
            labels = pickle.load(f)
        return sample, labels


def load_raw_data(basepath: str):
    # for now Amigos only ^_^
    amigos_raw_files = glob.glob(basepath + AmigosConstants.glob_to_original_data)
    # for now even only load one:
    person_id = 2
    print(amigos_raw_files[person_id])
    data = sciio.loadmat(amigos_raw_files[person_id])
    print(data.keys())
    print(len(data['ECG_DATA']))
    print(data['ECG_DATA'][0].shape)
    print(data['ECG_DATA'][0][0].shape)

    ecg_d = data['ECG_DATA'][0]

    starts = []
    ecg_dictionary = {}
    for idx in range(len(ecg_d)):
        d = ecg_d[idx]
        d_ts = d[:, 0]
        starts.append(d_ts[0])
        ecg_dictionary[d_ts[0]] = d
    starts = sorted(starts)
    for s in starts:
        d = ecg_dictionary[s]
        d_ts, d_left_ecg, d_right_ecg, d_accel_x, d_accel_y, d_accel_z = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[
                                                                                                                      :,
                                                                                                                      5]
        print(f'start: {d_ts[0]}, len:{(d_ts[-1] - d_ts[0]) / 1000}s')
        plt.plot(d_ts, d_left_ecg)
        plt.plot(d_ts, d_right_ecg)
        plt.show()


def load_preprocessed_data(basepath: str):
    # for now Amigos only ^_^
    amigos_raw_files = glob.glob(basepath + AmigosConstants.glob_to_preprocessed_data)
    print(amigos_raw_files)
    # for now even only load one:
    person_id = 2
    print(amigos_raw_files[person_id])
    data = sciio.loadmat(amigos_raw_files[person_id])
    print(data.keys())
    print(len(data["joined_data"]))
    print(len(data["joined_data"][0]))
    print(len(data["labels_ext_annotation"][0]))

    all_d = data["joined_data"][0]
    segment_labels = data["labels_ext_annotation"][0]
    self_labels = data["labels_selfassessment"][0]

    def compute_segment_lines(seg, channel, r, d_ecg):
        seg_val = (seg[:, channel] * r)
        print('seg l val len', len(seg_val))
        seg_val = np.array([[v] * (20 * 128) for v in seg_val]).flatten()
        print('*> ', len(seg_val))
        print('->', len(d_ecg))
        seg_val = seg_val[:len(d_ecg)]
        return seg_val

    for idx in range(len(all_d)):
        d = all_d[idx]
        d_left_ecg, d_right_ecg = d[:, 14], d[:, 15]

        # draw the scale over the chart range:
        mi, ma = min(min(d_left_ecg), min(d_right_ecg)), max(max(d_left_ecg), max(d_right_ecg))
        r = ma - mi

        seg_l_val = compute_segment_lines(segment_labels[idx], 1, r, d_left_ecg)
        seg_l_ar = compute_segment_lines(segment_labels[idx], 2, r, d_left_ecg)

        self_val = ((np.array([self_labels[idx][0, 1]] * len(d_left_ecg)) / 5.) - 1.) * r
        self_ar = ((np.array([self_labels[idx][0, 0]] * len(d_left_ecg)) / 5.) - 1.) * r

        print(self_labels[idx][0, :2])

        plt.plot(d_left_ecg)
        plt.plot(d_right_ecg)
        plt.plot(seg_l_val)
        plt.plot(seg_l_ar)
        plt.plot(self_val)
        plt.plot(self_ar)

        plt.show()
