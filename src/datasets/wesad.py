import glob
import pickle

import cv2
import numpy as np

import src.datasets.dataset_utils as du

from src.datasets.torch_helper import ECGCachedWindowsDataset


class WesadConstants:
    glob_to_pickled_data: str = "wesad/WESAD/**/*.pkl"
    glob_to_data: str = "wesad/WESAD/**/*.*"
    usr_num_regex: str = "S([0-9]+)_\w*.*"
    ecg_file_regex: str = "S[0-9]+_respiban.txt"
    lbls_file_regex: str = "S[0-9]+_quest.csv"
    data_file_key: str = 'data file'
    label_file_key = 'labels file'
    path_to_cache: str = "/Users/joergsimon/Documents/phd/HELENA/ssl-ecg/cache/wesad/"
    window_size: int = 2560


def iterate_wesad(basepath: str):
    pikls = glob.glob(basepath + WesadConstants.glob_to_pickled_data)
    for pf in pikls:
        with open(pf, 'rb') as f:
            blob = pickle.load(f, encoding='latin1')
            ecg = blob['signal']['chest']['ECG']
            lbls = blob['label']
            yield blob, ecg, lbls


def iterate_clean_labeled_sections(ecg, lbls):
    l_value = lbls[0]
    l_index = 0
    for i in range(1, len(lbls)):
        i_val = lbls[i]
        if i_val != l_value:
            yield l_value, ecg[l_index:i]
            l_value = i_val
            l_index = i
    yield l_value, ecg[l_index:]


def downsample(ecg):
    sampled = cv2.resize(ecg.astype(np.float), (1, int((ecg.shape[0] / 700.) * 256.)),
                        interpolation=cv2.INTER_LINEAR)
    return sampled


def load_ecg_windows(basepath: str):
    def make_windows(d_ecg_array):
        ws = WesadConstants.window_size
        max_len = du.get_max_len(d_ecg_array, ws)
        windows = du.make_windows_list(d_ecg_array, max_len, ws)
        return windows

    windows = []
    window_labels = []
    for _, ecg, lbls in iterate_wesad(basepath):
        mean, std = du.get_mean_std(ecg)
        ecg = du.normalize(ecg, mean, std)
        for section_lbl, section_ecg in iterate_clean_labeled_sections(ecg, lbls):
            section_ecg = downsample(section_ecg)
            ws = make_windows(section_ecg)
            windows += ws
            window_labels += [section_lbl]*len(ws)

    return windows, window_labels


class ECGWesadCachedWindowsDataset(ECGCachedWindowsDataset):

    def __init__(self, basepath: str):
        super(ECGWesadCachedWindowsDataset, self).__init__(basepath, WesadConstants.path_to_cache, load_ecg_windows)

    def get_item(self, idx):
        # else we assume it is a single index so:
        with open(f'{WesadConstants.path_to_cache}window-{idx}.data.npy', 'rb') as f:
            sample = np.load(f)
        with open(f'{WesadConstants.path_to_cache}window-{idx}.label.npy', 'rb') as f:
            labels = pickle.load(f)
        return sample, labels
