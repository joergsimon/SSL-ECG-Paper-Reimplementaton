import glob
import os
import os.path
import pickle
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tqdm

import src.datasets.dataset_utils as du
from src.constants import Constants as c
from src.datasets.torch_helper import ECGCachedWindowsDataset


class SwellConstants:
    glob_to_pickled_data: str = "swell/0 - Raw data/D - Physiology - raw data/Mobi signals text/*.txt"
    labels_file = 'swell/Behavioral-features - per minute.xlsx'
    path_to_cache: str = c.cache_base_path + "swell/"
    window_size: int = 2560


def iterate_swell_persons(basepath: str):
    files = glob.glob(basepath + SwellConstants.glob_to_pickled_data)
    person_to_files = defaultdict(list)
    fnames = [os.path.basename(f) for f in files]
    for i, f in zip(fnames, files):
        p_name = i[:i.find('_')].upper()
        person_to_files[p_name].append(f)
    for p in tqdm.tqdm(person_to_files.keys()):
        yield p, person_to_files[p]

def load_ecg_windows(basepath: str):
    def make_windows(d_ecg_array):
        ws = SwellConstants.window_size
        windows = []
        for d_ecg in d_ecg_array:
            max_len = du.get_max_len(d_ecg, ws)
            w = du.make_windows_list(d_ecg, max_len, ws)
            windows += w
        return windows

    label = pd.ExcelFile(basepath + SwellConstants.labels_file, engine='openpyxl')
    label_sheet_names = label.sheet_names
    labels = label.parse(label_sheet_names[0]) # we only need sheet one

    swell_labels = labels.drop_duplicates(subset=['PP', 'Blok'], keep='last')
    swell_labels = swell_labels.reset_index(drop=True)

    windows = []
    window_labels = []
    for p_id, files in iterate_swell_persons(basepath):
        all_ecg_d = []
        for f in files:
            signal = np.loadtxt(f)
            new_len = int((len(signal) / 2048) * 256)
            signal = cv2.resize(signal, (1, new_len), interpolation=cv2.INTER_LINEAR).reshape((new_len,))
            all_ecg_d.append(signal)
        all_ecg_np = np.concatenate(all_ecg_d)
        data_mean, data_std = du.get_mean_std(all_ecg_np)

        for s_idx, signal in enumerate(all_ecg_d):
            ecg = du.normalize(signal, data_mean, data_std)
            w = make_windows([ecg])
            windows += w

            label_set = swell_labels[(swell_labels['PP'] == p_id) & (swell_labels['Blok'] == s_idx+1)]
            label_set = label_set[
                ['Valence_rc', 'Arousal_rc', 'Dominance', 'Stress', 'MentalEffort', 'MentalDemand', 'PhysicalDemand',
                 'TemporalDemand', 'Effort', 'Performance_rc', 'Frustration']]
            label_set = np.asarray(label_set)
            wl = [label_set]*len(w)
            window_labels += wl
    return windows, window_labels


class ECGSwellCachedWindowsDataset(ECGCachedWindowsDataset):

    def __init__(self, basepath: str):
        super(ECGSwellCachedWindowsDataset, self).__init__(basepath, SwellConstants.path_to_cache, load_ecg_windows)

    def get_item(self, idx):
        # else we assume it is a single index so:
        with open(f'{SwellConstants.path_to_cache}window-{idx}.data.npy', 'rb') as f:
            sample = np.load(f)
        with open(f'{SwellConstants.path_to_cache}window-{idx}.label.npy', 'rb') as f:
            labels = pickle.load(f)
        return sample, labels
