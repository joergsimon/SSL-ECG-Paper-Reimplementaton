import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.io as sciio

import src.datasets.dataset_utils as du
from src.datasets.torch_helper import ECGCachedWindowsDataset


class DreamerConstants:
    path_to_original_data: str = "dreamer/DREAMER.mat"
    path_to_cache: str = "/home/jsimon/Documents/HELENA/ssl-ecg/cache/dreamer/"
    window_size: int = 2560


@dataclass
class PersonData:
    ecg_baseline: List
    ecg_stimuli: List
    valance: List
    arousal: List
    dominance: List


def iterate_persons(ppl_array):
    for person_idx in range(len(ppl_array)):
        ppl_struct = ppl_array[person_idx]
        ecg_struct = ppl_struct['ECG'][0][0]
        films_baseline_array = ecg_struct['baseline'][0][0]
        films_stimuli_array = ecg_struct['stimuli'][0][0]
        #print(ppl_struct.dtype)
        valance = ppl_struct['ScoreValence'][0][0]
        arousal = ppl_struct['ScoreArousal'][0][0]
        dominance = ppl_struct['ScoreDominance'][0][0]
        assert len(films_baseline_array) == len(
            films_stimuli_array), "len of baseline and simuli array not the same. Not every film is recorded fully maybe?"
        data = PersonData(films_baseline_array, films_stimuli_array, valance, arousal, dominance)
        yield data


@dataclass
class FilmData:
    ecg_baseline: np.ndarray
    ecg_stimuli: np.ndarray
    valance: float
    arousal: float
    dominance: float


def iterate_films(p_data: PersonData):
    for film_idx in range(len(p_data.ecg_baseline)):
        d_ecg_baseline = p_data.ecg_baseline[film_idx][0]
        d_ecg_stimuli = p_data.ecg_stimuli[film_idx][0]
        valance = p_data.valance[film_idx][0]
        arousal = p_data.arousal[film_idx][0]
        dominance = p_data.dominance[film_idx][0]
        data = FilmData(d_ecg_baseline, d_ecg_stimuli, valance, arousal, dominance)
        yield data


def load_ecg_windows(basepath: str):

    def get_all_baseline_data(films_baseline_array):
        all_data = []
        for f_idx in range(len(films_baseline_array)):
            d_ecg_1 = films_baseline_array[f_idx][0][:, 0]
            d_ecg_2 = films_baseline_array[f_idx][0][:, 1]
            all_data += [d_ecg_1, d_ecg_2]
        all_data = np.concatenate(all_data)
        return all_data

    def make_windows(d_ecg_array):
        ws = DreamerConstants.window_size
        windows = []
        for d_ecg in d_ecg_array:
            max_len = du.get_max_len(d_ecg, ws)
            w = du.make_windows_list(d_ecg, max_len, ws)
            windows += w
        return windows

    def make_labels(f_data, num_windows):
        wl = [(f_data.valance, f_data.arousal)] * num_windows
        return wl

    data = sciio.loadmat(basepath + DreamerConstants.path_to_original_data)
    ppl_array = data['DREAMER'][0][0]['Data'][0]
    windows = []
    window_labels = []
    for p_data in iterate_persons(ppl_array):
        all_baseline_data = get_all_baseline_data(p_data.ecg_baseline)
        data_mean, data_std = du.get_mean_std(all_baseline_data)

        for f_data in iterate_films(p_data):
            d_ecg_data  = [du.normalize(f_data.ecg_baseline[:, 0], data_mean, data_std)]
            d_ecg_data += [du.normalize(f_data.ecg_baseline[:, 1], data_mean, data_std)]
            d_ecg_data += [du.normalize(f_data.ecg_stimuli[:, 0], data_mean, data_std)]
            d_ecg_data += [du.normalize(f_data.ecg_stimuli[:, 1], data_mean, data_std)]

            ws = make_windows(d_ecg_data)
            windows += ws

            wl = make_labels(f_data, len(ws))
            window_labels += wl

    return windows, window_labels


class ECGDreamerCachedWindowsDataset(ECGCachedWindowsDataset):

    def __init__(self, basepath: str):
        super(ECGDreamerCachedWindowsDataset, self).__init__(basepath, DreamerConstants.path_to_cache, load_ecg_windows)

    def get_item(self, idx):
        # else we assume it is a single index so:
        with open(f'{DreamerConstants.path_to_cache}window-{idx}.data.npy', 'rb') as f:
            sample = np.load(f)
        with open(f'{DreamerConstants.path_to_cache}window-{idx}.label.npy', 'rb') as f:
            labels = pickle.load(f)
        return sample, labels


def load_raw_data(basepath: str):
    data = sciio.loadmat(basepath + DreamerConstants.path_to_original_data)
    # for now even only load one:
    person_id = 0

    data = sciio.loadmat(basepath + DreamerConstants.path_to_original_data)
    ppl_array = data['DREAMER'][0][0]['Data'][0]
    for person_idx in range(len(ppl_array)):
        films_baseline_array = ppl_array[person_idx]['ECG'][0][0]['baseline'][0][0]
        films_stimuli_array = ppl_array[person_idx]['ECG'][0][0]['baseline'][0][0]
        assert len(films_baseline_array) == len(
            films_stimuli_array), "len of baseline and simuli array not the same. Not every film is recorded fully maybe?"
        for film_idx in range(len(films_baseline_array)):
            d_ecg_baseline = films_baseline_array[film_idx][0]
            d_ecg_stimuli = films_stimuli_array[film_idx][0]
            print(f'p{person_idx} f{film_idx}: baseline len: {len(d_ecg_baseline)}, stimuli len: {len(d_ecg_stimuli)}')
    print(data['DREAMER'][0][0]['Data'][0][0].dtype)

    # print(data.keys())
    # print(data['DREAMER'].shape)
    # print(data['DREAMER'][0].shape)
    # print(data['DREAMER'][0][0].shape)
    # print(len(data['DREAMER'][0][0]))
    # print(data['DREAMER'][0][0].dtype)
    # print(len(data['DREAMER'][0][0]['Data']))
    # print(data['DREAMER'][0][0]['Data'][0].dtype)
    # print('ppl', len(data['DREAMER'][0][0]['Data'][0]))
    # print(data['DREAMER'][0][0]['Data'][0][0].dtype)
    # print(len(data['DREAMER'][0][0]['Data'][0][0]['ECG'][0][0]))
    # print(data['DREAMER'][0][0]['Data'][0][0]['ECG'][0][0].dtype)
    # print(len(data['DREAMER'][0][0]['Data'][0][0]['ECG'][0][0]['baseline']))
    # print('films', len(data['DREAMER'][0][0]['Data'][0][0]['ECG'][0][0]['baseline'][0][0]))
    # print(len(data['DREAMER'][0][0]['Data'][0][0]['ECG'][0][0]['baseline'][0][0][0][0]))
