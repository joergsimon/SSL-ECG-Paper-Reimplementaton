import torch
import os
from torch.utils.data import Dataset
import src.datasets.dataset_utils as du


class ECGCachedWindowsDataset(Dataset):

    def __init__(self, basepath: str, path_to_cache: str, load_ecg_windows):
        super(ECGCachedWindowsDataset, self).__init__()
        if du.cache_is_empty(path_to_cache):
            windows, window_labels = load_ecg_windows(basepath)
            du.save_windows_to_cache(path_to_cache, windows, window_labels)
            windows = None  # allow gc
        self.window_files = os.listdir(path_to_cache)

    def __len__(self):
        return len(self.window_files)//2

    @property
    def ts_length(self):
        return 2560

    @property
    def target_size(self):
        return -1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            samples = [self[ii] for ii in range(*idx.indices(len(self)))]
            return samples

        if isinstance(idx, list):
            samples = [self[ii] for ii in idx]
            return samples

        if idx > len(self):
            print("===== >____ Index out of Bounds _____< =======")
        return self.get_item(idx)

    def get_item(self, idx):
        print('return not implemented')
        raise NotImplementedError()