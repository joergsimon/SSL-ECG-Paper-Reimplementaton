import pickle
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import ClassVar
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

import src.augmentations as aug
import src.datasets.amigos as amigos
import src.datasets.dataset_utils as du
import src.datasets.dreamer as dreamer
import src.datasets.wesad as wesad


@dataclass
class DataConstants:
    basepath: ClassVar[str] = "/home/jsimon/Desktop/knownew/600 Datasets/human-telemetry/other_datasets_joerg/"#"/Users/joergsimon/Documents/work/datasets_cache/"#"/Volumes/knownew/600 Datasets/human-telemetry/other_datasets_joerg/"#


class DataSets(Enum):
    AMIGOS = 0
    DREAMER = 1
    WESAD = 2


ds_to_constructor = {
    DataSets.AMIGOS: amigos.ECGAmigosCachedWindowsDataset,
    DataSets.DREAMER: dreamer.ECGDreamerCachedWindowsDataset,
    DataSets.WESAD: wesad.ECGWesadCachedWindowsDataset
}


class CombinedECGDatasets(Dataset):

    def __init__(self, dataset_types: List[DataSets], basepath: str):
        super(CombinedECGDatasets, self).__init__()
        dataset_array = []
        for ds_type in dataset_types:
            ds_obj = ds_to_constructor[ds_type](basepath)
            dataset_array.append(ds_obj)
        self.datasets = dataset_array

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    @property
    def ts_length(self):
        return self.datasets[0].ts_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            samples = [self[ii] for ii in range(*idx.indices(len(self)))]
            return samples

        if isinstance(idx, list):
            samples = [self[ii] for ii in idx]
            return samples

        offset = 0
        for ds in self.datasets:
            if offset + len(ds) > idx:
                #print(f'return from {ds} at {idx-offset}')
                return ds[idx-offset]
            offset += len(ds)
        raise OverflowError()


class AugmentationsPretextDataset(Dataset):

    STD_AUG = [
        (aug.AugmentationTypes.ADD_NOISE, partial(aug.add_noise, SNR=15)),
        (aug.AugmentationTypes.NEGATE, aug.negate),
        #(aug.AugmentationTypes.PERMUTATE, partial(aug.permuatate, n_sections=5)),
        (aug.AugmentationTypes.SCALE, partial(aug.scale, beta=2)),
        #(aug.AugmentationTypes.TEMP_INV, aug.temp_invert),
        #(aug.AugmentationTypes.TIME_WRAP, partial(aug.time_warp, n_sections=5, k=2))
    ]

    def __init__(self, dataset: Dataset, augmentations):
        super(AugmentationsPretextDataset, self).__init__()
        self.dataset = dataset
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataset) * len(self.augmentations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            samples = [self[ii] for ii in range(*idx.indices(len(self)))]
            return samples

        if isinstance(idx, list):
            samples = [self[ii] for ii in idx]
            return samples

        # else we assume it is a single index so:
        if idx > len(self):
            print("===== >____ Index out of Bounds _____< =======")

        return self.get_item(idx)

    def get_item(self, idx):
        real_idx = idx // len(self.augmentations)
        #print(f'fetching data for augmentation at {idx} (real idx: {real_idx}) for pretext')
        data = self.dataset[real_idx]
        sample, label = data
        samples = [torch.tensor(sample)]
        labels = [aug.AugmentationTypes.ORIGINAL]
        for l, a in self.augmentations:
            aug_sample = a(sample)
            # print(f'do augmentation {l}')
            samples.append(torch.tensor(aug_sample))
            labels.append(l)
        labels = [l.value for l in labels]  # convert to int
        # print(samples, labels)
        return samples, labels


class EmbeddingsDataset(Dataset):

    path_to_cache = '/Users/joergsimon/Documents/phd/HELENA/ssl-ecg/cache/'
    
    def __init__(self, embedding_network, base_dataset, should_cache, path_to_cache, chache_identifier, train_on_gpu: bool):
        super(EmbeddingsDataset, self).__init__()
        self.embedding_network = embedding_network
        self.base_dataset = base_dataset
        self.should_cache = should_cache
        self.total_cache_path = path_to_cache + chache_identifier + '/'
        self.train_on_gpu = train_on_gpu
        if train_on_gpu:
            self.embedding_network = embedding_network.cuda()
        if self.should_cache:
            if du.cache_is_empty(self.total_cache_path):
                du.create_path_if_needed(self.total_cache_path)
                for i in range(len(base_dataset)):
                    emb, label = self.embed_item(i)
                    self.save_window_to_cache(self.total_cache_path, emb, label, i)

    def save_window_to_cache(self, path_to_cache, window, window_label, identifier):
        with open(f'{path_to_cache}window-{identifier}.data.pt', 'wb') as f:
            torch.save(window, f)
        with open(f'{path_to_cache}window-{identifier}.label.npy', 'wb') as f:
            pickle.dump(window_label, f)

    def __len__(self):
        return len(self.base_dataset)

    @property
    def ts_length(self):
        return self.base_dataset.ts_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            samples = [self[ii] for ii in range(*idx.indices(len(self)))]
            return samples

        if isinstance(idx, list):
            samples = [self[ii] for ii in idx]
            return samples

        if self.should_cache:
            return self.get_cached_item(idx)
        else:
            emb, label = self.embed_item(idx)
            return emb, label

    def embed_item(self, idx):
        data, label = self.base_dataset[idx]
        data = data.reshape((1, 1, data.shape[0]))
        data = torch.from_numpy(data).float()
        if self.train_on_gpu:
            data, label = data.cuda(), label.cuda()
        emb = self.embedding_network(data)
        return emb, label

    def get_cached_item(self, idx):
        # else we assume it is a single index so:
        with open(f'{self.total_cache_path}window-{idx}.data.pt', 'rb') as f:
            sample = torch.load(f)
        with open(f'{self.total_cache_path}window-{idx}.label.npy', 'rb') as f:
            labels = pickle.load(f)
        return sample, labels
