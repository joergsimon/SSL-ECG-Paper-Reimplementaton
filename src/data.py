from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import ClassVar
from typing import List

import torch
from torch.utils.data import Dataset

import src.augmentations as aug
import src.datasets.amigos as amigos
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
