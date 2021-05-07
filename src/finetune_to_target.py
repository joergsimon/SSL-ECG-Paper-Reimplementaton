from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataclasses import dataclass
import torch
import numpy as np
import src.data as d
import src.datasets.amigos as amigos
import src.utils as utils
from src.model import EcgNetwork, AvaragePretextLoss, labels_to_vec
import torch
import torch.nn as nn
import src.training.training_helper as th
from functools import partial
import tqdm


path_to_src_model: str = "/Users/joergsimon/Documents/phd/HELENA/ssl-ecg/model_data"
basepath_to_tuned_model: str = "/Users/joergsimon/Documents/phd/HELENA/ssl-ecg/model_data/tuned"


@dataclass
class TuningParams:
    batch_size:int = 32
    num_workers:int = 1#2
    epochs:int = 200
    valid_size = 0.2
    test_size = 0.1


def finetune_to_target_full_config(target_dataset: d.DataSets, target_id):
    default_params = TuningParams()
    train_on_gpu = torch.cuda.is_available()

    dataset = d.ds_to_constructor[target_dataset](d.DataConstants.basepath)

    does_not_matter = len(d.AugmentationsPretextDataset.STD_AUG) + 1
    ecg_net = EcgNetwork(does_not_matter, dataset.target_size)
    model = ecg_net.emotion_head
    embedder = ecg_net.cnn
    device = 'cuda' if train_on_gpu else 'cpu'
    state_dict = torch.load(f'{path_to_src_model}/model_embedding.pt', map_location=torch.device(device))
    embedder.load_state_dict(state_dict)
    for p in embedder.parameters():
        p.requires_grad = False

    dataset = d.EmbeddingsDataset(embedder, dataset, True, d.EmbeddingsDataset.path_to_cache, target_id,  train_on_gpu)
    optimizer = torch.optim.Adam(model.parameters(), 0.00001)
    criterion = nn.NLLLoss()
    if train_on_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
    finetune(model, optimizer, criterion, dataset, train_on_gpu, default_params, target_id)


def finetune(model, optimizer, criterion, dataset, train_on_gpu: bool, p: TuningParams, target_id):

    num_train = len(dataset)
    indices = list(range(num_train))
    train_idx, valid_idx, test_idx = utils.random_splits(indices, p.test_size, p.valid_size)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampeler = SubsetRandomSampler(test_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(dataset, batch_size=p.batch_size,
                              sampler=train_sampler, num_workers=p.num_workers)
    valid_loader = DataLoader(dataset, batch_size=p.batch_size,
                              sampler=valid_sampler, num_workers=p.num_workers)
    test_loader = DataLoader(dataset, batch_size=p.batch_size,
                             sampler=test_sampeler, num_workers=p.num_workers)

    def compute_loss_and_accuracy(data, labels):
        l_prime = model(data).squeeze()
        # for now, we just try to predict valance
        valances = labels[0]
        valances[valances != valances] = 0 # remove nans
        valances = valances.type(torch.LongTensor) # we quantisize
        if train_on_gpu:
            valances = valances.cuda()
        if torch.any(valances < 0):
            print(labels[0])
            print(valances)
            valances[valances < 0] = 0
        loss = criterion(l_prime, valances)

        predicted = torch.argmax(l_prime)
        accuracy = torch.sum(predicted == valances)/valances.shape[0]
        return loss, accuracy

    def save_model():
        torch.save(model.state_dict(), f'{basepath_to_tuned_model}/tuned_for_{target_id}.pt')

    th.std_train_loop(p.epochs, p.batch_size, train_loader, valid_loader, model, optimizer, compute_loss_and_accuracy, save_model, train_on_gpu)