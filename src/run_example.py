from src.model import EcgNetwork
import src.data as dta
import torch
import torch.nn as nn
import os
import src.constants as c

def create_model(dataset, target_id):
    does_not_matter = len(dta.AugmentationsPretextDataset.STD_AUG) + 1
    ecg_net = EcgNetwork(does_not_matter, dataset.target_size)
    model = ecg_net.emotion_head
    embedder = ecg_net.cnn
    train_on_gpu = torch.cuda.is_available()
    device = 'cuda' if train_on_gpu else 'cpu'
    state_dict_embeddings = torch.load(f'{c.model_base_path}model_embedding.pt', map_location=torch.device(device))
    embedder.load_state_dict(state_dict_embeddings)
    state_dict_model = torch.load(f'{c.model_base_path}tuned/tuned_for_{target_id}.pt',
                                  map_location=torch.device(device))
    model.load_state_dict(state_dict_model)

    return embedder, model


def run_example(target_dataset: dta.DataSets=[], target_id=None):
    dataset = dta.ds_to_constructor[target_dataset](dta.DataConstants.basepath)
    embedder, model = create_model(dataset, target_id)
    data = dataset[200]
    emb = embedder(data[0])
    res = model(emb)

    print('data: ')
    print(data[0])
    print('calculated: ')
    print(res)
    print('expected: ')
    print(data[1])