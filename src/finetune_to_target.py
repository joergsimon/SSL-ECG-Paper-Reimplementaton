import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import src.data as dta
import src.training.training_helper as th
import src.utils as utils
from src.constants import Constants as c
from src.model import EcgNetwork

path_to_src_model: str = c.model_base_path
basepath_to_tuned_model: str = c.model_base_path + "tuned/"


@dataclass
class TuningParams:
    batch_size:int = 32
    num_workers:int = 1#2
    epochs:int = 200
    valid_size = 0.2
    test_size = 0.1


good_params_for_single_run = {
    "finetune": {
        "batch_size": 16,
        "adam": {"lr": 0.000128268}
    }
}


def train_finetune_tune_task(target_dataset: dta.DataSets, target_id, num_samples=10, max_num_epochs=200, gpus_per_trial=0.5):
    config = {
        "finetune": {
            "batch_size": tune.choice([8, 16, 32]),
            "adam": {"lr": tune.loguniform(1e-4, 1e-1)}
        }
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(finetune_to_target_full_config, target_dataset=target_dataset, target_id=target_id),
        resources_per_trial={"cpu": 3, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    dfs = result.trial_dataframes
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Accuracy")
    plt.savefig('overview-finetuning.png')
    plt.show()

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    dataset = dta.ds_to_constructor[target_dataset](dta.DataConstants.basepath)
    does_not_matter = len(dta.AugmentationsPretextDataset.STD_AUG) + 1
    best_trained_model = EcgNetwork(does_not_matter, dataset.target_size).emotion_head

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        best_trained_model = best_trained_model.cuda()
        if torch.cuda.device_count() > 1:
            best_trained_model = nn.DataParallel(best_trained_model)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    print('------------------------------------------------------------------------------')
    print('               Saving best model from hyperparam search                       ')
    print('               for later use                                                  ')
    print('------------------------------------------------------------------------------')
    torch.save(best_trained_model.state_dict(), f'{basepath_to_tuned_model}tuned_for_{target_id}.pt')


def finetune_to_target_full_config(hyperparams_config, checkpoint_dir=None, target_dataset: dta.DataSets=[], target_id=None):
    default_params = TuningParams()
    default_params.batch_size = hyperparams_config['finetune']['batch_size']
    train_on_gpu = torch.cuda.is_available()

    dataset = dta.ds_to_constructor[target_dataset](dta.DataConstants.basepath)

    does_not_matter = len(dta.AugmentationsPretextDataset.STD_AUG) + 1
    ecg_net = EcgNetwork(does_not_matter, dataset.target_size)
    model = ecg_net.emotion_head
    model.debug_values = True
    embedder = ecg_net.cnn
    device = 'cuda' if train_on_gpu else 'cpu'
    state_dict = torch.load(f'{path_to_src_model}model_embedding.pt', map_location=torch.device(device))
    embedder.load_state_dict(state_dict)
    for p in embedder.parameters():
        p.requires_grad = False

    dataset = dta.EmbeddingsDataset(embedder, dataset, True, dta.EmbeddingsDataset.path_to_cache, target_id,  train_on_gpu)
    optimizer = torch.optim.Adam(model.parameters(), hyperparams_config['finetune']['adam']['lr'])
    criterion = nn.CrossEntropyLoss()

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
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
                              sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(dataset, batch_size=p.batch_size,
                              sampler=valid_sampler, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=p.batch_size,
                             sampler=test_sampeler, num_workers=0)

    print(model)

    def compute_loss_and_accuracy(data, labels):
        l_prime = model(data).squeeze()
        # for now, we just try to predict valance
        # print(labels.shape)
        # print("ooo")
        # print(labels)
        valances = labels[:,0]
        valances[valances != valances] = 0 # remove nans
        valances = valances.type(torch.LongTensor) # we quantisize
        if train_on_gpu:
            valances = valances.cuda()
        if torch.any(valances < 0):
            print(labels[0])
            print(valances)
            valances[valances < 0] = 0
        print('data_prime ', l_prime)
        print('valances', valances)
        loss = criterion(l_prime, valances)
        print('loss', loss)

        predicted = torch.argmax(l_prime, dim=1)
        accuracy = torch.sum(predicted == valances).type(torch.float)/valances.shape[0]
        return loss, accuracy

    def save_model():
        torch.save(model.state_dict(), f'{basepath_to_tuned_model}tuned_for_{target_id}.pt')

    th.std_train_loop(p.epochs, p.batch_size, train_loader, valid_loader, model, optimizer, compute_loss_and_accuracy, save_model, train_on_gpu)