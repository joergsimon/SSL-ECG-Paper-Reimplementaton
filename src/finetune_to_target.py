import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import src.data as dta
import src.training.training_helper as th
import src.utils as utils
from src.constants import Constants as c
from src.model import EcgNetwork, EcgAmigosHead

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
        "adam": {"lr": 0.000128268},
        "scheduler": {"decay": 0.9}
    }
}


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.LambdaLR(optimizer, lr_lambda, last_epoch)


def rm_nan(labels, column):
    col = labels[:, column]
    col[col != col] = 0.0  # remove nans
    return col


def binary_labels(labels):
    # most old fashiond literature just does a high/low valance/arousal classification (so 4 classes)
    # let's first also do this old fashioned task
    labels[:, 0] = rm_nan(labels, 0)
    labels[:, 1] = rm_nan(labels, 1)
    labels = (labels > 5.0).type(torch.FloatTensor)
    return labels


def train_finetune_tune_task(target_dataset: dta.DataSets, target_id, num_samples=10, max_num_epochs=200, gpus_per_trial=0.5):
    config = {
        "finetune": {
            "batch_size": tune.choice([8, 16, 32, 64, 128]),
            "adam": {"lr": tune.loguniform(5e-4, 1e-1)},
            "scheduler": {
                "type": tune.choice(['decay', 'cosine_w_restarts', 'none']),
                "decay": tune.loguniform(0.99, 0.90),
                "warmup": tune.randint(5, 15),
                "cycles": tune.randint(1, 4)
            }
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

    utils.print_ray_overview(result, 'finetuning')

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # dataset = dta.ds_to_constructor[target_dataset](dta.DataConstants.basepath)
    best_trained_model = EcgAmigosHead(2) # = EcgAmigosHead(dataset.target_size)

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        best_trained_model = best_trained_model.cuda()
        if torch.cuda.device_count() > 1:
            best_trained_model = nn.DataParallel(best_trained_model)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    device = 'cuda' if train_on_gpu else 'cpu'
    model_state, optimizer_state = torch.load(checkpoint_path, map_location=device)
    best_trained_model = utils.save_load_state_dict(best_trained_model, model_state)

    print('------------------------------------------------------------------------------')
    print('               Saving best model from hyperparam search                       ')
    print('               for later use                                                  ')
    print('------------------------------------------------------------------------------')
    torch.save(best_trained_model.state_dict(), f'{basepath_to_tuned_model}tuned_for_{target_id}.pt')


def finetune_to_target_full_config(hyperparams_config, checkpoint_dir=None, target_dataset: dta.DataSets=[], target_id=None, use_tune=True):
    default_params = TuningParams()
    default_params.batch_size = hyperparams_config['finetune']['batch_size']
    train_on_gpu = torch.cuda.is_available()

    dataset = dta.ds_to_constructor[target_dataset](dta.DataConstants.basepath)

    does_not_matter = len(dta.AugmentationsPretextDataset.STD_AUG) + 1
    ecg_net = EcgNetwork(does_not_matter, dataset.target_size)
    model = EcgAmigosHead(2)
    model.debug_values = False
    embedder = ecg_net.cnn
    device = 'cuda' if train_on_gpu else 'cpu'
    state_dict = torch.load(f'{path_to_src_model}model_embedding.pt', map_location=torch.device(device))
    embedder.load_state_dict(state_dict)
    # for p in embedder.parameters():
    #     p.requires_grad = False

    def check_zero_grad(gradient):
        data = gradient.data.clone().detach()
        debug_is_zero = torch.sum(data) == 0
        if debug_is_zero:
            print(gradient)

    embedder.conv_1.weight.register_hook(check_zero_grad)

    dataset = dta.EmbeddingsDataset(embedder, dataset, False, dta.EmbeddingsDataset.path_to_cache, target_id,  train_on_gpu) # set should cache to false so we see an effect on finetuning
    lr = hyperparams_config['finetune']['adam']['lr']



    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': embedder.parameters(), 'lr': lr/10}
    ], lr)
    scheduler_info = hyperparams_config['finetune']['scheduler']
    if scheduler_info['type'] == 'none':
        schedulder = None
    elif scheduler_info['type'] == 'decay':
        schedulder = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                            gamma=scheduler_info['decay'])
    elif scheduler_info['type'] == 'cosine_w_restarts':
        warmup = scheduler_info['warmup']
        training = default_params.epochs - warmup
        cycles = scheduler_info['cycles']
        schedulder = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, warmup, training, cycles)

    criterion = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        device = 'cuda' if train_on_gpu else 'cpu'
        model_state, optimizer_state = torch.load(checkpoint, map_location=device)
        model = utils.save_load_state_dict(model, model_state)
        optimizer.load_state_dict(optimizer_state)

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    finetune(model, optimizer, schedulder, criterion, dataset, train_on_gpu, default_params, target_id, use_tune)


def finetune(model, optimizer, schedulder, criterion, dataset, train_on_gpu: bool, p: TuningParams, target_id, use_tune: bool):

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
        # latent = cnn(data).squeeze()
        # y_prime = head(latent)
        y_prime = model(data).squeeze()
        y = binary_labels(labels)
        if train_on_gpu:
            y = y.cuda()
        loss = criterion(y_prime, y)
        # print('loss', loss)

        y_sigm = torch.sigmoid(y_prime).detach()
        predicted = (y_sigm > 0.5).type(torch.FloatTensor)  # torch.argmax(l_prime, dim=1)
        if train_on_gpu:
            predicted = predicted.cuda()
        same = predicted == y
        same_sum = torch.sum(same).type(torch.float)
        accuracy = same_sum / torch.numel(same)
        return loss, accuracy

    def save_model():
        torch.save(model.state_dict(), f'{basepath_to_tuned_model}tuned_for_{target_id}.pt')

    use_scaler = True
    th.std_train_loop(p.epochs, p.batch_size, train_loader, valid_loader, model, optimizer, schedulder, use_scaler, compute_loss_and_accuracy, save_model, train_on_gpu, use_tune)