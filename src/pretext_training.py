import os
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import src.data as dta
import src.utils as utils
from src.constants import Constants as c
from src.model import EcgNetwork, labels_to_vec

path_to_model: str = c.model_base_path


@dataclass
class PretextParams:
    batch_size:int = 32  # don't forget we get all the agumentations per example
    num_workers:int = 3
    epochs:int = 100
    valid_size = 0.2
    test_size = 0.1
    pin_memory = True


good_params_for_single_run = {
    "pretext": {
        "batch_size": 16,
        "adam": {"lr": 0.000154582}
    }
}


def train_pretext_tune_task(num_samples=10, max_num_epochs=100, gpus_per_trial=0.5):
    config = {
        "pretext": {
            "batch_size": tune.choice([8, 16, 32]),
            "adam": {"lr": tune.loguniform(9e-5, 2e-3)}
        }
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_pretext_full_config),
        resources_per_trial={"cpu": 3, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        log_to_file=True
    )

    dfs = result.trial_dataframes
    if len(dfs) > 0:
        if 'accuracy' in dfs.values()[0].columns:
            ax = None  # This plots everything on the same plot
            for d in dfs.values():
                if 'accuracy' in d.columns:
                    ax = d.accuracy.plot(ax=ax, legend=False)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            plt.savefig('overview-accuracy-pretext.png')
            plt.show()
        if 'loss' in dfs.values()[0].columns:
            ax = None  # This plots everything on the same plot
            for d in dfs.values():
                if 'loss' in d.columns:
                    ax = d.loss.plot(ax=ax, legend=False)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("loss")
            plt.savefig('overview-loss-pretext.png')
            plt.show()

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    best_trained_model = EcgNetwork(len(dta.AugmentationsPretextDataset.STD_AUG) + 1, 5)
    if torch.cuda.is_available():
        best_trained_model = best_trained_model.cuda()
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    print('------------------------------------------------------------------------------')
    print('               Saving best model from hyperparam search                       ')
    print('               for use in finetuning                                          ')
    print('------------------------------------------------------------------------------')
    torch.save(best_trained_model.cnn.state_dict(), f'{path_to_model}/model_embedding.pt')
    for i, t in enumerate(best_trained_model.task_heads):
        torch.save(t.state_dict(), f'{path_to_model}/task_head_{i}.pt')


def train_pretext_full_config(hyperparams_config, checkpoint_dir=None, use_tune=True):
    p = PretextParams()
    p.batch_size = hyperparams_config['pretext']['batch_size']
    model = EcgNetwork(len(dta.AugmentationsPretextDataset.STD_AUG) + 1, 5)
    optimizer = torch.optim.Adam(model.parameters(), hyperparams_config['pretext']['adam']['lr'])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    pos_weight = torch.tensor([0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    train_pretext(model, optimizer, criterion, train_on_gpu, p, use_tune=use_tune)


def train_pretext(model, optimizer, criterion, train_on_gpu: bool, p: PretextParams, use_tune=True):

    # we have convolutions here so allow the automatic optimization to ramp it up
    torch.backends.cudnn.benchmark = True

    dataset = dta.CombinedECGDatasets(dta.ds_to_constructor.keys(), dta.DataConstants.basepath)
    dataset = dta.AugmentationsPretextDataset(dataset, dta.AugmentationsPretextDataset.STD_AUG)

    num_train = len(dataset)
    indices = list(range(num_train))
    train_idx, valid_idx, test_idx = utils.random_splits(indices, p.test_size, p.valid_size)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampeler = SubsetRandomSampler(test_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(dataset, batch_size=p.batch_size,
                              sampler=train_sampler, num_workers=p.num_workers, pin_memory=p.pin_memory)
    valid_loader = DataLoader(dataset, batch_size=p.batch_size,
                              sampler=valid_sampler, num_workers=p.num_workers)
    test_loader = DataLoader(dataset, batch_size=p.batch_size,
                             sampler=test_sampeler, num_workers=p.num_workers)

    valid_loss_min = np.Inf  # track change in validation loss

    ltv = partial(labels_to_vec, n_tasks=len(dataset.augmentations) + 1, debug_ltv=False)  # just a shortcut

    for e in utils.pbar(range(p.epochs)):

        train_loss = 0.0
        valid_loss = 0.0

        train_accuracy = 0.0
        valid_accuracy = 0.0

        def iterate_batches(loader, loss_type):
            nonlocal train_loss, valid_loss, valid_accuracy, train_accuracy
            for i_batch, (data, labels) in enumerate(utils.pbar(loader, leave=False)):
                total_loss = None
                total_accuracy = None
                for aug_data, aug_labels in zip(data, labels):
                    if aug_data.shape[0] != p.batch_size:
                        #print('skipping too small batch')
                        continue  # if not full batch, just continue
                    if train_on_gpu:
                        aug_data = aug_data.cuda()
                        aug_labels = aug_labels.cuda()
                    if len(aug_data.shape) == 2:
                        aug_data = aug_data.unsqueeze(axis=1).float()
                    # clear the gradients of all optimized variables
                    for param in model.parameters():
                        param.grad = None
                    # optimizer.zero_grad()
                    tasks_out, _ = model(aug_data)
                    lbls = ltv(aug_labels)
                    if train_on_gpu:
                        lbls = lbls.cuda()
                    tasks_out = tasks_out.squeeze().T
                    task_loss = criterion(tasks_out, lbls)

                    predicted = torch.argmax(tasks_out, dim=1)
                    accuracy = torch.sum(predicted == aug_labels).type(torch.float) / aug_labels.shape[0]

                    task_loss.backward()
                    optimizer.step()

                    total_loss = utils.assign(total_loss, task_loss)
                    total_accuracy = utils.assign(total_accuracy, accuracy)
                if total_loss is None:
                    #print('skipping too small batch')
                    continue
                # total_loss.backward()
                # optimizer.step()
                total_loss = total_loss / len(labels)
                total_accuracy = total_accuracy / len(labels)
                # update training loss
                l = total_loss.item()
                a = total_accuracy.item()
                if loss_type == 'valid':
                    valid_loss += l
                    valid_accuracy += a
                else:
                    train_loss += l
                    train_accuracy += a

        ###################
        # train the model #
        ###################
        model.train()
        iterate_batches(train_loader, 'train')

        ######################
        # validate the model #
        ######################
        model.eval()
        iterate_batches(valid_loader, 'valid')

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)

        train_accuracy = train_accuracy / len(train_loader)
        valid_accuracy = valid_accuracy / len(valid_loader)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n\t\tTraining Accuracy: {:.3f} \tValidation Accuracy: {:.3f}'.format(
            e, train_loss, valid_loss, train_accuracy, valid_accuracy))

        if use_tune:
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=e) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (model.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=valid_loss, accuracy=valid_accuracy)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.cnn.state_dict(), f'{path_to_model}/model_embedding.pt')
            for i, t in enumerate(model.task_heads):
                torch.save(t.state_dict(), f'{path_to_model}/task_head_{i}.pt')
            valid_loss_min = valid_loss
