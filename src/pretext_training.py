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
from functools import partial
import tqdm


path_to_model: str = "/Users/joergsimon/Documents/phd/HELENA/ssl-ecg/model_data"


@dataclass
class PretextParams:
    batch_size:int = 32  # don't forget we get all the agumentations per example
    num_workers:int = 1#2
    epochs:int = 200
    valid_size = 0.2
    test_size = 0.1


def train_pretext_full_config():
    default_params = PretextParams()
    model = EcgNetwork(len(d.AugmentationsPretextDataset.STD_AUG) + 1, 5)
    optimizer = torch.optim.Adam(model.parameters(), 0.00001)
    #per_task_criterion = nn.BCELoss()
    criterion = nn.BCELoss()#AvaragePretextLoss(per_task_criterion, [1.]*5)
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model = model.cuda()
    train_pretext(model, optimizer, criterion, train_on_gpu, default_params)


def train_pretext(model, optimizer, criterion, train_on_gpu: bool, p: PretextParams):
    dataset_array = []
    for ds_type in d.ds_to_constructor.keys():
        ds_obj = d.ds_to_constructor[ds_type](d.DataConstants.basepath)
        dataset_array.append(ds_obj)
    #dataset = torch.utils.data.ConcatDataset(dataset_array)
    dataset = d.CombinedECGDatasets(d.ds_to_constructor.keys(), d.DataConstants.basepath)
    # dataset = amigos.ECGAmigosCachedWindowsDataset(d.DataConstants.basepath)
    dataset = d.AugmentationsPretextDataset(dataset, d.AugmentationsPretextDataset.STD_AUG)

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

    valid_loss_min = np.Inf  # track change in validation loss

    ltv = partial(labels_to_vec, n_tasks=len(dataset.augmentations) + 1, debug_ltv=False)  # just a shortcut

    for e in tqdm.tqdm(range(p.epochs)):

        train_loss = 0.0
        valid_loss = 0.0

        def iterate_batches(loader, loss_type):
            nonlocal train_loss, valid_loss
            for i_batch, (data, labels) in enumerate(tqdm.tqdm(loader, leave=False)):
                total_loss = None
                for aug_data, aug_labels in zip(data, labels):
                    if aug_data.shape[0] != p.batch_size:
                        print('skipping too small batch')
                        continue  # if not full batch, just continue
                    if train_on_gpu:
                        aug_data = aug_data.cuda()
                    if len(aug_data.shape) == 2:
                        aug_data = aug_data.unsqueeze(axis=1).float()
                    # clear the gradients of all optimized variables
                    optimizer.zero_grad()
                    tasks_out, _ = model(aug_data)
                    lbls = ltv(aug_labels)
                    tasks_out = tasks_out.squeeze().T
                    task_loss = criterion(tasks_out, lbls)
                    if total_loss is None:
                        total_loss = task_loss
                    else:
                        total_loss += task_loss
                if total_loss is None:
                    print('skipping too small batch')
                    continue
                total_loss.backward()
                optimizer.step()
                total_loss = total_loss / len(labels)
                # update training loss
                l = total_loss.item() * len(data) * data[0].size(0)
                if loss_type == 'valid':
                    valid_loss += l
                else:
                    train_loss += l

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
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            e, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.cnn.state_dict(), f'{path_to_model}/model_embedding.pt')
            for i, t in enumerate(model.task_heads):
                torch.save(t.state_dict(), f'{path_to_model}/task_head_{i}.pt')
            valid_loss_min = valid_loss
