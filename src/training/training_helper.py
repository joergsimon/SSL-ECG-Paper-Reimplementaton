import numpy as np
import src.utils as utils
import torch
from ray import tune
import os.path
from enum import Enum

class TrainingMode(Enum):
    TRAIN = 0
    VALID = 2


def iterate_batches(loader, optimizer, scaler, batch_size, train_on_gpu: bool, compute_loss, mode: TrainingMode):
    total_loss = 0.0
    total_accuracy = 0.0
    total_accuracy_list = []

    for i_batch, (data, labels) in enumerate(utils.pbar(loader, leave=False)):
        # labels = torch.stack(labels, dim=1).float()
        if data.shape[0] != batch_size:
            # print('skipping too small batch')
            continue  # if not full batch, just continue
        if train_on_gpu:
            data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss, accuracy = compute_loss(data, labels)
        if mode == TrainingMode.TRAIN:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_accuracy_list.append(accuracy.item())
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        # print(f'all accuracies: {total_accuracy_list}')
        # print(f'list based accuracy: {np.mean(np.array(total_accuracy_list))}')
        # print(f'sum of accuracy: {total_accuracy.item()}')
        # print(f'sum of accuracy normalised by loader length: {total_accuracy.item() / len(loader)}')
    return total_loss, total_accuracy


def std_train_loop(epochs, batch_size, train_loader, valid_loader, model, optimizer, schedulder, use_scaler, compute_loss_and_accuracy, save_model, train_on_gpu: bool, use_tune:bool):
    valid_loss_min = np.Inf  # track change in validation loss

    scaler = None
    if use_scaler:
        scaler = torch.cuda.amp.GradScaler()

    for e in utils.pbar(range(epochs)):

        for param_group in optimizer.param_groups:
            print(f'lr for epoch {e} ', param_group['lr'])

        train_loss = 0.0
        valid_loss = 0.0

        train_accuracy = 0.0
        valid_accuracy = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        l, a = iterate_batches(train_loader, optimizer, scaler, batch_size, train_on_gpu, compute_loss_and_accuracy, TrainingMode.TRAIN)
        train_loss += l
        train_accuracy += a

        ######################
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            l, a = iterate_batches(valid_loader, optimizer, scaler, batch_size, train_on_gpu, compute_loss_and_accuracy, TrainingMode.VALID)
        valid_loss += l
        valid_accuracy += a

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)

        train_accuracy = train_accuracy / len(train_loader)
        valid_accuracy = valid_accuracy / len(valid_loader)
        
        if schedulder is not None:
            schedulder.step()

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
            save_model()
            valid_loss_min = valid_loss
